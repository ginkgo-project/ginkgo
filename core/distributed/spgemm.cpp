// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/index_map.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/allocator.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/distributed/matrix_kernels.hpp"
#include "ginkgo/core/distributed/matrix.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace {


// Whether cuSPARSE provides a 64-bit index spgemm, which it only does from
// CUDA 13 on.
#if defined(GKO_CUDA_TOOLKIT_VERSION_MAJOR) && \
    (GKO_CUDA_TOOLKIT_VERSION_MAJOR >= 13)
constexpr bool cuda_has_int64_spgemm = true;
#else
constexpr bool cuda_has_int64_spgemm = false;
#endif


GKO_REGISTER_OPERATION(convert_ptrs_to_idxs, components::convert_ptrs_to_idxs);
GKO_REGISTER_OPERATION(separate_local_nonlocal_columns,
                       distributed_matrix::separate_local_nonlocal_columns);


// Merges the diagonal (local_mtx) and off-diagonal (non_local_mtx) Csr blocks
// of one operand into a single Csr whose column indices are global, using the
// operand's index map to map both blocks' local columns back to global ones.
// The returned matrix is created on `exec`.
//
// Note: within each row the merged entries are not sorted by column index (the
// off-diagonal columns may lie left or right of the diagonal ones); callers
// that require column-sorted input must sort the result themselves.
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<matrix::Csr<ValueType, GlobalIndexType>> merge_to_global_csr(
    std::shared_ptr<const Executor> exec, const LinOp* local_mtx,
    const LinOp* non_local_mtx,
    const index_map<LocalIndexType, GlobalIndexType>& imap)
{
    using local_csr = matrix::Csr<ValueType, LocalIndexType>;
    using global_csr = matrix::Csr<ValueType, GlobalIndexType>;

    auto local = as<local_csr>(local_mtx);
    auto non_local = as<local_csr>(non_local_mtx);

    auto host = exec->get_master();

    // The row_ptrs and values are read by the merge loop below, which runs on
    // the host, so the blocks themselves have to be available there.
    auto local_host = make_temporary_clone(host, local);
    auto non_local_host = make_temporary_clone(host, non_local);

    auto nrows = local_host->get_size()[0];
    auto ncols = static_cast<size_type>(imap.get_global_size());

    auto local_nnz = local_host->get_num_stored_elements();
    auto non_local_nnz = non_local_host->get_num_stored_elements();
    auto total_nnz = local_nnz + non_local_nnz;

    // The column indices, on the other hand, are only fed to map_to_global,
    // which runs on the index map's executor. View them on the blocks' own
    // executor instead of copying them through the host; the clone below is a
    // no-op in the usual case where both live on the same executor. Only the
    // mapped global columns are brought back for the merge.
    auto imap_exec = imap.get_executor();
    const auto local_cols_view =
        gko::detail::array_const_cast(make_const_array_view(
            local->get_executor(), local_nnz, local->get_const_col_idxs()));
    const auto non_local_cols_view = gko::detail::array_const_cast(
        make_const_array_view(non_local->get_executor(), non_local_nnz,
                              non_local->get_const_col_idxs()));
    auto local_cols = make_temporary_clone(imap_exec, &local_cols_view);
    auto non_local_cols = make_temporary_clone(imap_exec, &non_local_cols_view);
    const auto global_local_cols_dev =
        imap.map_to_global(*local_cols, index_space::local);
    const auto global_non_local_cols_dev =
        imap.map_to_global(*non_local_cols, index_space::non_local);
    auto global_local_cols = make_temporary_clone(host, &global_local_cols_dev);
    auto global_non_local_cols =
        make_temporary_clone(host, &global_non_local_cols_dev);

    // Build merged CSR arrays on host
    auto merged_row_ptrs = array<GlobalIndexType>(host, nrows + 1);
    auto merged_col_idxs = array<GlobalIndexType>(host, total_nnz);
    auto merged_values = array<ValueType>(host, total_nnz);

    auto local_row_ptrs = local_host->get_const_row_ptrs();
    auto non_local_row_ptrs = non_local_host->get_const_row_ptrs();
    auto local_vals = local_host->get_const_values();
    auto non_local_vals = non_local_host->get_const_values();
    auto global_local_cols_ptr = global_local_cols->get_const_data();
    auto global_non_local_cols_ptr = global_non_local_cols->get_const_data();

    size_type out_idx = 0;
    for (size_type row = 0; row < nrows; ++row) {
        merged_row_ptrs.get_data()[row] = static_cast<GlobalIndexType>(out_idx);

        // Copy local entries for this row
        for (auto k = local_row_ptrs[row]; k < local_row_ptrs[row + 1]; ++k) {
            merged_col_idxs.get_data()[out_idx] = global_local_cols_ptr[k];
            merged_values.get_data()[out_idx] = local_vals[k];
            ++out_idx;
        }

        // Copy non-local entries for this row
        for (auto k = non_local_row_ptrs[row]; k < non_local_row_ptrs[row + 1];
             ++k) {
            merged_col_idxs.get_data()[out_idx] = global_non_local_cols_ptr[k];
            merged_values.get_data()[out_idx] = non_local_vals[k];
            ++out_idx;
        }
    }
    merged_row_ptrs.get_data()[nrows] = static_cast<GlobalIndexType>(out_idx);

    // Create the merged CSR on the original executor.
    return global_csr::create(
        exec, dim<2>{nrows, ncols}, std::move(merged_values),
        std::move(merged_col_idxs), std::move(merged_row_ptrs));
}


}  // namespace


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::multiply(
    ptr_param<const Matrix> b, ptr_param<Matrix> c) const
{
    const auto* b_ptr = b.get();
    auto* c_ptr = c.get();

    auto exec = this->get_executor();
    auto host = exec->get_master();
    auto comm = this->get_communicator();
    auto rank = comm.rank();
    auto nprocs = comm.size();

    if (this->get_row_partition() == nullptr) {
        GKO_INVALID_STATE(
            "distributed spgemm requires a row partition on the left operand, "
            "which is only set when the matrix is filled by read_distributed");
    }
    if (b_ptr->get_row_partition() == nullptr) {
        GKO_INVALID_STATE(
            "distributed spgemm requires a row partition on the right operand, "
            "which is only set when the matrix is filled by read_distributed");
    }
    GKO_ASSERT_CONFORMANT(this, b_ptr);

    // The local product below runs in LocalIndexType, so 64-bit local indices
    // are only usable where the backend's spgemm supports them: rocSPARSE has
    // no 64-bit spgemm at all, and cuSPARSE only gained one in CUDA 13. Reject
    // those combinations here, before any communication happens, instead of
    // letting the vendor library fail with an opaque status code deep inside
    // the local product. Every rank runs the same check, so they all throw
    // together and none is left waiting in a collective.
    if (sizeof(LocalIndexType) > 4 &&
        (dynamic_cast<const HipExecutor*>(exec.get()) != nullptr ||
         (!cuda_has_int64_spgemm &&
          dynamic_cast<const CudaExecutor*>(exec.get()) != nullptr))) {
        throw NotSupported(__FILE__, __LINE__, __func__,
                           "64-bit LocalIndexType (rocSPARSE has no 64-bit "
                           "spgemm, cuSPARSE requires CUDA 13)");
    }

    // A's column partition must equal B's row partition. Partition::equals
    // short-circuits when both operands share the same partition object, which
    // is the common case.
    auto a_col_partition = this->imap_.get_partition();
    auto b_row_partition = b_ptr->get_row_partition();
    if (!a_col_partition->equals(*b_row_partition)) {
        GKO_INVALID_STATE(
            "distributed spgemm requires the column partition of the left "
            "operand to match the row partition of the right operand");
    }

    // Merge A and B to global-column CSR
    auto a_merged =
        merge_to_global_csr<ValueType, LocalIndexType, GlobalIndexType>(
            host, this->get_diag_matrix().get(),
            this->get_off_diag_matrix().get(), this->imap_);
    auto b_merged =
        merge_to_global_csr<ValueType, LocalIndexType, GlobalIndexType>(
            host, b_ptr->get_diag_matrix().get(),
            b_ptr->get_off_diag_matrix().get(), b_ptr->imap_);

    // The rows of B this rank has to supply are exactly the send indices of
    // A's row gatherer: A's non-local columns are B's remote rows, which only
    // holds because A's column partition is required to equal B's row
    // partition, checked above. They are exchanged by resizing that row
    // gatherer's communicator to the length of each row, so the communication
    // pattern and its ordering never leave the communicator.
    auto b_row_ptrs = b_merged->get_const_row_ptrs();
    auto b_col_idxs = b_merged->get_const_col_idxs();
    auto b_vals = b_merged->get_const_values();

    auto b_local_nrows = static_cast<GlobalIndexType>(b_merged->get_size()[0]);

    // nnz count of each row to supply, in the row gatherer's send order.
    // The send indices live on the row gatherer's executor, but they are read
    // by the packing loops below, which run on the host.
    auto a_row_gatherer = this->row_gatherer_;
    auto num_send_rows = a_row_gatherer->get_num_send_idxs();
    const auto send_rows_view = gko::detail::array_const_cast(
        make_const_array_view(a_row_gatherer->get_executor(), num_send_rows,
                              a_row_gatherer->get_const_send_idxs()));
    auto send_rows = make_temporary_clone(host, &send_rows_view);
    auto send_rows_ptr = send_rows->get_const_data();

    // could be combined with CSR/COO row gather -> get nnz from gathered mat
    vector<int> send_nnz_counts(num_send_rows, host);
    for (int i = 0; i < num_send_rows; ++i) {
        const auto local_row = static_cast<GlobalIndexType>(send_rows_ptr[i]);
        send_nnz_counts[i] =
            static_cast<int>(b_row_ptrs[local_row + 1] - b_row_ptrs[local_row]);
    }
    auto total_send_nnz =
        std::accumulate(send_nnz_counts.begin(), send_nnz_counts.end(), 0);

    // Pack the column indices and values of the requested rows.
    // @todo: Essentially row gather on CSR
    vector<GlobalIndexType> send_col_idxs(host);
    vector<ValueType> send_vals(host);
    send_col_idxs.reserve(total_send_nnz);
    send_vals.reserve(total_send_nnz);
    for (int i = 0; i < num_send_rows; ++i) {
        const auto local_row = static_cast<GlobalIndexType>(send_rows_ptr[i]);
        auto row_begin = b_row_ptrs[local_row];
        auto row_end = b_row_ptrs[local_row + 1];
        for (auto k = row_begin; k < row_end; ++k) {
            send_col_idxs.push_back(b_col_idxs[k]);
            send_vals.push_back(b_vals[k]);
        }
    }

    // Resizing exchanges the per-row nnz counts and yields a communicator that
    // moves that many entries for each row.
    auto [resized_comm, recv_nnz_counts] =
        this->row_gatherer_->get_collective_communicator()->resize(
            exec, std::vector(send_nnz_counts.begin(), send_nnz_counts.end()));

    // Exchange column indices
    vector<GlobalIndexType> recv_col_idxs(resized_comm->get_recv_size(), host);
    resized_comm
        ->i_all_to_all_v(host, send_col_idxs.data(), recv_col_idxs.data())
        .wait();

    // Exchange values
    vector<ValueType> recv_vals(resized_comm->get_recv_size(), host);
    resized_comm->i_all_to_all_v(host, send_vals.data(), recv_vals.data())
        .wait();

    // One received row per non-local column of A, in A's non-local order.
    auto num_recv_rows = this->imap_.get_non_local_size();
    auto total_recv_nnz = resized_comm->get_recv_size();

    auto a_nnz = a_merged->get_num_stored_elements();
    auto a_col_idxs = a_merged->get_const_col_idxs();
    auto a_nrows = static_cast<GlobalIndexType>(a_merged->get_size()[0]);

    // B_augmented has the local B rows [0, b_local_nrows) followed by the
    // received remote rows; remote row i sits at augmented index
    // b_local_nrows + i.
    auto b_aug_nrows =
        b_local_nrows + static_cast<GlobalIndexType>(num_recv_rows);

    // B_augmented row_ptrs: local-row lengths from b_merged, then remote-row
    // nnz counts.
    // All of this until the manipulation of A starts is essentially
    // appending two CSR matrices (or rather appending rows)
    vector<GlobalIndexType> b_aug_row_ptrs(b_aug_nrows + 1, 0, host);
    std::copy(b_row_ptrs, b_row_ptrs + b_local_nrows + 1,
              b_aug_row_ptrs.begin());
    std::inclusive_scan(recv_nnz_counts.begin(), recv_nnz_counts.end(),
                        b_aug_row_ptrs.begin() + b_local_nrows + 1, std::plus{},
                        b_aug_row_ptrs[b_local_nrows]);

    auto b_aug_nnz = b_aug_row_ptrs[b_aug_nrows];
    // sanity check
    GKO_THROW_IF_INVALID(
        b_aug_nnz == b_merged->get_num_stored_elements() + total_recv_nnz,
        "Invalid B augmented nnz");

    vector<GlobalIndexType> b_aug_col_idxs(b_aug_nnz, host);
    vector<ValueType> b_aug_vals(b_aug_nnz, host);

    // Copy local B data
    auto b_local_nnz = b_row_ptrs[b_local_nrows] - b_row_ptrs[0];
    GKO_THROW_IF_INVALID(b_local_nnz == b_merged->get_num_stored_elements(),
                         "Invalid B local nnz");

    std::copy(b_col_idxs, b_col_idxs + b_local_nnz, b_aug_col_idxs.begin());
    std::copy(recv_col_idxs.begin(), recv_col_idxs.end(),
              b_aug_col_idxs.begin() + b_local_nnz);

    std::copy(b_vals, b_vals + b_local_nnz, b_aug_vals.begin());
    std::copy(recv_vals.begin(), recv_vals.end(),
              b_aug_vals.begin() + b_local_nnz);

    // Remap A's global columns to B_augmented row indices via A's imap_
    // (combined index space); row_ptrs and values carry over from a_merged.
    auto a_row_ptrs = a_merged->get_row_ptrs();
    auto a_vals = a_merged->get_values();

    auto a_col_idxs_host =
        array<GlobalIndexType>(host, a_col_idxs, a_col_idxs + a_nnz);
    auto a_col_idxs_dev = make_temporary_clone(exec, &a_col_idxs_host);
    auto a_remap_local =
        this->imap_.map_to_local(*a_col_idxs_dev, index_space::combined);

    // The local spgemm runs with LocalIndexType (32-bit) indices, which every
    // backend supports (rocSPARSE has no 64-bit spgemm), so B_augmented's
    // global columns have to be brought into a local space.
    //
    // That space is an index map over B's column partition, which is exactly
    // what C's column index map has to be: its local part is the columns this
    // rank owns and its non-local part the remote ones, so after the product
    // the diagonal/off-diagonal split is a comparison against the local size
    // and the off-diagonal columns are already non-local indices. Building it
    // here rather than from C's off-diagonal entries afterwards also makes it
    // far cheaper -- the input is B_augmented's columns rather than one entry
    // per off-diagonal nonzero of C. index_map ignores the columns this rank
    // owns, so they can be passed in unfiltered.
    auto b_col_partition = b_ptr->imap_.get_partition();

    auto b_aug_col_idxs_arr = array<GlobalIndexType>(
        exec, b_aug_col_idxs.begin(), b_aug_col_idxs.end());
    c_ptr->imap_ = index_map<LocalIndexType, GlobalIndexType>(
        exec, b_col_partition, rank, b_aug_col_idxs_arr);
    const auto num_local_cols = c_ptr->imap_.get_local_size();
    const auto num_column_space =
        num_local_cols + c_ptr->imap_.get_non_local_size();
    auto b_aug_col_local =
        c_ptr->imap_.map_to_local(b_aug_col_idxs_arr, index_space::combined);

    // The local product is stored with LocalIndexType row pointers and column
    // indices, so its nnz and dimensions must fit that type.
    const auto local_index_max =
        static_cast<size_type>(std::numeric_limits<LocalIndexType>::max());
    if (a_nnz > local_index_max ||
        static_cast<size_type>(b_aug_nnz) > local_index_max ||
        num_column_space > local_index_max ||
        static_cast<size_type>(b_aug_nrows) > local_index_max) {
        throw OverflowError(__FILE__, __LINE__, "LocalIndexType");
    }

    // Local SpGEMM (LocalIndexType). a_remapped's columns index B_augmented's
    // rows; B_augmented's columns are C's combined column index space.
    array<LocalIndexType> a_row_ptrs_local(host);
    a_row_ptrs_local = make_array_view(host, a_nrows + 1, a_row_ptrs);
    array<LocalIndexType> b_aug_row_ptrs_local(host);
    b_aug_row_ptrs_local =
        make_array_view(host, b_aug_row_ptrs.size(), b_aug_row_ptrs.data());

    using local_csr = matrix::Csr<ValueType, LocalIndexType>;
    auto a_remapped = local_csr::create(
        exec,
        dim<2>{static_cast<size_type>(a_nrows),
               static_cast<size_type>(b_aug_nrows)},
        array<ValueType>(exec, a_vals, a_vals + a_nnz),
        std::move(a_remap_local), std::move(a_row_ptrs_local));

    auto b_augmented = local_csr::create(
        exec, dim<2>{static_cast<size_type>(b_aug_nrows), num_column_space},
        array<ValueType>(exec, b_aug_vals.begin(), b_aug_vals.end()),
        std::move(b_aug_col_local), std::move(b_aug_row_ptrs_local));

    auto c_local = local_csr::create(
        exec, dim<2>{static_cast<size_type>(a_nrows), num_column_space});
    if (a_nrows > 0) {
        // The local csr::spgemm requires column-sorted inputs.
        a_remapped->sort_by_column_index();
        b_augmented->sort_by_column_index();
        c_local = a_remapped->multiply(b_augmented);
    }

    // Reassemble the output entirely on the executor. c_local's columns are
    // already in C's combined index space, so the diagonal/off-diagonal split
    // is a comparison against the number of locally owned columns and the
    // off-diagonal columns come out as non-local indices needing no further
    // mapping. C's column index map was built before the product, so nothing
    // here has to look at the partition at all.
    //
    // This region is profiled as three phases: "split" and "read" are device
    // kernels, and "gatherer" is the only communication -- the collective
    // communicator's size exchange plus RowGatherer's i_all_to_all_v.
    // Already checked for null at function entry, and this is a const member
    // function, so the partition cannot have changed since.
    auto a_row_partition = this->get_row_partition();
    auto c_num_local_rows = c_local->get_size()[0];
    auto c_nnz = c_local->get_num_stored_elements();

    // Per-nonzero local row index. Columns and values are read-only, so view
    // them in place.
    array<LocalIndexType> local_rows(exec, c_nnz);
    exec->run(make_convert_ptrs_to_idxs(c_local->get_const_row_ptrs(),
                                        c_num_local_rows,
                                        local_rows.get_data()));
    auto combined_cols = make_array_view(exec, c_nnz, c_local->get_col_idxs());
    auto col_vals = make_array_view(exec, c_nnz, c_local->get_values());

    array<LocalIndexType> diag_rows(exec);
    array<LocalIndexType> diag_cols(exec);
    array<ValueType> diag_vals(exec);
    array<LocalIndexType> off_rows(exec);
    array<LocalIndexType> off_local_cols(exec);
    array<ValueType> off_vals(exec);
    exec->run(make_separate_local_nonlocal_columns(
        local_rows, combined_cols, col_vals,
        static_cast<LocalIndexType>(num_local_cols), diag_rows, diag_cols,
        diag_vals, off_rows, off_local_cols, off_vals));

    c_ptr->set_size(
        dim<2>{a_row_partition->get_size(), b_col_partition->get_size()});
    c_ptr->row_partition_ = a_row_partition;

    const auto num_remote_cols = c_ptr->imap_.get_non_local_size();
    device_matrix_data<ValueType, LocalIndexType> diag_data{
        exec, dim<2>{c_num_local_rows, num_local_cols}, std::move(diag_rows),
        std::move(diag_cols), std::move(diag_vals)};
    device_matrix_data<ValueType, LocalIndexType> off_data{
        exec, dim<2>{c_num_local_rows, num_remote_cols}, std::move(off_rows),
        std::move(off_local_cols), std::move(off_vals)};
    as<ReadableFromMatrixData<ValueType, LocalIndexType>>(c_ptr->diag_mtx_)
        ->read(std::move(diag_data));
    as<ReadableFromMatrixData<ValueType, LocalIndexType>>(c_ptr->off_diag_mtx_)
        ->read(std::move(off_data));

    c_ptr->row_gatherer_ = RowGatherer<LocalIndexType>::create(
        c_ptr->row_gatherer_->get_executor(),
        c_ptr->row_gatherer_->get_collective_communicator()
            ->create_with_same_type(comm, &c_ptr->imap_),
        c_ptr->imap_);
}


#define GKO_DECLARE_DISTRIBUTED_MATRIX_MULTIPLY(ValueType, LocalIndexType, \
                                                GlobalIndexType)           \
    void Matrix<ValueType, LocalIndexType, GlobalIndexType>::multiply(     \
        ptr_param<const Matrix> b, ptr_param<Matrix> c) const
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_DISTRIBUTED_MATRIX_MULTIPLY);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
