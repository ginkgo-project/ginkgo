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
GKO_REGISTER_OPERATION(separate_diag_off_diag_local_rows,
                       distributed_matrix::separate_diag_off_diag_local_rows);
GKO_REGISTER_OPERATION(compress_columns, distributed_matrix::compress_columns);


// Turns a per-rank counts vector into the corresponding exclusive-prefix-sum
// offsets vector (as used for all_to_all_v send/recv displacement arrays).
gko::vector<int> counts_to_offsets(const gko::vector<int>& counts)
{
    gko::vector<int> offsets(counts.size(), counts.get_allocator());
    std::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), 0);
    return offsets;
}


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

    // A's imap_ gives the remote global column indices (= B rows) this rank
    // needs and their owner ranks.
    const auto& remote_target_ids = this->imap_.get_remote_target_ids();
    const auto& remote_global_idxs = this->imap_.get_remote_global_idxs();
    auto n_remote_targets = static_cast<int>(remote_target_ids.get_size());

    // These index-map arrays live on the index map's executor; copy them to
    // host for the packing loops below.
    auto remote_target_ids_host =
        make_temporary_clone(host, &remote_target_ids);
    auto remote_offsets_host =
        make_temporary_clone(host, &remote_global_idxs.get_offsets());
    auto remote_target_ids_ptr = remote_target_ids_host->get_const_data();
    auto remote_offsets_ptr = remote_offsets_host->get_const_data();
    array<GlobalIndexType> remote_flat_host(host,
                                            remote_global_idxs.get_size());
    if (remote_global_idxs.get_size() > 0) {
        host->copy_from(remote_global_idxs.get_executor(),
                        remote_global_idxs.get_size(),
                        remote_global_idxs.get_const_flat_data(),
                        remote_flat_host.get_data());
    }
    auto remote_flat = remote_flat_host.get_const_data();

    // Per-rank count of B rows to request from each owner.
    gko::vector<int> send_row_counts(nprocs, 0, host);
    for (int t = 0; t < n_remote_targets; ++t) {
        auto target_rank = remote_target_ids_ptr[t];
        auto seg_begin = remote_offsets_ptr[t];
        auto seg_end = remote_offsets_ptr[t + 1];
        send_row_counts[target_rank] = static_cast<int>(seg_end - seg_begin);
    }
    auto send_row_offsets = counts_to_offsets(send_row_counts);

    // Exchange request counts
    gko::vector<int> recv_row_counts(nprocs, 0, host);
    comm.all_to_all(host, send_row_counts.data(), 1, recv_row_counts.data(), 1);

    auto recv_row_offsets = counts_to_offsets(recv_row_counts);
    int total_recv_rows =
        recv_row_offsets[nprocs - 1] + recv_row_counts[nprocs - 1];

    // Pack the requested global row indices, grouped by owner rank.
    auto total_send_rows = static_cast<int>(remote_global_idxs.get_size());
    gko::vector<GlobalIndexType> send_row_idxs(total_send_rows, host);
    for (int t = 0; t < n_remote_targets; ++t) {
        auto target_rank = remote_target_ids_ptr[t];
        auto seg_begin = remote_offsets_ptr[t];
        auto seg_end = remote_offsets_ptr[t + 1];
        std::copy(remote_flat + seg_begin, remote_flat + seg_end,
                  send_row_idxs.data() + send_row_offsets[target_rank]);
    }

    // Exchange the actual row index requests
    gko::vector<GlobalIndexType> recv_row_idxs(total_recv_rows, host);
    comm.all_to_all_v(host, send_row_idxs.data(), send_row_counts.data(),
                      send_row_offsets.data(), recv_row_idxs.data(),
                      recv_row_counts.data(), recv_row_offsets.data());

    // Reply to each requested row with its nnz count, then its column indices
    // and values.
    auto b_row_ptrs = b_merged->get_const_row_ptrs();
    auto b_col_idxs = b_merged->get_const_col_idxs();
    auto b_vals = b_merged->get_const_values();

    auto b_local_nrows = static_cast<GlobalIndexType>(b_merged->get_size()[0]);

    // Map each requested global B-row id to this rank's local row via A's
    // imap_ (invalid_index if not owned here). map_to_local runs on the index
    // map's executor, so place the input there and bring the result to host.
    auto recv_row_idxs_host = array<GlobalIndexType>(
        host, recv_row_idxs.begin(), recv_row_idxs.end());
    auto recv_row_idxs_dev = make_temporary_clone(exec, &recv_row_idxs_host);
    const auto recv_local_rows_map =
        this->imap_.map_to_local(*recv_row_idxs_dev, index_space::local);
    auto recv_local_rows_map_host =
        make_temporary_clone(host, &recv_local_rows_map);
    auto recv_local_rows_map_ptr = recv_local_rows_map_host->get_const_data();

    // Local row and nnz count for each requested row.
    gko::vector<GlobalIndexType> recv_local_rows(total_recv_rows, host);
    gko::vector<int> send_nnz_counts(total_recv_rows, host);
    for (int i = 0; i < total_recv_rows; ++i) {
        auto mapped = recv_local_rows_map_ptr[i];
        auto local_row = (mapped == invalid_index<LocalIndexType>())
                             ? GlobalIndexType{-1}
                             : static_cast<GlobalIndexType>(mapped);
        recv_local_rows[i] = local_row;
        if (local_row >= 0 && local_row < b_local_nrows) {
            send_nnz_counts[i] = static_cast<int>(b_row_ptrs[local_row + 1] -
                                                  b_row_ptrs[local_row]);
        } else {
            send_nnz_counts[i] = 0;
        }
    }

    // Exchange nnz counts (send/recv roles swapped: we reply for the rows
    // others requested from us).
    gko::vector<int> recv_nnz_counts(total_send_rows, host);
    comm.all_to_all_v(host, send_nnz_counts.data(), recv_row_counts.data(),
                      recv_row_offsets.data(), recv_nnz_counts.data(),
                      send_row_counts.data(), send_row_offsets.data());

    // Per-rank data (nnz) counts to send and receive.
    gko::vector<int> send_data_counts(nprocs, 0, host);
    for (int r = 0; r < nprocs; ++r) {
        for (int i = recv_row_offsets[r];
             i < recv_row_offsets[r] + recv_row_counts[r]; ++i) {
            send_data_counts[r] += send_nnz_counts[i];
        }
    }
    auto send_data_offsets = counts_to_offsets(send_data_counts);
    int total_send_data =
        send_data_offsets[nprocs - 1] + send_data_counts[nprocs - 1];

    gko::vector<int> recv_data_counts(nprocs, 0, host);
    for (int r = 0; r < nprocs; ++r) {
        for (int i = send_row_offsets[r];
             i < send_row_offsets[r] + send_row_counts[r]; ++i) {
            recv_data_counts[r] += recv_nnz_counts[i];
        }
    }
    auto recv_data_offsets = counts_to_offsets(recv_data_counts);
    int total_recv_data =
        recv_data_offsets[nprocs - 1] + recv_data_counts[nprocs - 1];

    // Pack the column indices and values of the requested rows.
    gko::vector<GlobalIndexType> send_col_idxs(total_send_data, host);
    gko::vector<ValueType> send_vals(total_send_data, host);
    {
        int pos = 0;
        for (int i = 0; i < total_recv_rows; ++i) {
            auto local_row = recv_local_rows[i];
            if (local_row >= 0 && local_row < b_local_nrows) {
                auto row_begin = b_row_ptrs[local_row];
                auto row_end = b_row_ptrs[local_row + 1];
                for (auto k = row_begin; k < row_end; ++k) {
                    send_col_idxs[pos] = b_col_idxs[k];
                    send_vals[pos] = b_vals[k];
                    ++pos;
                }
            }
        }
    }

    // Exchange column indices
    gko::vector<GlobalIndexType> recv_col_idxs(total_recv_data, host);
    comm.all_to_all_v(host, send_col_idxs.data(), send_data_counts.data(),
                      send_data_offsets.data(), recv_col_idxs.data(),
                      recv_data_counts.data(), recv_data_offsets.data());

    // Exchange values
    gko::vector<ValueType> recv_vals(total_recv_data, host);
    comm.all_to_all_v(host, send_vals.data(), send_data_counts.data(),
                      send_data_offsets.data(), recv_vals.data(),
                      recv_data_counts.data(), recv_data_offsets.data());

    auto a_nnz = a_merged->get_num_stored_elements();
    auto a_col_idxs = a_merged->get_const_col_idxs();
    auto a_nrows = static_cast<GlobalIndexType>(a_merged->get_size()[0]);

    // B_augmented has the local B rows [0, b_local_nrows) followed by the
    // received remote rows; remote row i sits at augmented index
    // b_local_nrows + i.
    auto b_aug_nrows =
        b_local_nrows + static_cast<GlobalIndexType>(total_send_rows);
    auto b_ncols = static_cast<GlobalIndexType>(b_merged->get_size()[1]);

    // B_augmented row_ptrs: local-row lengths from b_merged, then remote-row
    // nnz counts.
    gko::vector<GlobalIndexType> b_aug_row_ptrs(b_aug_nrows + 1, 0, host);
    for (GlobalIndexType row = 0; row < b_local_nrows; ++row) {
        b_aug_row_ptrs[row + 1] =
            static_cast<GlobalIndexType>(b_row_ptrs[row + 1] - b_row_ptrs[0]);
    }
    for (int i = 0; i < total_send_rows; ++i) {
        b_aug_row_ptrs[b_local_nrows + i + 1] =
            static_cast<GlobalIndexType>(recv_nnz_counts[i]);
    }
    // Prefix-sum the remote-row counts into offsets.
    std::partial_sum(b_aug_row_ptrs.begin() + b_local_nrows,
                     b_aug_row_ptrs.end(),
                     b_aug_row_ptrs.begin() + b_local_nrows);

    auto b_aug_nnz = b_aug_row_ptrs[b_aug_nrows];
    gko::vector<GlobalIndexType> b_aug_col_idxs(b_aug_nnz, host);
    gko::vector<ValueType> b_aug_vals(b_aug_nnz, host);

    // Copy local B data
    auto b_local_nnz = b_row_ptrs[b_local_nrows] - b_row_ptrs[0];
    for (GlobalIndexType k = 0; k < static_cast<GlobalIndexType>(b_local_nnz);
         ++k) {
        b_aug_col_idxs[k] = b_col_idxs[k];
        b_aug_vals[k] = b_vals[k];
    }

    // Copy remote B data into augmented rows b_local_nrows + i.
    {
        int data_pos = 0;
        for (int i = 0; i < total_send_rows; ++i) {
            auto nnz = recv_nnz_counts[i];
            auto row_start = b_aug_row_ptrs[b_local_nrows + i];
            for (int k = 0; k < nnz; ++k) {
                b_aug_col_idxs[row_start + k] = recv_col_idxs[data_pos + k];
                b_aug_vals[row_start + k] = recv_vals[data_pos + k];
            }
            data_pos += nnz;
        }
    }

    // Remap A's global columns to B_augmented row indices via A's imap_
    // (combined index space); row_ptrs and values carry over from a_merged.
    auto a_row_ptrs = a_merged->get_const_row_ptrs();
    auto a_vals = a_merged->get_const_values();

    auto a_col_idxs_host =
        array<GlobalIndexType>(host, a_col_idxs, a_col_idxs + a_nnz);
    auto a_col_idxs_dev = make_temporary_clone(exec, &a_col_idxs_host);
    const auto a_remap_local =
        this->imap_.map_to_local(*a_col_idxs_dev, index_space::combined);
    auto a_remap_local_host = make_temporary_clone(host, &a_remap_local);
    auto a_remap_local_ptr = a_remap_local_host->get_const_data();

    // Every column of A has to resolve in the combined index space; an
    // unmapped one would be written out as invalid_index and then read as an
    // out-of-bounds row of B_augmented, so check it rather than corrupt the
    // product. The flag keeps the branch out of the assignment path.
    gko::vector<LocalIndexType> a_remap_col_idxs(a_nnz, host);
    bool all_cols_mapped = true;
    for (size_type k = 0; k < a_nnz; ++k) {
        const auto mapped = a_remap_local_ptr[k];
        if (mapped == invalid_index<LocalIndexType>()) {
            all_cols_mapped = false;
        }
        a_remap_col_idxs[k] = mapped;
    }
    if (!all_cols_mapped) {
        GKO_INVALID_STATE(
            "a column of the left operand could not be mapped into the "
            "combined index space of its column index map");
    }

    // The local spgemm runs with LocalIndexType (32-bit) indices, which every
    // backend supports (rocSPARSE has no 64-bit spgemm). Compress B_augmented's
    // global columns to a compact local space on the executor;
    // b_aug_distinct_cols maps each compact index back to its global column for
    // the reassemble below.
    array<GlobalIndexType> b_aug_cols_dev(exec, b_aug_col_idxs.begin(),
                                          b_aug_col_idxs.end());
    array<LocalIndexType> b_aug_col_local(exec);
    array<GlobalIndexType> b_aug_distinct_cols(exec);
    exec->run(make_compress_columns(b_aug_cols_dev, b_aug_col_local,
                                    b_aug_distinct_cols));
    const auto num_distinct_cols = b_aug_distinct_cols.get_size();

    // The local product is stored with LocalIndexType row pointers and column
    // indices, so its nnz and dimensions must fit that type.
    const auto local_index_max =
        static_cast<size_type>(std::numeric_limits<LocalIndexType>::max());
    if (a_nnz > local_index_max ||
        static_cast<size_type>(b_aug_nnz) > local_index_max ||
        num_distinct_cols > local_index_max ||
        static_cast<size_type>(b_aug_nrows) > local_index_max) {
        throw OverflowError(__FILE__, __LINE__, "LocalIndexType");
    }

    // Local SpGEMM (LocalIndexType). a_remapped's columns index B_augmented's
    // rows; B_augmented's columns are the compact column space.
    gko::vector<LocalIndexType> a_row_ptrs_local(a_nrows + 1, host);
    for (GlobalIndexType i = 0; i <= a_nrows; ++i) {
        a_row_ptrs_local[i] = static_cast<LocalIndexType>(a_row_ptrs[i]);
    }
    gko::vector<LocalIndexType> b_aug_row_ptrs_local(b_aug_row_ptrs.size(),
                                                     host);
    for (size_type i = 0; i < b_aug_row_ptrs.size(); ++i) {
        b_aug_row_ptrs_local[i] =
            static_cast<LocalIndexType>(b_aug_row_ptrs[i]);
    }

    using local_csr = matrix::Csr<ValueType, LocalIndexType>;
    auto a_remapped =
        local_csr::create(exec,
                          dim<2>{static_cast<size_type>(a_nrows),
                                 static_cast<size_type>(b_aug_nrows)},
                          array<ValueType>(exec, a_vals, a_vals + a_nnz),
                          array<LocalIndexType>(exec, a_remap_col_idxs.begin(),
                                                a_remap_col_idxs.end()),
                          array<LocalIndexType>(exec, a_row_ptrs_local.begin(),
                                                a_row_ptrs_local.end()));

    auto b_augmented = local_csr::create(
        exec, dim<2>{static_cast<size_type>(b_aug_nrows), num_distinct_cols},
        array<ValueType>(exec, b_aug_vals.begin(), b_aug_vals.end()),
        std::move(b_aug_col_local),
        array<LocalIndexType>(exec, b_aug_row_ptrs_local.begin(),
                              b_aug_row_ptrs_local.end()));

    auto c_local = local_csr::create(
        exec, dim<2>{static_cast<size_type>(a_nrows), num_distinct_cols});
    if (a_nrows > 0) {
        // The local csr::spgemm requires column-sorted inputs.
        a_remapped->sort_by_column_index();
        b_augmented->sort_by_column_index();
        c_local = a_remapped->multiply(b_augmented);
    }

    // Reassemble the output entirely on the executor. c_local has this rank's
    // local rows with compact columns (indices into b_aug_distinct_cols);
    // separate_diag_off_diag_local_rows resolves them to global columns and
    // splits into c's diagonal (local columns) and off-diagonal (remote
    // columns) blocks, from which we build c's column index map and row
    // gatherer -- all with device kernels, avoiding any host round-trip.
    // Already checked for null at function entry, and this is a const member
    // function, so the partition cannot have changed since.
    auto a_row_partition = this->get_row_partition();
    auto b_col_partition = b_ptr->imap_.get_partition();
    auto c_num_local_rows = c_local->get_size()[0];
    auto c_nnz = c_local->get_num_stored_elements();

    // Per-nonzero local row index. Columns/values are read-only, so view them
    // in place; col_map turns c_local's compact columns back into global ones.
    array<LocalIndexType> local_rows(exec, c_nnz);
    exec->run(make_convert_ptrs_to_idxs(c_local->get_const_row_ptrs(),
                                        c_num_local_rows,
                                        local_rows.get_data()));
    auto compact_cols = make_array_view(exec, c_nnz, c_local->get_col_idxs());
    auto col_vals = make_array_view(exec, c_nnz, c_local->get_values());

    // Split into diagonal (local columns) and off-diagonal (global columns)
    // COO blocks on the executor, keeping the local rows. b_aug_distinct_cols
    // is the compact-to-global column map produced by the compression above.
    array<LocalIndexType> diag_rows(exec);
    array<LocalIndexType> diag_cols(exec);
    array<ValueType> diag_vals(exec);
    array<LocalIndexType> off_rows(exec);
    array<GlobalIndexType> off_global_cols(exec);
    array<ValueType> off_vals(exec);
    auto b_col_partition_dev = make_temporary_clone(exec, b_col_partition);
    exec->run(make_separate_diag_off_diag_local_rows(
        local_rows, compact_cols, b_aug_distinct_cols, col_vals,
        b_col_partition_dev.get(), rank, diag_rows, diag_cols, diag_vals,
        off_rows, off_global_cols, off_vals));

    // C's column index map from the off-diagonal global columns, then map them
    // to non-local indices.
    c_ptr->set_size(
        dim<2>{a_row_partition->get_size(), b_col_partition->get_size()});
    c_ptr->imap_ = index_map<LocalIndexType, GlobalIndexType>(
        exec, b_col_partition, rank, off_global_cols);
    c_ptr->row_partition_ = a_row_partition;
    auto off_local_cols =
        c_ptr->imap_.map_to_local(off_global_cols, index_space::non_local);

    const auto num_local_cols =
        static_cast<size_type>(b_col_partition->get_part_size(rank));
    const auto num_remote_cols =
        c_ptr->imap_.get_remote_global_idxs().get_size();
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
