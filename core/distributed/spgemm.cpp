// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/index_map.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/components/format_conversion_kernels.hpp"
#include "core/distributed/matrix_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "ginkgo/core/distributed/matrix.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace {


GKO_REGISTER_OPERATION(local_spgemm, csr::spgemm);
GKO_REGISTER_OPERATION(convert_ptrs_to_idxs, components::convert_ptrs_to_idxs);
GKO_REGISTER_OPERATION(separate_diag_off_diag_local_rows,
                       distributed_matrix::separate_diag_off_diag_local_rows);


// Turns a per-rank counts vector into the corresponding exclusive-prefix-sum
// offsets vector (as used for all_to_all_v send/recv displacement arrays).
std::vector<int> counts_to_offsets(const std::vector<int>& counts)
{
    std::vector<int> offsets(counts.size(), 0);
    for (std::size_t r = 1; r < counts.size(); ++r) {
        offsets[r] = offsets[r - 1] + counts[r - 1];
    }
    return offsets;
}


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

    // Clone the input blocks to host for index manipulation.
    auto local_host = make_temporary_clone(host, local);
    auto non_local_host = make_temporary_clone(host, non_local);

    auto nrows = local_host->get_size()[0];
    auto ncols = static_cast<size_type>(imap.get_global_size());

    auto local_nnz = local_host->get_num_stored_elements();
    auto non_local_nnz = non_local_host->get_num_stored_elements();
    auto total_nnz = local_nnz + non_local_nnz;

    // Collect the block column indices on host.
    auto local_cols_host = array<LocalIndexType>(host, local_nnz);
    if (local_nnz > 0) {
        std::copy_n(local_host->get_const_col_idxs(), local_nnz,
                    local_cols_host.get_data());
    }
    auto non_local_cols_host = array<LocalIndexType>(host, non_local_nnz);
    if (non_local_nnz > 0) {
        std::copy_n(non_local_host->get_const_col_idxs(), non_local_nnz,
                    non_local_cols_host.get_data());
    }

    // map_to_global runs on the index map's executor: place the inputs there
    // and bring the mapped global columns back to host for the merge below.
    auto imap_exec = imap.get_executor();
    auto local_cols = make_temporary_clone(imap_exec, &local_cols_host);
    auto non_local_cols = make_temporary_clone(imap_exec, &non_local_cols_host);
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
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::spgemm(
    ptr_param<const Matrix> b, ptr_param<Matrix> c) const
{
    const auto* b_ptr = b.get();
    auto* c_ptr = c.get();

    using global_csr = matrix::Csr<ValueType, GlobalIndexType>;

    auto exec = this->get_executor();
    auto host = exec->get_master();
    auto comm = this->get_communicator();
    auto rank = comm.rank();
    auto nprocs = comm.size();

    GKO_ASSERT(this->get_row_partition() != nullptr);
    GKO_ASSERT_CONFORMANT(this, b_ptr);

    if (b_ptr->get_row_partition()) {
        auto a_col_partition = this->imap_.get_partition();
        auto b_row_partition_check = b_ptr->get_row_partition();
        auto a_col_partition_host = a_col_partition->clone(host);
        auto b_row_partition_host_check = b_row_partition_check->clone(host);
        bool partitions_match =
            a_col_partition_host->get_size() ==
                b_row_partition_host_check->get_size() &&
            a_col_partition_host->get_num_ranges() ==
                b_row_partition_host_check->get_num_ranges();
        if (partitions_match) {
            auto num_ranges = a_col_partition_host->get_num_ranges();
            auto a_bounds = a_col_partition_host->get_range_bounds();
            auto b_bounds = b_row_partition_host_check->get_range_bounds();
            auto a_ids = a_col_partition_host->get_part_ids();
            auto b_ids = b_row_partition_host_check->get_part_ids();
            for (size_type r = 0; r < num_ranges && partitions_match; ++r) {
                partitions_match = partitions_match &&
                                   a_bounds[r] == b_bounds[r] &&
                                   a_ids[r] == b_ids[r];
            }
            partitions_match = partitions_match &&
                               a_bounds[num_ranges] == b_bounds[num_ranges];
        }
        GKO_ASSERT(partitions_match);
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
    std::vector<int> send_row_counts(nprocs, 0);
    for (int t = 0; t < n_remote_targets; ++t) {
        auto target_rank = remote_target_ids_ptr[t];
        auto seg_begin = remote_offsets_ptr[t];
        auto seg_end = remote_offsets_ptr[t + 1];
        send_row_counts[target_rank] = static_cast<int>(seg_end - seg_begin);
    }
    auto send_row_offsets = counts_to_offsets(send_row_counts);

    // Exchange request counts
    std::vector<int> recv_row_counts(nprocs, 0);
    comm.all_to_all(host, send_row_counts.data(), 1, recv_row_counts.data(), 1);

    auto recv_row_offsets = counts_to_offsets(recv_row_counts);
    int total_recv_rows =
        recv_row_offsets[nprocs - 1] + recv_row_counts[nprocs - 1];

    // Pack the requested global row indices, grouped by owner rank.
    auto total_send_rows = static_cast<int>(remote_global_idxs.get_size());
    std::vector<GlobalIndexType> send_row_idxs(total_send_rows);
    for (int t = 0; t < n_remote_targets; ++t) {
        auto target_rank = remote_target_ids_ptr[t];
        auto seg_begin = remote_offsets_ptr[t];
        auto seg_end = remote_offsets_ptr[t + 1];
        std::copy(remote_flat + seg_begin, remote_flat + seg_end,
                  send_row_idxs.data() + send_row_offsets[target_rank]);
    }

    // Exchange the actual row index requests
    std::vector<GlobalIndexType> recv_row_idxs(total_recv_rows);
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
    std::vector<GlobalIndexType> recv_local_rows(total_recv_rows);
    std::vector<int> send_nnz_counts(total_recv_rows);
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
    std::vector<int> recv_nnz_counts(total_send_rows);
    comm.all_to_all_v(host, send_nnz_counts.data(), recv_row_counts.data(),
                      recv_row_offsets.data(), recv_nnz_counts.data(),
                      send_row_counts.data(), send_row_offsets.data());

    // Per-rank data (nnz) counts to send and receive.
    std::vector<int> send_data_counts(nprocs, 0);
    for (int r = 0; r < nprocs; ++r) {
        for (int i = recv_row_offsets[r];
             i < recv_row_offsets[r] + recv_row_counts[r]; ++i) {
            send_data_counts[r] += send_nnz_counts[i];
        }
    }
    auto send_data_offsets = counts_to_offsets(send_data_counts);
    int total_send_data =
        send_data_offsets[nprocs - 1] + send_data_counts[nprocs - 1];

    std::vector<int> recv_data_counts(nprocs, 0);
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
    std::vector<GlobalIndexType> send_col_idxs(total_send_data);
    std::vector<ValueType> send_vals(total_send_data);
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
    std::vector<GlobalIndexType> recv_col_idxs(total_recv_data);
    comm.all_to_all_v(host, send_col_idxs.data(), send_data_counts.data(),
                      send_data_offsets.data(), recv_col_idxs.data(),
                      recv_data_counts.data(), recv_data_offsets.data());

    // Exchange values
    std::vector<ValueType> recv_vals(total_recv_data);
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
    std::vector<GlobalIndexType> b_aug_row_ptrs(b_aug_nrows + 1, 0);
    for (GlobalIndexType row = 0; row < b_local_nrows; ++row) {
        b_aug_row_ptrs[row + 1] =
            static_cast<GlobalIndexType>(b_row_ptrs[row + 1] - b_row_ptrs[0]);
    }
    for (int i = 0; i < total_send_rows; ++i) {
        b_aug_row_ptrs[b_local_nrows + i + 1] =
            static_cast<GlobalIndexType>(recv_nnz_counts[i]);
    }
    // Prefix-sum the remote-row counts into offsets.
    for (GlobalIndexType row = b_local_nrows; row < b_aug_nrows; ++row) {
        b_aug_row_ptrs[row + 1] += b_aug_row_ptrs[row];
    }

    auto b_aug_nnz = b_aug_row_ptrs[b_aug_nrows];
    std::vector<GlobalIndexType> b_aug_col_idxs(b_aug_nnz);
    std::vector<ValueType> b_aug_vals(b_aug_nnz);

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

    std::vector<GlobalIndexType> a_remap_col_idxs(a_nnz);
    for (size_type k = 0; k < a_nnz; ++k) {
        GKO_ASSERT(a_remap_local_ptr[k] != invalid_index<LocalIndexType>());
        a_remap_col_idxs[k] =
            static_cast<GlobalIndexType>(a_remap_local_ptr[k]);
    }

    // Local SpGEMM
    auto a_remapped = global_csr::create(
        exec,
        dim<2>{static_cast<size_type>(a_nrows),
               static_cast<size_type>(b_aug_nrows)},
        array<ValueType>(exec, a_vals, a_vals + a_nnz),
        array<GlobalIndexType>(exec, a_remap_col_idxs.begin(),
                               a_remap_col_idxs.end()),
        array<GlobalIndexType>(exec, a_row_ptrs, a_row_ptrs + a_nrows + 1));

    auto b_augmented = global_csr::create(
        exec,
        dim<2>{static_cast<size_type>(b_aug_nrows),
               static_cast<size_type>(b_ncols)},
        array<ValueType>(exec, b_aug_vals.begin(), b_aug_vals.end()),
        array<GlobalIndexType>(exec, b_aug_col_idxs.begin(),
                               b_aug_col_idxs.end()),
        array<GlobalIndexType>(exec, b_aug_row_ptrs.begin(),
                               b_aug_row_ptrs.end()));

    auto c_local =
        global_csr::create(exec, dim<2>{static_cast<size_type>(a_nrows),
                                        static_cast<size_type>(b_ncols)});
    if (a_nrows > 0) {
        // The local csr::spgemm requires column-sorted inputs.
        a_remapped->sort_by_column_index();
        b_augmented->sort_by_column_index();
        exec->run(make_local_spgemm(a_remapped.get(), b_augmented.get(),
                                    c_local.get()));
    }

    // Reassemble the output entirely on the executor. c_local has this rank's
    // local rows with global columns; separate_diag_off_diag_local_rows splits
    // it into c's diagonal (local columns) and off-diagonal (remote columns)
    // blocks, from which we build c's column index map and row gatherer -- all
    // with device kernels, avoiding any host round-trip of the nonzeros.
    auto a_row_partition = this->get_row_partition();
    GKO_ASSERT(a_row_partition != nullptr);
    auto b_col_partition = b_ptr->imap_.get_partition();
    auto c_num_local_rows = c_local->get_size()[0];
    auto c_nnz = c_local->get_num_stored_elements();

    // Per-nonzero local row index (int64 ptrs -> int32 idxs). The split kernel
    // reads c_local's columns/values read-only, so view them in place rather
    // than copying.
    array<LocalIndexType> local_rows(exec, c_nnz);
    exec->run(make_convert_ptrs_to_idxs(c_local->get_const_row_ptrs(),
                                        c_num_local_rows,
                                        local_rows.get_data()));
    auto global_cols = make_array_view(exec, c_nnz, c_local->get_col_idxs());
    auto global_vals = make_array_view(exec, c_nnz, c_local->get_values());

    // Split into diagonal (local columns) and off-diagonal (global columns)
    // COO blocks on the executor, keeping the local rows.
    array<LocalIndexType> diag_rows(exec);
    array<LocalIndexType> diag_cols(exec);
    array<ValueType> diag_vals(exec);
    array<LocalIndexType> off_rows(exec);
    array<GlobalIndexType> off_global_cols(exec);
    array<ValueType> off_vals(exec);
    auto b_col_partition_dev = make_temporary_clone(exec, b_col_partition);
    exec->run(make_separate_diag_off_diag_local_rows(
        local_rows, global_cols, global_vals, b_col_partition_dev.get(), rank,
        diag_rows, diag_cols, diag_vals, off_rows, off_global_cols, off_vals));

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


#define GKO_DECLARE_DISTRIBUTED_MATRIX_SPGEMM(ValueType, LocalIndexType, \
                                              GlobalIndexType)           \
    void Matrix<ValueType, LocalIndexType, GlobalIndexType>::spgemm(     \
        ptr_param<const Matrix> b, ptr_param<Matrix> c) const
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_DISTRIBUTED_MATRIX_SPGEMM);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
