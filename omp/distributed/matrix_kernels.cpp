// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/matrix_kernels.hpp"

#include <algorithm>
#include <vector>

#include <omp.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include "core/base/allocator.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "reference/distributed/partition_helpers.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace distributed_matrix {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void separate_diag_off_diag(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<LocalIndexType>& diag_row_idxs,
    array<LocalIndexType>& diag_col_idxs, array<ValueType>& diag_values,
    array<LocalIndexType>& off_diag_row_idxs,
    array<GlobalIndexType>& off_diag_col_idxs,
    array<ValueType>& off_diag_values)
{
    using range_index_type = GlobalIndexType;
    using global_nonzero = matrix_data_entry<ValueType, GlobalIndexType>;
    using local_nonzero = matrix_data_entry<ValueType, LocalIndexType>;
    auto input_row_idxs = input.get_const_row_idxs();
    auto input_col_idxs = input.get_const_col_idxs();
    auto input_vals = input.get_const_values();
    auto row_part_ids = row_partition->get_part_ids();
    auto col_part_ids = col_partition->get_part_ids();
    auto num_parts = row_partition->get_num_parts();
    size_type row_range_id_hint = 0;
    size_type col_range_id_hint = 0;

    // store off-diag entries with global column idxs
    vector<global_nonzero> off_diag_entries(exec);
    vector<local_nonzero> diag_entries(exec);

    auto num_threads = static_cast<size_type>(omp_get_max_threads());
    auto num_input = input.get_num_stored_elements();
    auto size_per_thread = (num_input + num_threads - 1) / num_threads;
    vector<size_type> diag_entry_offsets(num_threads, 0, exec);
    vector<size_type> off_diag_entry_offsets(num_threads, 0, exec);

#pragma omp parallel firstprivate(col_range_id_hint, row_range_id_hint)
    {
        vector<global_nonzero> thread_off_diag_entries(exec);
        vector<local_nonzero> thread_diag_entries(exec);
        auto thread_id = omp_get_thread_num();
        auto thread_begin = thread_id * size_per_thread;
        auto thread_end = std::min(thread_begin + size_per_thread, num_input);
        // separate diag and off-diag entries for our input chunk
        for (auto i = thread_begin; i < thread_end; ++i) {
            const auto global_row = input_row_idxs[i];
            const auto global_col = input_col_idxs[i];
            const auto value = input_vals[i];
            auto row_range_id =
                find_range(global_row, row_partition, row_range_id_hint);
            row_range_id_hint = row_range_id;
            // skip rows that aren't owned by this rank
            if (row_part_ids[row_range_id] == local_part) {
                // map to part-local indices
                auto local_row =
                    map_to_local(global_row, row_partition, row_range_id);

                auto col_range_id =
                    find_range(global_col, col_partition, col_range_id_hint);
                col_range_id_hint = col_range_id;
                if (col_part_ids[col_range_id] == local_part) {
                    // store diag entry
                    auto local_col =
                        map_to_local(global_col, col_partition, col_range_id);
                    thread_diag_entries.emplace_back(local_row, local_col,
                                                     value);
                } else {
                    thread_off_diag_entries.emplace_back(local_row, global_col,
                                                         value);
                }
            }
        }
        diag_entry_offsets[thread_id] = thread_diag_entries.size();
        off_diag_entry_offsets[thread_id] = thread_off_diag_entries.size();

#pragma omp barrier
#pragma omp single
        {
            // assign output ranges to the individual threads
            size_type diag{};
            size_type off_diag{};
            for (size_type thread = 0; thread < num_threads; ++thread) {
                auto size_diag = diag_entry_offsets[thread];
                auto size_off_diag = off_diag_entry_offsets[thread];
                diag_entry_offsets[thread] = diag;
                off_diag_entry_offsets[thread] = off_diag;
                diag += size_diag;
                off_diag += size_off_diag;
            }
            diag_entries.resize(diag);
            off_diag_entries.resize(off_diag);
        }
        // write back the diag data to the output ranges
        auto diag = diag_entry_offsets[thread_id];
        auto off_diag = off_diag_entry_offsets[thread_id];
        for (const auto& entry : thread_diag_entries) {
            diag_entries[diag] = entry;
            diag++;
        }
        for (const auto& entry : thread_off_diag_entries) {
            off_diag_entries[off_diag] = entry;
            off_diag++;
        }
    }
    // store diag data to output
    diag_row_idxs.resize_and_reset(diag_entries.size());
    diag_col_idxs.resize_and_reset(diag_entries.size());
    diag_values.resize_and_reset(diag_entries.size());
#pragma omp parallel for
    for (size_type i = 0; i < diag_entries.size(); ++i) {
        const auto& entry = diag_entries[i];
        diag_row_idxs.get_data()[i] = entry.row;
        diag_col_idxs.get_data()[i] = entry.column;
        diag_values.get_data()[i] = entry.value;
    }
    // map off-diag values to local column indices
    off_diag_row_idxs.resize_and_reset(off_diag_entries.size());
    off_diag_col_idxs.resize_and_reset(off_diag_entries.size());
    off_diag_values.resize_and_reset(off_diag_entries.size());
#pragma omp parallel for
    for (size_type i = 0; i < off_diag_entries.size(); i++) {
        auto global = off_diag_entries[i];
        off_diag_row_idxs.get_data()[i] =
            static_cast<LocalIndexType>(global.row);
        off_diag_col_idxs.get_data()[i] = global.column;
        off_diag_values.get_data()[i] = global.value;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_SEPARATE_DIAG_OFF_DIAG);


template <typename ValueType, typename LocalIndexType>
void separate_local_nonlocal_columns(
    std::shared_ptr<const DefaultExecutor> exec,
    const array<LocalIndexType>& row_idxs,
    const array<LocalIndexType>& col_idxs, const array<ValueType>& values,
    LocalIndexType num_local_cols, array<LocalIndexType>& diag_row_idxs,
    array<LocalIndexType>& diag_col_idxs, array<ValueType>& diag_values,
    array<LocalIndexType>& off_diag_row_idxs,
    array<LocalIndexType>& off_diag_col_idxs, array<ValueType>& off_diag_values)
{
    auto row_ptr = row_idxs.get_const_data();
    auto col_ptr = col_idxs.get_const_data();
    auto val_ptr = values.get_const_data();
    const auto nnz = col_idxs.get_size();

    size_type num_diag = 0;
#pragma omp parallel for reduction(+ : num_diag)
    for (size_type i = 0; i < nnz; ++i) {
        if (col_ptr[i] < num_local_cols) {
            num_diag++;
        }
    }
    const auto num_off = nnz - num_diag;

    diag_row_idxs.resize_and_reset(num_diag);
    diag_col_idxs.resize_and_reset(num_diag);
    diag_values.resize_and_reset(num_diag);
    off_diag_row_idxs.resize_and_reset(num_off);
    off_diag_col_idxs.resize_and_reset(num_off);
    off_diag_values.resize_and_reset(num_off);

    // Sequential fill, so that the output keeps the input order.
    size_type di = 0;
    size_type oi = 0;
    for (size_type i = 0; i < nnz; ++i) {
        if (col_ptr[i] < num_local_cols) {
            diag_row_idxs.get_data()[di] = row_ptr[i];
            diag_col_idxs.get_data()[di] = col_ptr[i];
            diag_values.get_data()[di] = val_ptr[i];
            ++di;
        } else {
            off_diag_row_idxs.get_data()[oi] = row_ptr[i];
            off_diag_col_idxs.get_data()[oi] = col_ptr[i] - num_local_cols;
            off_diag_values.get_data()[oi] = val_ptr[i];
            ++oi;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SEPARATE_LOCAL_NONLOCAL_COLUMNS);


template <typename LocalIndexType, typename GlobalIndexType>
void unique_nonlocal_columns(
    std::shared_ptr<const DefaultExecutor> exec,
    const array<GlobalIndexType>& global_cols,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<GlobalIndexType>& nonlocal_cols)
{
    const auto n = global_cols.get_size();
    auto in = global_cols.get_const_data();
    auto col_part_ids = col_partition->get_part_ids();

    std::vector<GlobalIndexType> remote;
    remote.reserve(n);
    size_type col_range_id = 0;
    for (size_type i = 0; i < n; ++i) {
        col_range_id = find_range(in[i], col_partition, col_range_id);
        if (col_part_ids[col_range_id] != local_part) {
            remote.push_back(in[i]);
        }
    }
    std::sort(remote.begin(), remote.end());
    remote.erase(std::unique(remote.begin(), remote.end()), remote.end());

    nonlocal_cols.resize_and_reset(remote.size());
    std::copy(remote.begin(), remote.end(), nonlocal_cols.get_data());
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_UNIQUE_NONLOCAL_COLUMNS);


}  // namespace distributed_matrix
}  // namespace omp
}  // namespace kernels
}  // namespace gko
