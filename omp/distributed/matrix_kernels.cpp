// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/matrix_kernels.hpp"


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/base/allocator.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace distributed_matrix {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void build_local_nonlocal(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<LocalIndexType>& local_row_idxs,
    array<LocalIndexType>& local_col_idxs, array<ValueType>& local_values,
    array<LocalIndexType>& non_local_row_idxs,
    array<LocalIndexType>& non_local_col_idxs,
    array<ValueType>& non_local_values,
    array<LocalIndexType>& local_gather_idxs,
    array<comm_index_type>& recv_sizes,
    array<GlobalIndexType>& non_local_to_global)
{
    using partition_type =
        experimental::distributed::Partition<LocalIndexType, GlobalIndexType>;
    using range_index_type = GlobalIndexType;
    using global_nonzero = matrix_data_entry<ValueType, GlobalIndexType>;
    using local_nonzero = matrix_data_entry<ValueType, LocalIndexType>;
    auto input_row_idxs = input.get_const_row_idxs();
    auto input_col_idxs = input.get_const_col_idxs();
    auto input_vals = input.get_const_values();
    auto row_part_ids = row_partition->get_part_ids();
    auto col_part_ids = col_partition->get_part_ids();
    auto num_parts = row_partition->get_num_parts();
    auto recv_sizes_ptr = recv_sizes.get_data();
    size_type row_range_id_hint = 0;
    size_type col_range_id_hint = 0;
    // zero recv_sizes values
    std::fill_n(recv_sizes_ptr, num_parts, comm_index_type{});

    auto find_range = [](GlobalIndexType idx, const partition_type* partition,
                         size_type hint) {
        auto range_bounds = partition->get_range_bounds();
        auto num_ranges = partition->get_num_ranges();
        if (range_bounds[hint] <= idx && idx < range_bounds[hint + 1]) {
            return hint;
        } else {
            auto it = std::upper_bound(range_bounds + 1,
                                       range_bounds + num_ranges + 1, idx);
            return static_cast<size_type>(std::distance(range_bounds + 1, it));
        }
    };
    auto map_to_local = [](GlobalIndexType idx, const partition_type* partition,
                           size_type range_id) {
        auto range_bounds = partition->get_range_bounds();
        auto range_starting_indices = partition->get_range_starting_indices();
        return static_cast<LocalIndexType>(idx - range_bounds[range_id]) +
               range_starting_indices[range_id];
    };

    // store non-local columns and their range indices
    map<GlobalIndexType, range_index_type> non_local_cols(exec);
    // store non-local entries with global column idxs
    vector<global_nonzero> non_local_entries(exec);
    vector<local_nonzero> local_entries(exec);

    auto num_threads = static_cast<size_type>(omp_get_max_threads());
    auto num_input = input.get_num_stored_elements();
    auto size_per_thread = (num_input + num_threads - 1) / num_threads;
    std::vector<size_type> local_entry_offsets(num_threads, 0);
    std::vector<size_type> non_local_entry_offsets(num_threads, 0);

#pragma omp parallel firstprivate(col_range_id_hint, row_range_id_hint)
    {
        std::unordered_map<GlobalIndexType, range_index_type>
            thread_non_local_cols;
        std::vector<global_nonzero> thread_non_local_entries;
        std::vector<local_nonzero> thread_local_entries;
        std::vector<comm_index_type> thread_recv_sizes;
        auto thread_id = omp_get_thread_num();
        auto thread_begin = thread_id * size_per_thread;
        auto thread_end = std::min(thread_begin + size_per_thread, num_input);
        // separate local and non-local entries for our input chunk
        for (auto i = thread_begin; i < thread_end; ++i) {
            const auto global_row = input_row_idxs[i];
            const auto global_col = input_col_idxs[i];
            const auto value = input_vals[i];
            auto row_range_id =
                find_range(global_row, row_partition, row_range_id_hint);
            row_range_id_hint = row_range_id;
            // skip non-local rows
            if (row_part_ids[row_range_id] == local_part) {
                // map to part-local indices
                auto local_row =
                    map_to_local(global_row, row_partition, row_range_id);

                auto col_range_id =
                    find_range(global_col, col_partition, col_range_id_hint);
                col_range_id_hint = col_range_id;
                if (col_part_ids[col_range_id] == local_part) {
                    // store local entry
                    auto local_col =
                        map_to_local(global_col, col_partition, col_range_id);
                    thread_local_entries.emplace_back(local_row, local_col,
                                                      value);
                } else {
                    thread_non_local_cols.emplace(global_col, col_range_id);
                    thread_non_local_entries.emplace_back(local_row, global_col,
                                                          value);
                }
            }
        }
        local_entry_offsets[thread_id] = thread_local_entries.size();
        non_local_entry_offsets[thread_id] = thread_non_local_entries.size();

#pragma omp critical
        {
            // collect global non-local columns
            non_local_cols.insert(thread_non_local_cols.begin(),
                                  thread_non_local_cols.end());
        }
#pragma omp barrier
#pragma omp single
        {
            // assign output ranges to the individual threads
            size_type local{};
            size_type non_local{};
            for (size_type thread = 0; thread < num_threads; ++thread) {
                auto size_local = local_entry_offsets[thread];
                auto size_non_local = non_local_entry_offsets[thread];
                local_entry_offsets[thread] = local;
                non_local_entry_offsets[thread] = non_local;
                local += size_local;
                non_local += size_non_local;
            }
            local_entries.resize(local);
            non_local_entries.resize(non_local);
        }
        // write back the local data to the output ranges
        auto local = local_entry_offsets[thread_id];
        auto non_local = non_local_entry_offsets[thread_id];
        for (const auto& entry : thread_local_entries) {
            local_entries[local] = entry;
            local++;
        }
        for (const auto& entry : thread_non_local_entries) {
            non_local_entries[non_local] = entry;
            non_local++;
        }
    }
    // store local data to output
    local_row_idxs.resize_and_reset(local_entries.size());
    local_col_idxs.resize_and_reset(local_entries.size());
    local_values.resize_and_reset(local_entries.size());
#pragma omp parallel for
    for (size_type i = 0; i < local_entries.size(); ++i) {
        const auto& entry = local_entries[i];
        local_row_idxs.get_data()[i] = entry.row;
        local_col_idxs.get_data()[i] = entry.column;
        local_values.get_data()[i] = entry.value;
    }

    // count non-local columns per part
    for (const auto& entry : non_local_cols) {
        auto col_range_id = entry.second;
        recv_sizes_ptr[col_part_ids[col_range_id]]++;
    }
    const auto num_non_local_cols = std::accumulate(
        recv_sizes_ptr, recv_sizes_ptr + num_parts, size_type{});
    components::prefix_sum_nonnegative(exec, recv_sizes_ptr, num_parts);

    // collect and renumber offdiagonal columns
    local_gather_idxs.resize_and_reset(num_non_local_cols);
    std::unordered_map<GlobalIndexType, LocalIndexType>
        non_local_global_to_local;
    for (const auto& entry : non_local_cols) {
        auto range = entry.second;
        auto part = col_part_ids[range];
        auto idx = recv_sizes_ptr[part];
        local_gather_idxs.get_data()[idx] =
            map_to_local(entry.first, col_partition, entry.second);
        non_local_global_to_local[entry.first] = idx;
        ++recv_sizes_ptr[part];
    }

    // build local-to-global map for non-local columns
    non_local_to_global.resize_and_reset(num_non_local_cols);
    std::fill_n(non_local_to_global.get_data(), non_local_to_global.get_size(),
                invalid_index<GlobalIndexType>());
    for (const auto& key_value : non_local_global_to_local) {
        const auto global_idx = key_value.first;
        const auto local_idx = key_value.second;
        non_local_to_global.get_data()[local_idx] = global_idx;
    }

    // compute sizes from shifted offsets
    for (size_type i = num_parts - 1; i > 0; --i) {
        recv_sizes_ptr[i] -= recv_sizes_ptr[i - 1];
    }

    // map non-local values to local column indices
    non_local_row_idxs.resize_and_reset(non_local_entries.size());
    non_local_col_idxs.resize_and_reset(non_local_entries.size());
    non_local_values.resize_and_reset(non_local_entries.size());
#pragma omp parallel for
    for (size_type i = 0; i < non_local_entries.size(); i++) {
        auto global = non_local_entries[i];
        non_local_row_idxs.get_data()[i] =
            static_cast<LocalIndexType>(global.row);
        non_local_col_idxs.get_data()[i] =
            non_local_global_to_local[global.column];
        non_local_values.get_data()[i] = global.value;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_LOCAL_NONLOCAL);


}  // namespace distributed_matrix
}  // namespace omp
}  // namespace kernels
}  // namespace gko
