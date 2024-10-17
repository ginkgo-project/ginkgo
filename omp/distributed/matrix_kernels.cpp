// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/matrix_kernels.hpp"

#include <algorithm>

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
void count_overlap_entries(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    comm_index_type local_part, array<comm_index_type>& overlap_count,
    array<GlobalIndexType>& overlap_positions,
    array<GlobalIndexType>& original_positions)
{
    auto num_input_elements = input.get_num_stored_elements();
    auto input_row_idxs = input.get_const_row_idxs();
    auto row_part_ids = row_partition->get_part_ids();
    array<comm_index_type> row_part_ids_per_entry{exec, num_input_elements};

    size_type row_range_id = 0;
#pragma omp parallel for firstprivate(row_range_id)
    for (size_type i = 0; i < input.get_num_stored_elements(); ++i) {
        auto global_row = input_row_idxs[i];
        row_range_id = find_range(global_row, row_partition, row_range_id);
        auto row_part_id = row_part_ids[row_range_id];
        row_part_ids_per_entry.get_data()[i] = row_part_id;
        if (row_part_id != local_part) {
#pragma omp atomic
            overlap_count.get_data()[row_part_id]++;
            original_positions.get_data()[i] = i;
        } else {
            original_positions.get_data()[i] = -1;
        }
    }

    auto comp = [row_part_ids_per_entry, local_part](auto i, auto j) {
        comm_index_type a =
            i == -1 ? local_part : row_part_ids_per_entry.get_const_data()[i];
        comm_index_type b =
            j == -1 ? local_part : row_part_ids_per_entry.get_const_data()[j];
        return a < b;
    };
    std::stable_sort(original_positions.get_data(),
                     original_positions.get_data() + num_input_elements, comp);

#pragma omp parallel for
    for (size_type i = 0; i < num_input_elements; i++) {
        overlap_positions.get_data()[i] =
            original_positions.get_const_data()[i] == -1 ? 0 : 1;
    }

    components::prefix_sum_nonnegative(exec, overlap_positions.get_data(),
                                       num_input_elements);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_COUNT_OVERLAP_ENTRIES);


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void fill_overlap_send_buffers(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    comm_index_type local_part, const array<GlobalIndexType>& overlap_positions,
    const array<GlobalIndexType>& original_positions,
    array<GlobalIndexType>& overlap_row_idxs,
    array<GlobalIndexType>& overlap_col_idxs, array<ValueType>& overlap_values)
{
    auto input_row_idxs = input.get_const_row_idxs();
    auto input_col_idxs = input.get_const_col_idxs();
    auto input_vals = input.get_const_values();

#pragma omp parallel for
    for (size_type i = 0; i < input.get_num_stored_elements(); ++i) {
        auto in_pos = original_positions.get_const_data()[i];
        if (in_pos >= 0) {
            auto out_pos = overlap_positions.get_const_data()[i];
            overlap_row_idxs.get_data()[out_pos] = input_row_idxs[in_pos];
            overlap_col_idxs.get_data()[out_pos] = input_col_idxs[in_pos];
            overlap_values.get_data()[out_pos] = input_vals[in_pos];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_FILL_OVERLAP_SEND_BUFFERS);


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void separate_local_nonlocal(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<LocalIndexType>& local_row_idxs,
    array<LocalIndexType>& local_col_idxs, array<ValueType>& local_values,
    array<LocalIndexType>& non_local_row_idxs,
    array<GlobalIndexType>& non_local_col_idxs,
    array<ValueType>& non_local_values)
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

    // store non-local entries with global column idxs
    vector<global_nonzero> non_local_entries(exec);
    vector<local_nonzero> local_entries(exec);

    auto num_threads = static_cast<size_type>(omp_get_max_threads());
    auto num_input = input.get_num_stored_elements();
    auto size_per_thread = (num_input + num_threads - 1) / num_threads;
    vector<size_type> local_entry_offsets(num_threads, 0, exec);
    vector<size_type> non_local_entry_offsets(num_threads, 0, exec);

#pragma omp parallel firstprivate(col_range_id_hint, row_range_id_hint)
    {
        vector<global_nonzero> thread_non_local_entries(exec);
        vector<local_nonzero> thread_local_entries(exec);
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
                    thread_non_local_entries.emplace_back(local_row, global_col,
                                                          value);
                }
            }
        }
        local_entry_offsets[thread_id] = thread_local_entries.size();
        non_local_entry_offsets[thread_id] = thread_non_local_entries.size();

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
    // map non-local values to local column indices
    non_local_row_idxs.resize_and_reset(non_local_entries.size());
    non_local_col_idxs.resize_and_reset(non_local_entries.size());
    non_local_values.resize_and_reset(non_local_entries.size());
#pragma omp parallel for
    for (size_type i = 0; i < non_local_entries.size(); i++) {
        auto global = non_local_entries[i];
        non_local_row_idxs.get_data()[i] =
            static_cast<LocalIndexType>(global.row);
        non_local_col_idxs.get_data()[i] = global.column;
        non_local_values.get_data()[i] = global.value;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_SEPARATE_LOCAL_NONLOCAL);


}  // namespace distributed_matrix
}  // namespace omp
}  // namespace kernels
}  // namespace gko
