// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/dd_matrix_kernels.hpp"

#include <omp.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include "core/base/allocator.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "reference/distributed/partition_helpers.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace distributed_dd_matrix {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void filter_non_owning_idxs(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<GlobalIndexType>& non_owning_row_idxs,
    array<GlobalIndexType>& non_owning_col_idxs)
{
    auto input_row_idxs = input.get_const_row_idxs();
    auto input_col_idxs = input.get_const_col_idxs();
    auto input_vals = input.get_const_values();
    auto row_part_ids = row_partition->get_part_ids();
    auto col_part_ids = col_partition->get_part_ids();
    auto num_parts = row_partition->get_num_parts();
    size_type row_range_id_hint = 0;
    size_type col_range_id_hint = 0;

    // store non-local entries with global column idxs
    vector<GlobalIndexType> non_local_row_idxs(exec);
    vector<GlobalIndexType> non_local_col_idxs(exec);

    auto num_threads = static_cast<size_type>(omp_get_max_threads());
    auto num_input = input.get_num_stored_elements();
    auto size_per_thread = (num_input + num_threads - 1) / num_threads;
    vector<size_type> non_local_col_offsets(num_threads, 0, exec);
    vector<size_type> non_local_row_offsets(num_threads, 0, exec);

#pragma omp parallel firstprivate(col_range_id_hint, row_range_id_hint)
    {
        vector<GlobalIndexType> thread_non_local_col_idxs(exec);
        vector<GlobalIndexType> thread_non_local_row_idxs(exec);
        auto thread_id = omp_get_thread_num();
        auto thread_begin = thread_id * size_per_thread;
        auto thread_end = std::min(thread_begin + size_per_thread, num_input);
        // Count non local row and colunm idxs per thread
        for (size_type i = thread_begin; i < thread_end; i++) {
            auto global_col = input_col_idxs[i];
            auto global_row = input_row_idxs[i];
            col_range_id_hint =
                find_range(global_col, col_partition, col_range_id_hint);
            row_range_id_hint =
                find_range(global_row, row_partition, row_range_id_hint);
            if (col_part_ids[col_range_id_hint] != local_part) {
                thread_non_local_col_idxs.push_back(global_col);
            }
            if (row_part_ids[row_range_id_hint] != local_part) {
                thread_non_local_row_idxs.push_back(global_row);
            }
        }
        non_local_col_offsets[thread_id] = thread_non_local_col_idxs.size();
        non_local_row_offsets[thread_id] = thread_non_local_row_idxs.size();

#pragma omp barrier
#pragma omp single
        {
            // assign output ranges to the individual threads
            size_type n_non_local_col_idxs{};
            size_type n_non_local_row_idxs{};
            for (size_type thread = 0; thread < num_threads; thread++) {
                auto size_col_idxs = non_local_col_offsets[thread];
                auto size_row_idxs = non_local_row_offsets[thread];
                non_local_col_offsets[thread] = n_non_local_col_idxs;
                non_local_row_offsets[thread] = n_non_local_row_idxs;
                n_non_local_col_idxs += size_col_idxs;
                n_non_local_row_idxs += size_row_idxs;
            }
            non_local_col_idxs.resize(n_non_local_col_idxs);
            non_local_row_idxs.resize(n_non_local_row_idxs);
        }
        // write back the non_local idxs to the output ranges
        auto col_counter = non_local_col_offsets[thread_id];
        auto row_counter = non_local_row_offsets[thread_id];
        for (const auto& non_local_col : thread_non_local_col_idxs) {
            non_local_col_idxs[col_counter] = non_local_col;
            col_counter++;
        }
        for (const auto& non_local_row : thread_non_local_row_idxs) {
            non_local_row_idxs[row_counter] = non_local_row;
            row_counter++;
        }
    }

    non_owning_col_idxs.resize_and_reset(non_local_col_idxs.size());
#pragma omp parallel for
    for (size_type i = 0; i < non_local_col_idxs.size(); i++) {
        non_owning_col_idxs.get_data()[i] = non_local_col_idxs[i];
    }
    non_owning_row_idxs.resize_and_reset(non_local_row_idxs.size());
#pragma omp parallel for
    for (size_type i = 0; i < non_local_row_idxs.size(); i++) {
        non_owning_row_idxs.get_data()[i] = non_local_row_idxs[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_FILTER_NON_OWNING_IDXS);


}  // namespace distributed_dd_matrix
}  // namespace omp
}  // namespace kernels
}  // namespace gko
