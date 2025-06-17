// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/fixed_coarsening_kernels.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"

namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace fixed_coarsening {


template <typename IndexType>
void renumber(std::shared_ptr<const DefaultExecutor> exec,
              const array<IndexType>& coarse_row, array<IndexType>* coarse_map)
{
    array<IndexType> temp_map(exec, coarse_map->get_size());
    components::fill_array(exec, temp_map.get_data(), temp_map.get_size(),
                           zero<IndexType>());
    run_kernel(
        exec,
        [] GKO_KERNEL(auto tidx, auto coarse_row, auto temp_map) {
            temp_map[coarse_row[tidx]] = 1;
        },
        coarse_row.get_size(), coarse_row.get_const_data(),
        temp_map.get_data());
    components::prefix_sum_nonnegative(exec, temp_map.get_data(),
                                       temp_map.get_size());
    run_kernel(
        exec,
        [] GKO_KERNEL(auto tidx, auto coarse_row, auto temp_map,
                      auto coarse_map) {
            auto selected_idx = coarse_row[tidx];
            coarse_map[selected_idx] = temp_map[selected_idx];
        },
        coarse_row.get_size(), coarse_row.get_const_data(),
        temp_map.get_const_data(), coarse_map->get_data());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_FIXED_COARSENING_RENUMBER_KERNEL);


template <typename IndexType>
void build_row_ptrs(std::shared_ptr<const DefaultExecutor> exec,
                    size_type original_nrows,
                    const IndexType* original_row_ptrs,
                    const IndexType* original_col_idxs,
                    const array<IndexType>& coarse_rows,
                    const array<IndexType>& coarse_cols_map,
                    size_type new_nrows, IndexType* new_row_ptrs)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto original_row_ptrs, auto original_col_idxs,
                      auto coarse_rows, auto coarse_cols_map,
                      auto new_row_ptrs) {
            auto original_row = coarse_rows[row];
            new_row_ptrs[row] = 0;
            for (auto i = original_row_ptrs[original_row];
                 i < original_row_ptrs[original_row + 1]; i++) {
                new_row_ptrs[row] += (coarse_cols_map[original_col_idxs[i]] !=
                                      invalid_index<IndexType>());
            }
        },
        new_nrows, original_row_ptrs, original_col_idxs,
        coarse_rows.get_const_data(), coarse_cols_map.get_const_data(),
        new_row_ptrs);
    components::prefix_sum_nonnegative(exec, new_row_ptrs, new_nrows + 1);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_FIXED_COARSENING_BUILD_ROW_PTRS_KERNEL);


template <typename ValueType, typename IndexType>
void map_to_coarse(std::shared_ptr<const DefaultExecutor> exec,
                   size_type original_nrows, const IndexType* original_row_ptrs,
                   const IndexType* original_col_idxs,
                   const ValueType* original_values,
                   const array<IndexType>& coarse_rows,
                   const array<IndexType>& coarse_cols_map, size_type new_nrows,
                   const IndexType* new_row_ptrs, IndexType* new_col_idxs,
                   ValueType* new_values)
{
    // We only run one thread per row.
    // It might also be done by assigning warp per row with ballot or using
    // lookup table
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto original_row_ptrs, auto original_col_idxs,
                      auto original_values, auto coarse_rows,
                      auto coarse_cols_map, auto new_row_ptrs,
                      auto new_col_idxs, auto new_values) {
            auto original_row = coarse_rows[row];
            IndexType index = new_row_ptrs[row];
            for (auto i = original_row_ptrs[original_row];
                 i < original_row_ptrs[original_row + 1]; i++) {
                auto new_index = coarse_cols_map[original_col_idxs[i]];
                if (new_index != invalid_index<IndexType>()) {
                    new_col_idxs[index] = new_index;
                    new_values[index] = original_values[i];
                    index++;
                }
            }
        },
        new_nrows, original_row_ptrs, original_col_idxs, original_values,
        coarse_rows.get_const_data(), coarse_cols_map.get_const_data(),
        new_row_ptrs, new_col_idxs, new_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FIXED_COARSENING_MAP_TO_COARSE_KERNEL);


}  // namespace fixed_coarsening
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
