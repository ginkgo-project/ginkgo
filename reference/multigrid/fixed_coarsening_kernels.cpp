// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/fixed_coarsening_kernels.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace fixed_coarsening {


template <typename IndexType>
void renumber(std::shared_ptr<const DefaultExecutor> exec,
              const array<IndexType>& coarse_row, array<IndexType>* coarse_map)
{
    array<IndexType> temp_map(exec, coarse_map->get_size());
    for (size_type i = 0; i < temp_map.get_size(); i++) {
        temp_map.get_data()[i] = 0;
    }
    for (size_type i = 0; i < coarse_row.get_size(); i++) {
        temp_map.get_data()[coarse_row.get_const_data()[i]] = 1;
    }
    components::prefix_sum_nonnegative(exec, temp_map.get_data(),
                                       temp_map.get_size());
    for (size_type i = 0; i < coarse_row.get_size(); i++) {
        auto selected_idx = coarse_row.get_const_data()[i];
        coarse_map->get_data()[selected_idx] =
            temp_map.get_const_data()[selected_idx];
    }
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
    std::cout << "build_row_ptrs (";
    for (size_type row = 0; row < new_nrows; row++) {
        auto original_row = coarse_rows.get_const_data()[row];
        GKO_ASSERT(original_row < original_nrows);
        new_row_ptrs[row] = 0;
        std::cout << "row-" << row << "[";
        for (auto i = original_row_ptrs[original_row];
             i < original_row_ptrs[original_row + 1]; i++) {
            if (coarse_cols_map.get_const_data()[original_col_idxs[i]] !=
                invalid_index<IndexType>()) {
                new_row_ptrs[row]++;
            }
            std::cout << original_col_idxs[i] << "+"
                      << coarse_cols_map.get_const_data()[original_col_idxs[i]]
                      << ",";
        }
        std::cout << "] ";
        std::cout << row << "->" << new_row_ptrs[row];
    }
    std::cout << ") ";

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
    for (size_type row = 0; row < new_nrows; row++) {
        auto original_row = coarse_rows.get_const_data()[row];
        IndexType index = new_row_ptrs[row];
        for (auto i = original_row_ptrs[original_row];
             i < original_row_ptrs[original_row + 1]; i++) {
            auto new_index =
                coarse_cols_map.get_const_data()[original_col_idxs[i]];
            if (new_index != invalid_index<IndexType>()) {
                new_col_idxs[index] = new_index;
                new_values[index] = original_values[i];
                index++;
            }
        }
        GKO_ASSERT(index == new_row_ptrs[row + 1]);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FIXED_COARSENING_MAP_TO_COARSE_KERNEL);


}  // namespace fixed_coarsening
}  // namespace reference
}  // namespace kernels
}  // namespace gko
