// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/device_matrix_data_kernels.hpp"


#include <algorithm>


#include <omp.h>


#include "core/base/allocator.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace components {


template <typename ValueType, typename IndexType>
void remove_zeros(std::shared_ptr<const DefaultExecutor> exec,
                  array<ValueType>& values, array<IndexType>& row_idxs,
                  array<IndexType>& col_idxs)
{
    const auto size = values.get_size();
    const auto num_threads = omp_get_max_threads();
    const auto per_thread = static_cast<size_type>(ceildiv(size, num_threads));
    vector<size_type> partial_counts(num_threads, {exec});
#pragma omp parallel num_threads(num_threads)
    {
        const auto tidx = static_cast<size_type>(omp_get_thread_num());
        const auto begin = per_thread * tidx;
        const auto end = std::min(size, begin + per_thread);
        for (auto i = begin; i < end; i++) {
            partial_counts[tidx] +=
                is_nonzero(values.get_const_data()[i]) ? 1 : 0;
        }
    }
    std::partial_sum(partial_counts.begin(), partial_counts.end(),
                     partial_counts.begin());
    auto nnz = static_cast<size_type>(partial_counts.back());
    if (nnz < size) {
        array<ValueType> new_values{exec, nnz};
        array<IndexType> new_row_idxs{exec, nnz};
        array<IndexType> new_col_idxs{exec, nnz};
#pragma omp parallel num_threads(num_threads)
        {
            const auto tidx = static_cast<size_type>(omp_get_thread_num());
            const auto begin = per_thread * tidx;
            const auto end = std::min(size, begin + per_thread);
            auto out_idx = tidx == 0 ? size_type{} : partial_counts[tidx - 1];
            for (auto i = begin; i < end; i++) {
                auto val = values.get_const_data()[i];
                if (is_nonzero(val)) {
                    new_values.get_data()[out_idx] = val;
                    new_row_idxs.get_data()[out_idx] =
                        row_idxs.get_const_data()[i];
                    new_col_idxs.get_data()[out_idx] =
                        col_idxs.get_const_data()[i];
                    out_idx++;
                }
            }
        }
        values = std::move(new_values);
        row_idxs = std::move(new_row_idxs);
        col_idxs = std::move(new_col_idxs);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_REMOVE_ZEROS_KERNEL);


template <typename ValueType, typename IndexType>
void sum_duplicates(std::shared_ptr<const DefaultExecutor> exec,
                    size_type num_rows, array<ValueType>& values,
                    array<IndexType>& row_idxs, array<IndexType>& col_idxs)
{
    const auto size = values.get_size();
    array<int64> row_ptrs_array{exec, num_rows + 1};
    array<int64> out_row_ptrs_array{exec, num_rows + 1};
    components::convert_idxs_to_ptrs(exec, row_idxs.get_const_data(),
                                     row_idxs.get_size(), num_rows,
                                     row_ptrs_array.get_data());
    const auto row_ptrs = row_ptrs_array.get_const_data();
    const auto out_row_ptrs = out_row_ptrs_array.get_data();
#pragma omp parallel for
    for (IndexType row = 0; row < num_rows; row++) {
        int64 count_unique{};
        auto col = invalid_index<IndexType>();
        for (auto i = row_ptrs[row]; i < row_ptrs[row + 1]; i++) {
            const auto new_col = col_idxs.get_const_data()[i];
            if (col != new_col) {
                col = new_col;
                count_unique++;
            }
        }
        out_row_ptrs[row] = count_unique;
    }
    components::prefix_sum_nonnegative(exec, out_row_ptrs, num_rows + 1);
    const auto out_size = static_cast<size_type>(out_row_ptrs[num_rows]);
    if (out_size < size) {
        array<ValueType> new_values{exec, out_size};
        array<IndexType> new_row_idxs{exec, out_size};
        array<IndexType> new_col_idxs{exec, out_size};
#pragma omp parallel for
        for (IndexType row = 0; row < num_rows; row++) {
            auto out_i = out_row_ptrs[row] - 1;
            auto col = invalid_index<IndexType>();
            for (auto i = row_ptrs[row]; i < row_ptrs[row + 1]; i++) {
                const auto new_col = col_idxs.get_const_data()[i];
                if (col != new_col) {
                    col = new_col;
                    out_i++;
                    new_row_idxs.get_data()[out_i] = row;
                    new_col_idxs.get_data()[out_i] = col;
                    new_values.get_data()[out_i] = zero<ValueType>();
                }
                new_values.get_data()[out_i] += values.get_const_data()[i];
            }
        }
        values = std::move(new_values);
        row_idxs = std::move(new_row_idxs);
        col_idxs = std::move(new_col_idxs);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SUM_DUPLICATES_KERNEL);


template <typename ValueType, typename IndexType>
void sort_row_major(std::shared_ptr<const DefaultExecutor> exec,
                    device_matrix_data<ValueType, IndexType>& data)
{
    array<matrix_data_entry<ValueType, IndexType>> tmp{
        exec, data.get_num_stored_elements()};
    soa_to_aos(exec, data, tmp);
    std::sort(tmp.get_data(), tmp.get_data() + tmp.get_size());
    aos_to_soa(exec, tmp, data);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SORT_ROW_MAJOR_KERNEL);


}  // namespace components
}  // namespace omp
}  // namespace kernels
}  // namespace gko
