// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/device_matrix_data_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/math.hpp>


#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace components {


template <typename ValueType, typename IndexType>
void soa_to_aos(std::shared_ptr<const DefaultExecutor> exec,
                const device_matrix_data<ValueType, IndexType>& in,
                array<matrix_data_entry<ValueType, IndexType>>& out)
{
    for (size_type i = 0; i < in.get_num_stored_elements(); i++) {
        out.get_data()[i] = {in.get_const_row_idxs()[i],
                             in.get_const_col_idxs()[i],
                             in.get_const_values()[i]};
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SOA_TO_AOS_KERNEL);


template <typename ValueType, typename IndexType>
void aos_to_soa(std::shared_ptr<const DefaultExecutor> exec,
                const array<matrix_data_entry<ValueType, IndexType>>& in,
                device_matrix_data<ValueType, IndexType>& out)
{
    for (size_type i = 0; i < in.get_size(); i++) {
        const auto entry = in.get_const_data()[i];
        out.get_row_idxs()[i] = entry.row;
        out.get_col_idxs()[i] = entry.column;
        out.get_values()[i] = entry.value;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_AOS_TO_SOA_KERNEL);


template <typename ValueType, typename IndexType>
void remove_zeros(std::shared_ptr<const DefaultExecutor> exec,
                  array<ValueType>& values, array<IndexType>& row_idxs,
                  array<IndexType>& col_idxs)
{
    auto size = values.get_size();
    auto nnz = static_cast<size_type>(
        std::count_if(values.get_const_data(), values.get_const_data() + size,
                      is_nonzero<ValueType>));
    if (nnz < size) {
        array<ValueType> new_values{exec, nnz};
        array<IndexType> new_row_idxs{exec, nnz};
        array<IndexType> new_col_idxs{exec, nnz};
        size_type out_i{};
        for (size_type i = 0; i < size; i++) {
            if (is_nonzero(values.get_const_data()[i])) {
                new_values.get_data()[out_i] = values.get_const_data()[i];
                new_row_idxs.get_data()[out_i] = row_idxs.get_const_data()[i];
                new_col_idxs.get_data()[out_i] = col_idxs.get_const_data()[i];
                out_i++;
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
void sum_duplicates(std::shared_ptr<const DefaultExecutor> exec, size_type,
                    array<ValueType>& values, array<IndexType>& row_idxs,
                    array<IndexType>& col_idxs)
{
    auto row = invalid_index<IndexType>();
    auto col = invalid_index<IndexType>();
    const auto size = values.get_size();
    size_type count_unique{};
    for (size_type i = 0; i < size; i++) {
        const auto new_row = row_idxs.get_const_data()[i];
        const auto new_col = col_idxs.get_const_data()[i];
        if (row != new_row || col != new_col) {
            row = new_row;
            col = new_col;
            count_unique++;
        }
    }
    if (count_unique < size) {
        array<ValueType> new_values{exec, count_unique};
        array<IndexType> new_row_idxs{exec, count_unique};
        array<IndexType> new_col_idxs{exec, count_unique};
        row = invalid_index<IndexType>();
        col = invalid_index<IndexType>();
        int64 out_i = -1;
        for (size_type i = 0; i < size; i++) {
            const auto new_row = row_idxs.get_const_data()[i];
            const auto new_col = col_idxs.get_const_data()[i];
            const auto new_val = values.get_const_data()[i];
            if (row != new_row || col != new_col) {
                row = new_row;
                col = new_col;
                out_i++;
                new_row_idxs.get_data()[out_i] = row;
                new_col_idxs.get_data()[out_i] = col;
                new_values.get_data()[out_i] = zero<ValueType>();
            }
            new_values.get_data()[out_i] += new_val;
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
}  // namespace reference
}  // namespace kernels
}  // namespace gko
