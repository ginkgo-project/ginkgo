// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <oneapi/dpl/algorithm>

#include "core/base/device_matrix_data_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>

#include "dpcpp/base/math.hpp"
#include "dpcpp/base/onedpl.hpp"
#include "dpcpp/base/types.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace components {


template <typename ValueType, typename IndexType>
void remove_zeros(std::shared_ptr<const DefaultExecutor> exec,
                  array<ValueType>& values, array<IndexType>& row_idxs,
                  array<IndexType>& col_idxs)
{
    using device_value_type = device_type<ValueType>;
    auto size = values.get_size();
    auto policy = onedpl_policy(exec);
    auto nnz =
        std::count_if(policy, as_device_type(values.get_const_data()),
                      as_device_type(values.get_const_data()) + size,
                      [](device_value_type val) { return is_nonzero(val); });
    if (nnz < size) {
        // allocate new storage
        array<ValueType> new_values{exec, static_cast<size_type>(nnz)};
        array<IndexType> new_row_idxs{exec, static_cast<size_type>(nnz)};
        array<IndexType> new_col_idxs{exec, static_cast<size_type>(nnz)};
        // copy nonzeros
        auto input_it = oneapi::dpl::make_zip_iterator(
            row_idxs.get_const_data(), col_idxs.get_const_data(),
            as_device_type(values.get_const_data()));
        auto output_it = oneapi::dpl::make_zip_iterator(
            new_row_idxs.get_data(), new_col_idxs.get_data(),
            as_device_type(new_values.get_data()));
        std::copy_if(policy, input_it, input_it + size, output_it,
                     [](auto tuple) { return is_nonzero(std::get<2>(tuple)); });
        // swap out storage
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
    using nonzero_type = matrix_data_entry<ValueType, IndexType>;
    auto size = values.get_size();
    if (size == 0) {
        return;
    }
    auto policy = onedpl_policy(exec);
    auto in_loc_it = oneapi::dpl::make_zip_iterator(row_idxs.get_const_data(),
                                                    col_idxs.get_const_data());
    auto adj_in_loc_it =
        oneapi::dpl::make_zip_iterator(in_loc_it, in_loc_it + 1);
    auto nnz =
        1 + std::count_if(policy, adj_in_loc_it, adj_in_loc_it + (size - 1),
                          [](auto pair) {
                              return std::get<0>(pair) != std::get<1>(pair);
                          });
    if (nnz < size) {
        GKO_NOT_IMPLEMENTED;
        /* TODO uncomment once oneDPL reduce_by_segment is fixed
        // allocate new storage
        array<ValueType> new_values{exec, static_cast<size_type>(nnz)};
        array<IndexType> new_row_idxs{exec, static_cast<size_type>(nnz)};
        array<IndexType> new_col_idxs{exec, static_cast<size_type>(nnz)};
        // copy nonzeros
        auto out_loc_it = oneapi::dpl::make_zip_iterator(
            new_row_idxs.get_data(), new_col_idxs.get_data());
        oneapi::dpl::reduce_by_segment(policy, in_loc_it, in_loc_it + size,
                                       values.get_const_data(), out_loc_it,
                                       new_values.get_data());
        // swap out storage
        values = std::move(new_values);
        row_idxs = std::move(new_row_idxs);
        col_idxs = std::move(new_col_idxs);
        */
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SUM_DUPLICATES_KERNEL);


template <typename ValueType, typename IndexType>
void sort_row_major(std::shared_ptr<const DefaultExecutor> exec,
                    size_type num_elems, IndexType* row_idxs,
                    IndexType* col_idxs, ValueType* vals)
{
    auto policy = onedpl_policy(exec);
    auto input_it = oneapi::dpl::make_zip_iterator(row_idxs, col_idxs, vals);
    std::sort(policy, input_it, input_it + num_elems, [](auto a, auto b) {
        return std::tie(std::get<0>(a), std::get<1>(a)) <
               std::tie(std::get<0>(b), std::get<1>(b));
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SORT_ROW_MAJOR_KERNEL);


}  // namespace components
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
