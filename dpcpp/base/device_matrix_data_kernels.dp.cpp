/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

// force-top: on
// oneDPL needs to be first to avoid issues with libstdc++ TBB impl
#include <oneapi/dpl/algorithm>
// force-top: off


#include "core/base/device_matrix_data_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>


#include "dpcpp/base/onedpl.hpp"


namespace gko {
namespace kernels {
namespace sycl {
namespace components {


template <typename ValueType, typename IndexType>
void remove_zeros(std::shared_ptr<const DefaultExecutor> exec,
                  array<ValueType>& values, array<IndexType>& row_idxs,
                  array<IndexType>& col_idxs)
{
    using nonzero_type = matrix_data_entry<ValueType, IndexType>;
    auto size = values.get_num_elems();
    auto policy = onedpl_policy(exec);
    auto nnz = std::count_if(
        policy, values.get_const_data(), values.get_const_data() + size,
        [](ValueType val) { return is_nonzero<ValueType>(val); });
    if (nnz < size) {
        // allocate new storage
        array<ValueType> new_values{exec, static_cast<size_type>(nnz)};
        array<IndexType> new_row_idxs{exec, static_cast<size_type>(nnz)};
        array<IndexType> new_col_idxs{exec, static_cast<size_type>(nnz)};
        // copy nonzeros
        auto input_it = oneapi::dpl::make_zip_iterator(
            row_idxs.get_const_data(), col_idxs.get_const_data(),
            values.get_const_data());
        auto output_it = oneapi::dpl::make_zip_iterator(new_row_idxs.get_data(),
                                                        new_col_idxs.get_data(),
                                                        new_values.get_data());
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
    auto size = values.get_num_elems();
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
                    device_matrix_data<ValueType, IndexType>& data)
{
    auto policy = onedpl_policy(exec);
    auto input_it = oneapi::dpl::make_zip_iterator(
        data.get_row_idxs(), data.get_col_idxs(), data.get_values());
    std::sort(policy, input_it, input_it + data.get_num_elems(),
              [](auto a, auto b) {
                  return std::tie(std::get<0>(a), std::get<1>(a)) <
                         std::tie(std::get<0>(b), std::get<1>(b));
              });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SORT_ROW_MAJOR_KERNEL);


}  // namespace components
}  // namespace sycl
}  // namespace kernels
}  // namespace gko
