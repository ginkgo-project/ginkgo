/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include "core/components/device_matrix_data_kernels.hpp"


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
                Array<matrix_data_entry<ValueType, IndexType>>& out)
{
    for (size_type i = 0; i < in.get_num_elems(); i++) {
        out.get_data()[i] = {in.get_const_row_idxs()[i],
                             in.get_const_col_idxs()[i],
                             in.get_const_values()[i]};
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SOA_TO_AOS_KERNEL);


template <typename ValueType, typename IndexType>
void aos_to_soa(std::shared_ptr<const DefaultExecutor> exec,
                const Array<matrix_data_entry<ValueType, IndexType>>& in,
                device_matrix_data<ValueType, IndexType>& out)
{
    for (size_type i = 0; i < in.get_num_elems(); i++) {
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
                  Array<ValueType>& values, Array<IndexType>& row_idxs,
                  Array<IndexType>& col_idxs)
{
    auto size = values.get_num_elems();
    auto nnz = static_cast<size_type>(
        std::count_if(values.get_const_data(), values.get_const_data() + size,
                      is_nonzero<ValueType>));
    if (nnz < size) {
        Array<ValueType> new_values{exec, nnz};
        Array<IndexType> new_row_idxs{exec, nnz};
        Array<IndexType> new_col_idxs{exec, nnz};
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
void sort_row_major(std::shared_ptr<const DefaultExecutor> exec,
                    device_matrix_data<ValueType, IndexType>& data)
{
    Array<matrix_data_entry<ValueType, IndexType>> tmp{exec,
                                                       data.get_num_elems()};
    soa_to_aos(exec, data, tmp);
    std::sort(tmp.get_data(), tmp.get_data() + tmp.get_num_elems());
    aos_to_soa(exec, tmp, data);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SORT_ROW_MAJOR_KERNEL);


}  // namespace components
}  // namespace reference
}  // namespace kernels
}  // namespace gko
