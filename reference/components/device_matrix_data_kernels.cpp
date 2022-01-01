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


#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace components {


template <typename ValueType, typename IndexType>
void remove_zeros(std::shared_ptr<const DefaultExecutor> exec,
                  Array<matrix_data_entry<ValueType, IndexType>>& data)
{
    auto size = data.get_num_elems();
    auto is_nonzero = [](matrix_data_entry<ValueType, IndexType> entry) {
        return entry.value != zero<ValueType>();
    };
    auto nnz = std::count_if(data.get_const_data(),
                             data.get_const_data() + size, is_nonzero);
    if (nnz < size) {
        Array<matrix_data_entry<ValueType, IndexType>> result{
            exec, static_cast<size_type>(nnz)};
        std::copy_if(data.get_const_data(), data.get_const_data() + size,
                     result.get_data(), is_nonzero);
        data = std::move(result);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_REMOVE_ZEROS_KERNEL);


template <typename ValueType, typename IndexType>
void sort_row_major(std::shared_ptr<const DefaultExecutor> exec,
                    Array<matrix_data_entry<ValueType, IndexType>>& data)
{
    std::sort(data.get_data(), data.get_data() + data.get_num_elems());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SORT_ROW_MAJOR_KERNEL);


template <typename ValueType, typename IndexType, typename RowPtrType>
void build_row_ptrs(std::shared_ptr<const DefaultExecutor> exec,
                    const Array<matrix_data_entry<ValueType, IndexType>>& data,
                    size_type num_rows, RowPtrType* row_ptrs)
{
    std::fill_n(row_ptrs, num_rows + 1, 0);
    for (size_type i = 0; i < data.get_num_elems(); i++) {
        row_ptrs[data.get_const_data()[i].row]++;
    }
    components::prefix_sum(exec, row_ptrs, num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_KERNEL32);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_KERNEL64);


}  // namespace components
}  // namespace reference
}  // namespace kernels
}  // namespace gko
