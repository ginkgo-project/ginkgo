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


#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>


#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace components {


template <typename ValueType, typename IndexType>
void remove_zeros(std::shared_ptr<const DefaultExecutor> exec,
                  Array<matrix_data_entry<ValueType, IndexType>>& data)
{
    using nonzero_type = cuda_type<matrix_data_entry<ValueType, IndexType>>;
    using host_nonzero_type = matrix_data_entry<ValueType, IndexType>;
    static_assert(sizeof(nonzero_type) == sizeof(host_nonzero_type),
                  "mismatching size");
    auto size = data.get_num_elems();
    auto nnz = thrust::count_if(
        thrust::device_pointer_cast(as_cuda_type(data.get_const_data())),
        thrust::device_pointer_cast(as_cuda_type(data.get_const_data() + size)),
        [] __device__(nonzero_type entry) {
            return entry.value != zero(entry.value);
        });
    if (nnz < size) {
        Array<matrix_data_entry<ValueType, IndexType>> result{
            exec, static_cast<size_type>(nnz)};
        thrust::copy_if(
            thrust::device,
            thrust::device_pointer_cast(as_cuda_type(data.get_const_data())),
            thrust::device_pointer_cast(
                as_cuda_type(data.get_const_data() + size)),
            thrust::device_pointer_cast(as_cuda_type(result.get_data())),
            [] __device__(nonzero_type entry) {
                return entry.value != zero(entry.value);
            });
        data = std::move(result);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_REMOVE_ZEROS_KERNEL);


template <typename ValueType, typename IndexType>
void sort_row_major(std::shared_ptr<const DefaultExecutor> exec,
                    Array<matrix_data_entry<ValueType, IndexType>>& data)
{
    using nonzero_type = cuda_type<matrix_data_entry<ValueType, IndexType>>;
    using host_nonzero_type = matrix_data_entry<ValueType, IndexType>;
    static_assert(sizeof(nonzero_type) == sizeof(host_nonzero_type),
                  "mismatching size");
    thrust::sort(thrust::device,
                 thrust::device_pointer_cast(as_cuda_type(data.get_data())),
                 thrust::device_pointer_cast(
                     as_cuda_type(data.get_data() + data.get_num_elems())),
                 [] __device__(nonzero_type a, nonzero_type b) {
                     return thrust::tie(a.row, a.column) <
                            thrust::tie(b.row, b.column);
                 });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SORT_ROW_MAJOR_KERNEL);


}  // namespace components
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
