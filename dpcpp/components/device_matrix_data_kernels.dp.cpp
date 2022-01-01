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

// force-top: on
// oneDPL needs to be first to avoid issues with libstdc++ TBB impl
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
// force-top: off


#include "core/components/device_matrix_data_kernels.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace components {


template <typename ValueType, typename IndexType>
void remove_zeros(std::shared_ptr<const DefaultExecutor> exec,
                  Array<matrix_data_entry<ValueType, IndexType>>& data)
{
    using nonzero_type = matrix_data_entry<ValueType, IndexType>;
    auto size = data.get_num_elems();
    auto policy =
        oneapi::dpl::execution::make_device_policy(*exec->get_queue());
    auto nnz = std::count_if(
        policy, data.get_const_data(), data.get_const_data() + size,
        [](nonzero_type entry) { return entry.value != zero<ValueType>(); });
    if (nnz < size) {
        Array<nonzero_type> result{exec, static_cast<size_type>(nnz)};
        std::copy_if(policy, data.get_const_data(),
                     data.get_const_data() + size, result.get_data(),
                     [](nonzero_type entry) {
                         return entry.value != zero<ValueType>();
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
    using nonzero_type = matrix_data_entry<ValueType, IndexType>;
    auto policy =
        oneapi::dpl::execution::make_device_policy(*exec->get_queue());
    std::sort(policy, data.get_data(), data.get_data() + data.get_num_elems(),
              [](nonzero_type a, nonzero_type b) {
                  return std::tie(a.row, a.column) < std::tie(b.row, b.column);
              });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SORT_ROW_MAJOR_KERNEL);


}  // namespace components
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
