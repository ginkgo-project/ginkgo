/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "core/distributed/matrix_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Csr matrix format namespace.
 *
 * @ingroup csr
 */
namespace distributed_matrix {


template <typename LocalIndexType>
void check_indices_within_span(std::shared_ptr<const DefaultExecutor> exec,
                               const Array<LocalIndexType>& indices,
                               const Array<global_index_type>& to_global,
                               gko::span valid_span,
                               Array<bool>& index_is_valid)
{
    auto transform = [] GKO_KERNEL(const auto i, const auto* indices,
                                   const auto* to_global, const auto valid_span,
                                   auto* index_is_valid) {
        const auto idx = indices[i];
        auto global_idx = to_global[idx];
        index_is_valid[i] =
            valid_span.begin <= global_idx && global_idx < valid_span.end;
    };
    index_is_valid.resize_and_reset(indices.get_num_elems());
    run_kernel(exec, transform, indices.get_num_elems(),
               indices.get_const_data(), to_global.get_const_data(), valid_span,
               index_is_valid.get_data());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CHECK_INDICES_WITHIN_SPAN);


template <typename ValueType, typename IndexType>
void zero_out_invalid_columns(std::shared_ptr<const DefaultExecutor> exec,
                              const Array<bool>& column_index_is_valid,
                              device_matrix_data<ValueType, IndexType>& data)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(const auto i, const auto* column_index_is_valid,
                      auto* data) {
            auto col_idx = data[i].column;
            if (!column_index_is_valid[col_idx]) {
                data[i].value = zero(data[i].value);
            }
        },
        data.nonzeros.get_num_elems(), column_index_is_valid.get_const_data(),
        data.nonzeros.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ZERO_OUT_INVALID_COLUMNS);


template <typename ValueType>
void add_to_array(std::shared_ptr<const DefaultExecutor> exec,
                  Array<ValueType>& array, const ValueType value)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(const auto i, auto* data, const auto value) {
            data[i] += value;
        },
        array.get_num_elems(), array.get_data(), value);
}

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_ADD_TO_ARRAY);


}  // namespace distributed_matrix
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
