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

#include "core/matrix/scaled_permutation_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace scaled_permutation {


template <typename ValueType, typename IndexType>
void invert(std::shared_ptr<const DefaultExecutor> exec,
            const ValueType* input_scale, const IndexType* input_permutation,
            size_type size, ValueType* output_scale,
            IndexType* output_permutation)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto input_scale, auto input_permutation,
                      auto output_scale, auto output_permutation) {
            const auto ip = input_permutation[i];
            output_permutation[ip] = i;
            output_scale[i] = one(input_scale[ip]) / input_scale[ip];
        },
        size, input_scale, input_permutation, output_scale, output_permutation);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SCALED_PERMUTATION_INVERT_KERNEL);


template <typename ValueType, typename IndexType>
void combine(std::shared_ptr<const DefaultExecutor> exec,
             const ValueType* first_scale, const IndexType* first_permutation,
             const ValueType* second_scale, const IndexType* second_permutation,
             size_type size, ValueType* output_scale,
             IndexType* output_permutation)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto first_scale, auto first_permutation,
                      auto second_scale, auto second_permutation,
                      auto output_permutation, auto output_scale) {
            const auto second_permuted = second_permutation[i];
            const auto combined_permuted = first_permutation[second_permuted];
            output_permutation[i] = combined_permuted;
            output_scale[combined_permuted] =
                first_scale[combined_permuted] * second_scale[second_permuted];
        },
        size, first_scale, first_permutation, second_scale, second_permutation,
        output_permutation, output_scale);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SCALED_PERMUTATION_COMBINE_KERNEL);


}  // namespace scaled_permutation
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
