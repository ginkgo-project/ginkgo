// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
void compose(std::shared_ptr<const DefaultExecutor> exec,
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
    GKO_DECLARE_SCALED_PERMUTATION_COMPOSE_KERNEL);


}  // namespace scaled_permutation
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
