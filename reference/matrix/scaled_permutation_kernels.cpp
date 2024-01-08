// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/scaled_permutation_kernels.hpp"


#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace kernels {
namespace reference {
namespace scaled_permutation {


template <typename ValueType, typename IndexType>
void invert(std::shared_ptr<const DefaultExecutor> exec,
            const ValueType* input_scale, const IndexType* input_permutation,
            size_type size, ValueType* output_scale,
            IndexType* output_permutation)
{
    for (size_type i = 0; i < size; i++) {
        const auto ip = input_permutation[i];
        output_permutation[ip] = i;
        output_scale[i] = one<ValueType>() / input_scale[ip];
    }
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
    // P_2 S_2 P_1 S_1 = P_2 P_1 S'_2 S_1 with S'_2 = P_1^-1 S_2 P_1^-T
    // P_2 P_1 does a row permutation of P_1 with indices from P_2
    // row i of P_2 P_1 x accesses row P_2[i] of P_1 x = row P_1[P_2[i]] of x
    for (size_type i = 0; i < size; i++) {
        const auto second_permuted = second_permutation[i];
        const auto combined_permuted = first_permutation[second_permuted];
        output_permutation[i] = combined_permuted;
        // output_scale[i] = first_scale[i] * second_scale[inv_first_perm[i]];
        // second_perm[i] = inv_first_perm[combined_perm[i]];
        output_scale[combined_permuted] =
            first_scale[combined_permuted] * second_scale[second_permuted];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SCALED_PERMUTATION_COMPOSE_KERNEL);


}  // namespace scaled_permutation
}  // namespace reference
}  // namespace kernels
}  // namespace gko
