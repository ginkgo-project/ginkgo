// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/permutation_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace permutation {


template <typename IndexType>
void invert(std::shared_ptr<const DefaultExecutor> exec,
            const IndexType* permutation, size_type size,
            IndexType* output_permutation)
{
    for (size_type i = 0; i < size; i++) {
        output_permutation[permutation[i]] = i;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PERMUTATION_INVERT_KERNEL);


template <typename IndexType>
void compose(std::shared_ptr<const DefaultExecutor> exec,
             const IndexType* first_permutation,
             const IndexType* second_permutation, size_type size,
             IndexType* output_permutation)
{
    // P_2 P_1 does a row permutation of P_1 with indices from P_2
    // row i of P_2 P_1 x accesses row P_2[i] of P_1 x = row P_1[P_2[i]] of x
    for (size_type i = 0; i < size; i++) {
        output_permutation[i] = first_permutation[second_permutation[i]];
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PERMUTATION_COMPOSE_KERNEL);


}  // namespace permutation
}  // namespace reference
}  // namespace kernels
}  // namespace gko
