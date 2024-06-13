// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/permutation_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace permutation {


template <typename IndexType>
void invert(std::shared_ptr<const DefaultExecutor> exec,
            const IndexType* permutation_indices, size_type size,
            IndexType* inv_permutation)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto permutation, auto inv_permutation) {
            inv_permutation[permutation[i]] = i;
        },
        size, permutation_indices, inv_permutation);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PERMUTATION_INVERT_KERNEL);


template <typename IndexType>
void compose(std::shared_ptr<const DefaultExecutor> exec,
             const IndexType* first_permutation,
             const IndexType* second_permutation, size_type size,
             IndexType* output_permutation)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto first_permutation, auto second_permutation,
                      auto output_permutation) {
            output_permutation[i] = first_permutation[second_permutation[i]];
        },
        size, first_permutation, second_permutation, output_permutation);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PERMUTATION_COMPOSE_KERNEL);


}  // namespace permutation
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
