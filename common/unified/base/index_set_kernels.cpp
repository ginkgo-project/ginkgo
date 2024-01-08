// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/index_set_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The index_set namespace.
 *
 * @ingroup index_set
 */
namespace idx_set {


template <typename IndexType>
void compute_validity(std::shared_ptr<const DefaultExecutor> exec,
                      const array<IndexType>* local_indices,
                      array<bool>* validity_array)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto elem, auto local_indices, auto validity_array) {
            validity_array[elem] =
                local_indices[elem] != invalid_index<IndexType>();
        },
        local_indices->get_size(), *local_indices, *validity_array);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_INDEX_SET_COMPUTE_VALIDITY_KERNEL);


}  // namespace idx_set
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
