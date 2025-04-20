// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/format_conversion_kernels.hpp"

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <ginkgo/core/base/types.hpp>

#include "common/cuda_hip/base/thrust.hpp"
#include "common/cuda_hip/components/bitvector_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace components {


template <typename IndexType, typename RowPtrType>
void convert_ptrs_to_idxs(std::shared_ptr<const DefaultExecutor> exec,
                          const RowPtrType* ptrs, size_type num_blocks,
                          IndexType* idxs)
{
    const auto policy = thrust_policy(exec);
    const auto num_elements = exec->copy_val_to_host(ptrs + num_blocks);
    // transform the ptrs to a bitvector in unary delta encoding, i.e.
    // every row with n elements is encoded as 1 0 ... n times ... 0
    auto it = thrust::make_transform_iterator(
        thrust::make_counting_iterator(IndexType{}),
        [ptrs] __device__(IndexType i) -> RowPtrType { return ptrs[i] + i; });
    auto bv = bitvector::bitvector_from_sorted_indices(
        exec, it, num_blocks, num_blocks + num_elements);
    auto device_bv = bv.device_view();
    thrust::for_each_n(policy, thrust::make_counting_iterator(IndexType{}),
                       num_blocks + num_elements,
                       [device_bv, idxs] __device__(RowPtrType i) {
                           if (!device_bv.get(i)) {
                               auto rank = device_bv.rank(i);
                               idxs[i - rank] = rank - 1;
                           }
                       });
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_PTRS_TO_IDXS32);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_PTRS_TO_IDXS64);


}  // namespace components
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
