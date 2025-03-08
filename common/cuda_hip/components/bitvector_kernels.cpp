// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/bitvector_kernels.hpp"

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

#include <ginkgo/core/base/intrinsics.hpp>

#include "common/cuda_hip/base/thrust.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace bitvector {


template <typename IndexType>
void compute_bits_and_ranks(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* indices,
    IndexType num_indices, IndexType size,
    typename device_bitvector<IndexType>::storage_type* bits, IndexType* ranks)
{
    const auto policy = thrust_policy(exec);
    using bv = device_bitvector<IndexType>;
    using storage_type = typename bv::storage_type;
    const auto num_blocks = ceildiv(size, bv::block_size);
    thrust::fill_n(policy, bits, num_blocks, 0u);
    thrust::for_each_n(
        policy, indices, num_indices, [bits] __device__(IndexType idx) {
            constexpr auto block_size = device_bitvector<IndexType>::block_size;
            const auto block = idx / block_size;
            const auto local = idx % block_size;
            atomicOr(bits + block, storage_type{1} << local);
        });
    const auto it = thrust::make_transform_iterator(
        bits, [] __device__(storage_type word) -> IndexType {
            return gko::detail::popcount(word);
        });
    thrust::exclusive_scan(policy, it, it + num_blocks, ranks, IndexType{});
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_BITVECTOR_COMPUTE_BITS_AND_RANKS_KERNEL);


}  // namespace bitvector
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
