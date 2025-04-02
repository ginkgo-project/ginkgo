// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_COMPONENTS_BITVECTOR_KERNELS_HPP_
#define GKO_COMMON_CUDA_HIP_COMPONENTS_BITVECTOR_KERNELS_HPP_

#include "core/components/bitvector.hpp"

#include <ginkgo/core/base/intrinsics.hpp>

#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace bitvector {


constexpr auto default_block_size = 512;


namespace kernel {


template <typename IndexType, typename DevicePredicate>
__global__ __launch_bounds__(default_block_size) void bitvector_from_predicate(
    IndexType size,
    typename device_bitvector<IndexType>::storage_type* __restrict__ bits,
    IndexType* __restrict__ popcounts, DevicePredicate predicate)
{
    constexpr auto block_size = device_bitvector<IndexType>::block_size;
    const auto subwarp_id = thread::get_subwarp_id_flat<block_size>();
    const auto subwarp_base = subwarp_id * block_size;
    if (subwarp_base >= size) {
        return;
    }
    const auto subwarp =
        group::tiled_partition<block_size>(group::this_thread_block());
    const auto i = static_cast<IndexType>(subwarp_base + subwarp.thread_rank());
    const auto bit = i < size ? predicate(i) : false;
    const auto mask = subwarp.ballot(bit);
    if (subwarp.thread_rank() == 0) {
        bits[subwarp_id] = mask;
        popcounts[subwarp_id] = gko::detail::popcount(mask);
    }
}


}  // namespace kernel


template <typename IndexType, typename DevicePredicate>
gko::bitvector<IndexType> bitvector_from_predicate(
    std::shared_ptr<const DefaultExecutor> exec, IndexType size,
    DevicePredicate device_predicate)
{
    constexpr auto block_size = device_bitvector<IndexType>::block_size;
    const auto num_blocks = static_cast<size_type>(ceildiv(size, block_size));
    array<uint32> bits{exec, num_blocks};
    array<IndexType> ranks{exec, num_blocks};
    if (num_blocks > 0) {
        const auto num_threadblocks =
            ceildiv(num_blocks, default_block_size / block_size);
        kernel::bitvector_from_predicate<<<num_threadblocks, default_block_size,
                                           0, exec->get_stream()>>>(
            size, bits.get_data(), ranks.get_data(), device_predicate);
        components::prefix_sum_nonnegative(exec, ranks.get_data(), num_blocks);
    }

    return gko::bitvector<IndexType>{std::move(bits), std::move(ranks), size};
}


}  // namespace bitvector
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_CUDA_HIP_COMPONENTS_BITVECTOR_KERNELS_HPP_
