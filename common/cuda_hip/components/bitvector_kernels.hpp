// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_COMPONENTS_BITVECTOR_KERNELS_HPP_
#define GKO_COMMON_CUDA_HIP_COMPONENTS_BITVECTOR_KERNELS_HPP_

#include "core/components/bitvector.hpp"

#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>

#include <ginkgo/core/base/intrinsics.hpp>

#include "common/cuda_hip/base/thrust.hpp"
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


template <typename IndexType>
struct bitvector_bit_functor {
    using storage_type = typename device_bitvector<IndexType>::storage_type;
    constexpr static auto block_size = device_bitvector<IndexType>::block_size;
    __device__ storage_type operator()(IndexType i)
    {
        return storage_type{1} << (i % block_size);
    }

    __device__ storage_type operator()(storage_type a, storage_type b)
    {
        // there must not be any duplicate indices
        assert(a ^ b == 0);
        return a | b;
    }
};


template <typename IndexType>
struct bitvector_block_functor {
    constexpr static auto block_size = device_bitvector<IndexType>::block_size;
    __device__ IndexType operator()(IndexType i)
    {
        assert(i >= 0);
        assert(i < size);
        return i / block_size;
    }

    IndexType size;
};


template <typename IndexType>
struct bitvector_popcnt_functor {
    using storage_type = typename device_bitvector<IndexType>::storage_type;
    __device__ IndexType operator()(storage_type mask)
    {
        return gko::detail::popcount(mask);
    }
};


template <typename IndexIterator>
gko::bitvector<typename std::iterator_traits<IndexIterator>::value_type>
bitvector_from_sorted_indices(
    std::shared_ptr<const DefaultExecutor> exec, IndexIterator it,
    typename std::iterator_traits<IndexIterator>::difference_type count,
    typename std::iterator_traits<IndexIterator>::value_type size)
{
    using index_type = typename std::iterator_traits<IndexIterator>::value_type;
    using storage_type = typename device_bitvector<index_type>::storage_type;
    constexpr auto block_size = device_bitvector<index_type>::block_size;
    const auto num_blocks = static_cast<size_type>(ceildiv(size, block_size));
    const auto policy = thrust_policy(exec);
    array<storage_type> bits_compact{exec, num_blocks};
    array<index_type> bits_position{exec, num_blocks};
    array<storage_type> bits{exec, num_blocks};
    array<index_type> ranks{exec, num_blocks};
    const auto block_it = thrust::make_transform_iterator(
        it, bitvector_block_functor<index_type>{size});
    const auto bit_it = thrust::make_transform_iterator(
        it, bitvector_bit_functor<index_type>{});
    auto out_pos_it = bits_position.get_data();
    auto out_bit_it = bits_compact.get_data();
    auto [out_pos_end, out_bit_end] = thrust::reduce_by_key(
        policy, block_it, block_it + count, bit_it, out_pos_it, out_bit_it,
        thrust::equal_to<index_type>{}, bitvector_bit_functor<storage_type>{});
    assert(thrust::is_sorted(policy, out_pos_it, out_pos_end));
    const auto out_size = out_pos_end - out_pos_it;
    thrust::fill_n(policy, bits.get_data(), num_blocks, 0);
    thrust::scatter(policy, out_bit_it, out_bit_it + out_size, out_pos_it,
                    bits.get_data());
    const auto rank_it = thrust::make_transform_iterator(
        bits.get_const_data(), bitvector_popcnt_functor<index_type>{});
    thrust::exclusive_scan(policy, rank_it, rank_it + num_blocks,
                           ranks.get_data(), index_type{});

    return gko::bitvector<index_type>{std::move(bits), std::move(ranks), size};
}


}  // namespace bitvector
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_CUDA_HIP_COMPONENTS_BITVECTOR_KERNELS_HPP_
