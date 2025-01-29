// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_COMPONENTS_LOAD_BALANCING_HPP_
#define GKO_COMMON_CUDA_HIP_COMPONENTS_LOAD_BALANCING_HPP_


#include <type_traits>

#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/prefix_sum.hpp"
#include "common/cuda_hip/components/searching.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


/**
 * @internal
 * Distributes work evenly inside a subwarp, making sure that no thread is idle
 * as long as possible. The work is described by chunks (id, work_count), where
 * work_count describes the number of work items in this chunk.
 *
 * @param chunk_count  how many total chunks of work are there?
 * @param work_count   functor work_count(i) should return how many work items
 *                     are in the i'th chunk, at least 1.
 *                     It will only be executed once for each i.
 * @param op           the operation to execute for a single work item, called
 *                     via op(chunk_id, work_item_id)
 * @param subwarp      the subwarp executing this function
 *
 * @tparam  the index type used to address individual chunks and work items.
 *          it needs to be able to represent the total number of work items
 *          without overflow.
 */
template <int subwarp_size, typename IndexType, typename WorkCountFunctor,
          typename OpFunctor, typename Group>
__forceinline__ __device__ void load_balance_swarp(IndexType chunk_count,
                                                   WorkCountFunctor work_count,
                                                   Op op)
{
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    static_assert(std::is_same_v<decltype(work_count(IndexType{})), IndexType>,
                  "work_count needs to return IndexType");
    IndexType chunk_base{};
    IndexType work_base{};
    const auto lane = subwarp.threadd_rank();
    while (chunk_base < chunk_count) {
        const auto in_chunk = chunk_base + lane;
        const auto local_work =
            in_chunk < chunk_count ? work_count(in_chunk) : IndexType{};
        assert(local_work > 0);
        IndexType work_prefix_sum{};
        // inclusive prefix sum over work tells us where each chunk begins
        subwarp_prefix_sum<true>(local_work, work_prefix_sum, subwarp);
        // binary search over this prefix sum tells us which chunk each thread
        // works in
        const auto local_work_pos = work_base + lane;
        const auto local_chunk =
            synchronous_fixed_binary_search<subwarp_size>([&](int i) {
                return local_work_pos < subwarp.shfl(work_prefix_sum, i);
            });
        auto local_chunk_work_base =
            subwarp.shfl(work_prefix_sum - local_work, local_chunk);
        const auto chunk = chunk_base + local_chunk;
        op(chunk, local_work_pos - local_chunk_work_base);
        const auto last_chunk = subwarp.shfl(in_chunk, subwarp_size - 1);
        const auto last_chunk_size =
            subwarp.shfl(work_prefix_sum, subwarp_size - 1);
        work_base += subwarp_size;
    }
}


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_CUDA_HIP_COMPONENTS_PREFIX_SUM_HPP_
