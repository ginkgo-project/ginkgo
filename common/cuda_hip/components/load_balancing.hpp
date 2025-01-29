// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_COMPONENTS_LOAD_BALANCING_HPP_
#define GKO_COMMON_CUDA_HIP_COMPONENTS_LOAD_BALANCING_HPP_

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
          typename OpFunctor>
__forceinline__ __device__ void load_balance_subwarp_nonempty(
    IndexType chunk_count, WorkCountFunctor work_count, OpFunctor op)
{
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    static_assert(std::is_same_v<decltype(work_count(IndexType{})), IndexType>,
                  "work_count needs to return IndexType");
    const auto lane = subwarp.thread_rank();
    IndexType chunk_base{};
    IndexType work_base{};
    IndexType local_work = lane < chunk_count ? work_count(lane) : IndexType{1};
    // inclusive prefix sum over work tells us where each chunk begins
    IndexType work_prefix_sum{};
    subwarp_prefix_sum<true>(local_work, work_prefix_sum, subwarp);
    while (chunk_base < chunk_count) {
        assert(local_work > 0);
        // binary search over this prefix sum tells us which chunk each thread
        // works in
        const auto local_work_pos = work_base + lane;
        const auto local_chunk =
            synchronous_fixed_binary_search<subwarp_size>([&](int i) {
                return local_work_pos < subwarp.shfl(work_prefix_sum, i);
            });
        assert(local_chunk < subwarp_size);
        auto local_chunk_work_base =
            subwarp.shfl(work_prefix_sum - local_work, local_chunk);
        const auto chunk = chunk_base + local_chunk;
        // do the work inside this chunk
        if (chunk < chunk_count) {
            op(chunk, local_work_pos - local_chunk_work_base, local_work_pos);
        }
        const auto last_local_chunk =
            subwarp.shfl(local_chunk, subwarp_size - 1);
        const auto last_local_chunk_end =
            subwarp.shfl(work_prefix_sum, last_local_chunk);
        assert(last_local_chunk < subwarp_size);
        assert(last_local_chunk_end > local_work_pos);
        work_base += subwarp_size;
        // how many chunks have we completed? The last one is completed if its
        // end matches work_base after the update
        const auto chunk_advance =
            last_local_chunk + (last_local_chunk_end == work_base ? 1 : 0);
        chunk_base += chunk_advance;
        // shift down local_work and work_prefix_sum,
        // adding new values when necessary
        local_work = subwarp.shfl_down(local_work, chunk_advance);
        // find the last value of the prefix sum and remember it for later
        const auto work_prefix_sum_end =
            subwarp.shfl(work_prefix_sum, subwarp_size - 1);
        // this shuffle leaves the trailing elements unchaged, we need to
        // overwrite them later
        work_prefix_sum = subwarp.shfl_down(work_prefix_sum, chunk_advance);
        IndexType work_prefix_sum_add{};
        if (lane >= subwarp_size - chunk_advance) {
            const auto in_chunk = chunk_base + lane;
            // load new work counters at the end
            local_work =
                in_chunk < chunk_count ? work_count(in_chunk) : IndexType{1};
            work_prefix_sum_add = local_work;
            // fill the trailing work_prefix_sum with the last element
            work_prefix_sum = work_prefix_sum_end;
        }
        // compute a prefix sum over new chunks and add to the prefix sum
        subwarp_prefix_sum<true>(work_prefix_sum_add, work_prefix_sum_add,
                                 subwarp);
        work_prefix_sum += work_prefix_sum_add;
    }
}


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_CUDA_HIP_COMPONENTS_PREFIX_SUM_HPP_
