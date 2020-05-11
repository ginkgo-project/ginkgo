/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

namespace kernel {


constexpr auto searchtree_height = 8;
constexpr auto searchtree_width = 1 << searchtree_height;
constexpr auto searchtree_inner_size = searchtree_width - 1;
constexpr auto searchtree_size = searchtree_width + searchtree_inner_size;

constexpr auto oversampling_factor = 4;
constexpr auto sample_size = searchtree_width * oversampling_factor;

constexpr auto basecase_size = 1024;
constexpr auto basecase_local_size = 4;
constexpr auto basecase_block_size = basecase_size / basecase_local_size;


// must be launched with one thread block and block size == searchtree_width
/**
 * @internal
 *
 * Samples `searchtree_width - 1` uniformly distributed elements
 * and stores them in a binary search tree as splitters.
 */
template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(searchtree_width) void build_searchtree(
    const ValueType *__restrict__ input, IndexType size,
    remove_complex<ValueType> *__restrict__ tree_output)
{
    using AbsType = remove_complex<ValueType>;
    auto idx = threadIdx.x;
    AbsType samples[oversampling_factor];
    // assuming rounding towards zero
    auto stride = double(size) / sample_size;
#pragma unroll
    for (auto i = 0; i < oversampling_factor; ++i) {
        auto lidx = idx * oversampling_factor + i;
        auto val = input[static_cast<IndexType>(lidx * stride)];
        samples[i] = abs(val);
    }
    __shared__ AbsType sh_samples[sample_size];
    bitonic_sort<sample_size, oversampling_factor>(samples, sh_samples);
    if (idx > 0) {
        // root has level 0
        auto level = searchtree_height - ffs(threadIdx.x);
        // we get the in-level index by removing trailing 10000...
        auto idx_in_level = threadIdx.x >> ffs(threadIdx.x);
        // we get the global index by adding previous levels
        auto previous_levels = (1 << level) - 1;
        tree_output[idx_in_level + previous_levels] = samples[0];
    }
    tree_output[threadIdx.x + searchtree_inner_size] = samples[0];
}


// must be launched with default_block_size >= searchtree_width
/**
 * @internal
 *
 * Computes the number of elements in each of the buckets defined
 * by the splitter search tree. Stores the thread-block local
 * results packed by bucket idx.
 */
template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void count_buckets(
    const ValueType *__restrict__ input, IndexType size,
    const remove_complex<ValueType> *__restrict__ tree, IndexType *counter,
    unsigned char *oracles, int items_per_thread)
{
    // load tree into shared memory, initialize counters
    __shared__ remove_complex<ValueType> sh_tree[searchtree_inner_size];
    __shared__ IndexType sh_counter[searchtree_width];
    if (threadIdx.x < searchtree_inner_size) {
        sh_tree[threadIdx.x] = tree[threadIdx.x];
    }
    if (threadIdx.x < searchtree_width) {
        sh_counter[threadIdx.x] = 0;
    }
    group::this_thread_block().sync();

    // work distribution: each thread block gets a consecutive index range
    auto begin = threadIdx.x + default_block_size *
                                   static_cast<IndexType>(blockIdx.x) *
                                   items_per_thread;
    auto block_end = default_block_size *
                     static_cast<IndexType>(blockIdx.x + 1) * items_per_thread;
    auto end = min(block_end, size);
    for (IndexType i = begin; i < end; i += default_block_size) {
        // traverse the search tree with the input element
        auto el = abs(input[i]);
        IndexType tree_idx{};
#pragma unroll
        for (auto level = 0; level < searchtree_height; ++level) {
            auto cmp = !(el < sh_tree[tree_idx]);
            tree_idx = 2 * tree_idx + 1 + cmp;
        }
        // increment the bucket counter and store the bucket index
        uint32 bucket = tree_idx - searchtree_inner_size;
        // post-condition: sample[bucket] <= el < sample[bucket + 1]
        atomic_add<IndexType>(sh_counter + bucket, 1);
        oracles[i] = bucket;
    }
    group::this_thread_block().sync();

    // write back the block-wide counts to global memory
    if (threadIdx.x < searchtree_width) {
        counter[blockIdx.x + threadIdx.x * gridDim.x] = sh_counter[threadIdx.x];
    }
}


// must be launched with default_block_size threads per block
/**
 * @internal
 *
 * Simultaneously computes a prefix and total sum of the block-local counts for
 * each bucket. The results are then used as base offsets for the following
 * filter step.
 */
template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void block_prefix_sum(
    IndexType *__restrict__ counters, IndexType *__restrict__ totals,
    IndexType num_blocks)
{
    constexpr auto num_warps = default_block_size / config::warp_size;
    static_assert(num_warps < config::warp_size,
                  "block size needs to be smaller");
    __shared__ IndexType warp_sums[num_warps];

    auto block = group::this_thread_block();
    auto warp = group::tiled_partition<config::warp_size>(block);

    auto bucket = blockIdx.x;
    auto local_counters = counters + num_blocks * bucket;
    auto work_per_warp = ceildiv(num_blocks, warp.size());
    auto warp_idx = threadIdx.x / warp.size();
    auto warp_lane = warp.thread_rank();

    // compute prefix sum over warp-sized blocks
    IndexType total{};
    auto base_idx = warp_idx * work_per_warp * warp.size();
    for (auto step = 0; step < work_per_warp; ++step) {
        auto idx = warp_lane + step * warp.size() + base_idx;
        auto val = idx < num_blocks ? local_counters[idx] : zero<IndexType>();
        IndexType warp_total{};
        IndexType warp_prefix{};
        // compute inclusive prefix sum
        subwarp_prefix_sum<false>(val, warp_prefix, warp_total, warp);

        if (idx < num_blocks) {
            local_counters[idx] = warp_prefix + total;
        }
        total += warp_total;
    }

    // store total sum
    if (warp_lane == 0) {
        warp_sums[warp_idx] = total;
    }

    // compute prefix sum over all warps in a single warp
    block.sync();
    if (warp_idx == 0) {
        auto in_bounds = warp_lane < num_warps;
        auto val = in_bounds ? warp_sums[warp_lane] : zero<IndexType>();
        IndexType prefix_sum{};
        IndexType total_sum{};
        // compute inclusive prefix sum
        subwarp_prefix_sum<false>(val, prefix_sum, total_sum, warp);
        if (in_bounds) {
            warp_sums[warp_lane] = prefix_sum;
        }
        if (warp_lane == 0) {
            totals[bucket] = total_sum;
        }
    }

    // add block prefix sum to each warp's block of data
    block.sync();
    auto warp_prefixsum = warp_sums[warp_idx];
    for (auto step = 0; step < work_per_warp; ++step) {
        auto idx = warp_lane + step * warp.size() + base_idx;
        auto val = idx < num_blocks ? local_counters[idx] : zero<IndexType>();
        if (idx < num_blocks) {
            local_counters[idx] += warp_prefixsum;
        }
    }
}


// must be launched with default_block_size >= searchtree_width
/**
 * @internal
 *
 * This copies all elements from a single bucket of the input to the output.
 */
template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void filter_bucket(
    const ValueType *__restrict__ input, IndexType size, unsigned char bucket,
    const unsigned char *oracles, const IndexType *block_offsets,
    remove_complex<ValueType> *__restrict__ output, int items_per_thread)
{
    // initialize the counter with the block prefix sum.
    __shared__ IndexType counter;
    if (threadIdx.x == 0) {
        counter = block_offsets[blockIdx.x + bucket * gridDim.x];
    }
    group::this_thread_block().sync();

    // same work-distribution as in count_buckets
    auto begin = threadIdx.x + default_block_size *
                                   static_cast<IndexType>(blockIdx.x) *
                                   items_per_thread;
    auto block_end = default_block_size *
                     static_cast<IndexType>(blockIdx.x + 1) * items_per_thread;
    auto end = min(block_end, size);
    for (IndexType i = begin; i < end; i += default_block_size) {
        // only copy the element when it belongs to the target bucket
        auto found = bucket == oracles[i];
        auto ofs = atomic_add<IndexType>(&counter, found);
        if (found) {
            output[ofs] = abs(input[i]);
        }
    }
}


/**
 * @internal
 *
 * Selects the `rank`th smallest element from a small array by sorting it.
 */
template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(basecase_block_size) void basecase_select(
    const ValueType *__restrict__ input, IndexType size, IndexType rank,
    ValueType *__restrict__ out)
{
    constexpr auto sentinel = device_numeric_limits<ValueType>::inf;
    ValueType local[basecase_local_size];
    __shared__ ValueType sh_local[basecase_size];
    for (int i = 0; i < basecase_local_size; ++i) {
        auto idx = threadIdx.x + i * basecase_block_size;
        local[i] = idx < size ? input[idx] : sentinel;
    }
    bitonic_sort<basecase_size, basecase_local_size>(local, sh_local);
    if (threadIdx.x == rank / basecase_local_size) {
        *out = local[rank % basecase_local_size];
    }
}


/**
 * @internal
 *
 * Finds the bucket that contains the element with the given rank
 * and stores it and the bucket's base rank and size in the place of the prefix
 * sum.
 */
template <typename IndexType>
__global__ __launch_bounds__(config::warp_size) void find_bucket(
    IndexType *prefix_sum, IndexType rank)
{
    auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    auto idx = group_wide_search(0, searchtree_width, warp, [&](int i) {
        return prefix_sum[i + 1] > rank;
    });
    if (warp.thread_rank() == 0) {
        auto base = prefix_sum[idx];
        auto size = prefix_sum[idx + 1] - base;
        // don't overwrite anything before having loaded everything!
        prefix_sum[0] = idx;
        prefix_sum[1] = base;
        prefix_sum[2] = size;
    }
}


}  // namespace kernel