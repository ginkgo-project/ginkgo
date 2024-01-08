// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

namespace kernel {


constexpr auto searchtree_width = 1 << sampleselect_searchtree_height;
constexpr auto searchtree_inner_size = searchtree_width - 1;
constexpr auto searchtree_size = searchtree_width + searchtree_inner_size;

constexpr auto sample_size = searchtree_width * sampleselect_oversampling;

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
void build_searchtree(const ValueType* __restrict__ input, IndexType size,
                      remove_complex<ValueType>* __restrict__ tree_output,
                      sycl::nd_item<3> item_ct1,
                      remove_complex<ValueType>* sh_samples)
{
    using AbsType = remove_complex<ValueType>;
    auto idx = item_ct1.get_local_id(2);
    AbsType samples[sampleselect_oversampling];
    // assuming rounding towards zero
    auto stride = remove_complex<ValueType>(size) / sample_size;
#pragma unroll
    for (int i = 0; i < sampleselect_oversampling; ++i) {
        auto lidx = idx * sampleselect_oversampling + i;
        auto val = input[static_cast<IndexType>(lidx * stride)];
        samples[i] = std::abs(val);
    }

    bitonic_sort<sample_size, sampleselect_oversampling>(samples, sh_samples,
                                                         item_ct1);
    if (idx > 0) {
        // root has level 0
        auto level =
            sampleselect_searchtree_height - ffs(item_ct1.get_local_id(2));
        // we get the in-level index by removing trailing 10000...
        auto idx_in_level =
            item_ct1.get_local_id(2) >> ffs(item_ct1.get_local_id(2));
        // we get the global index by adding previous levels
        auto previous_levels = (1 << level) - 1;
        tree_output[idx_in_level + previous_levels] = samples[0];
    }
    tree_output[item_ct1.get_local_id(2) + searchtree_inner_size] = samples[0];
}

template <typename ValueType, typename IndexType>
void build_searchtree(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                      sycl::queue* queue, const ValueType* input,
                      IndexType size, remove_complex<ValueType>* tree_output)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<remove_complex<ValueType>, 1,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            sh_samples_acc_ct1(sycl::range<1>(1024 /*sample_size*/), cgh);

        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_size(config::warp_size)]] {
                                 build_searchtree(
                                     input, size, tree_output, item_ct1,
                                     sh_samples_acc_ct1.get_pointer());
                             });
    });
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
void count_buckets(const ValueType* __restrict__ input, IndexType size,
                   const remove_complex<ValueType>* __restrict__ tree,
                   IndexType* counter, unsigned char* oracles,
                   int items_per_thread, sycl::nd_item<3> item_ct1,
                   remove_complex<ValueType>* sh_tree, IndexType* sh_counter)
{
    // load tree into shared memory, initialize counters

    if (item_ct1.get_local_id(2) < searchtree_inner_size) {
        sh_tree[item_ct1.get_local_id(2)] = tree[item_ct1.get_local_id(2)];
    }
    if (item_ct1.get_local_id(2) < searchtree_width) {
        sh_counter[item_ct1.get_local_id(2)] = 0;
    }
    group::this_thread_block(item_ct1).sync();

    // work distribution: each thread block gets a consecutive index range
    auto begin = item_ct1.get_local_id(2) +
                 default_block_size *
                     static_cast<IndexType>(item_ct1.get_group(2)) *
                     items_per_thread;
    auto block_end = default_block_size *
                     static_cast<IndexType>(item_ct1.get_group(2) + 1) *
                     items_per_thread;
    auto end = min(block_end, size);
    for (IndexType i = begin; i < end; i += default_block_size) {
        // traverse the search tree with the input element
        auto el = std::abs(input[i]);
        IndexType tree_idx{};
#pragma unroll
        for (int level = 0; level < sampleselect_searchtree_height; ++level) {
            auto cmp = !(el < sh_tree[tree_idx]);
            tree_idx = 2 * tree_idx + 1 + cmp;
        }
        // increment the bucket counter and store the bucket index
        uint32 bucket = tree_idx - searchtree_inner_size;
        // post-condition: sample[bucket] <= el < sample[bucket + 1]
        atomic_add<atomic::local_space>(sh_counter + bucket, IndexType{1});
        oracles[i] = bucket;
    }
    group::this_thread_block(item_ct1).sync();

    // write back the block-wide counts to global memory
    if (item_ct1.get_local_id(2) < searchtree_width) {
        counter[item_ct1.get_group(2) +
                item_ct1.get_local_id(2) * item_ct1.get_group_range(2)] =
            sh_counter[item_ct1.get_local_id(2)];
    }
}

template <typename ValueType, typename IndexType>
void count_buckets(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                   sycl::queue* queue, const ValueType* input, IndexType size,
                   const remove_complex<ValueType>* tree, IndexType* counter,
                   unsigned char* oracles, int items_per_thread)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<remove_complex<ValueType>, 1,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            sh_tree_acc_ct1(sycl::range<1>(255 /*searchtree_inner_size*/), cgh);
        sycl::accessor<IndexType, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sh_counter_acc_ct1(sycl::range<1>(256 /*searchtree_width*/), cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                count_buckets(
                    input, size, tree, counter, oracles, items_per_thread,
                    item_ct1,
                    (remove_complex<ValueType>*)sh_tree_acc_ct1.get_pointer(),
                    (IndexType*)sh_counter_acc_ct1.get_pointer());
            });
    });
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
void block_prefix_sum(IndexType* __restrict__ counters,
                      IndexType* __restrict__ totals, IndexType num_blocks,
                      sycl::nd_item<3> item_ct1, IndexType* warp_sums)
{
    constexpr auto num_warps = default_block_size / config::warp_size;
    static_assert(num_warps < config::warp_size,
                  "block size needs to be smaller");


    auto block = group::this_thread_block(item_ct1);
    auto warp = group::tiled_partition<config::warp_size>(block);

    auto bucket = item_ct1.get_group(2);
    auto local_counters = counters + num_blocks * bucket;
    auto work_per_warp = ceildiv(num_blocks, warp.size());
    auto warp_idx = item_ct1.get_local_id(2) / warp.size();
    auto warp_lane = warp.thread_rank();

    // compute prefix sum over warp-sized blocks
    IndexType total{};
    auto base_idx = warp_idx * work_per_warp * warp.size();
    for (IndexType step = 0; step < work_per_warp; ++step) {
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
    for (IndexType step = 0; step < work_per_warp; ++step) {
        auto idx = warp_lane + step * warp.size() + base_idx;
        auto val = idx < num_blocks ? local_counters[idx] : zero<IndexType>();
        if (idx < num_blocks) {
            local_counters[idx] += warp_prefixsum;
        }
    }
}

template <typename IndexType>
void block_prefix_sum(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                      sycl::queue* queue, IndexType* counters,
                      IndexType* totals, IndexType num_blocks)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<IndexType, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            warp_sums_acc_ct1(sycl::range<1>(default_block_size /
                                             config::warp_size /*num_warps*/),
                              cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(config::warp_size)]] {
                    block_prefix_sum(
                        counters, totals, num_blocks, item_ct1,
                        (IndexType*)warp_sums_acc_ct1.get_pointer());
                });
    });
}


// must be launched with default_block_size >= searchtree_width
/**
 * @internal
 *
 * This copies all elements from a single bucket of the input to the output.
 */
template <typename ValueType, typename IndexType>
void filter_bucket(const ValueType* __restrict__ input, IndexType size,
                   unsigned char bucket, const unsigned char* oracles,
                   const IndexType* block_offsets,
                   remove_complex<ValueType>* __restrict__ output,
                   int items_per_thread, sycl::nd_item<3> item_ct1,
                   IndexType* counter)
{
    // initialize the counter with the block prefix sum.

    if (item_ct1.get_local_id(2) == 0) {
        *counter = block_offsets[item_ct1.get_group(2) +
                                 bucket * item_ct1.get_group_range(2)];
    }
    group::this_thread_block(item_ct1).sync();

    // same work-distribution as in count_buckets
    auto begin = item_ct1.get_local_id(2) +
                 default_block_size *
                     static_cast<IndexType>(item_ct1.get_group(2)) *
                     items_per_thread;
    auto block_end = default_block_size *
                     static_cast<IndexType>(item_ct1.get_group(2) + 1) *
                     items_per_thread;
    auto end = min(block_end, size);
    for (IndexType i = begin; i < end; i += default_block_size) {
        // only copy the element when it belongs to the target bucket
        auto found = bucket == oracles[i];
        auto ofs = atomic_add<atomic::local_space>(&*counter, IndexType{found});
        if (found) {
            output[ofs] = std::abs(input[i]);
        }
    }
}

template <typename ValueType, typename IndexType>
void filter_bucket(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                   sycl::queue* queue, const ValueType* input, IndexType size,
                   unsigned char bucket, const unsigned char* oracles,
                   const IndexType* block_offsets,
                   remove_complex<ValueType>* output, int items_per_thread)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<IndexType, 0, sycl::access_mode::read_write,
                       sycl::access::target::local>
            counter_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                filter_bucket(input, size, bucket, oracles, block_offsets,
                              output, items_per_thread, item_ct1,
                              (IndexType*)counter_acc_ct1.get_pointer());
            });
    });
}


/**
 * @internal
 *
 * Selects the `rank`th smallest element from a small array by sorting it.
 */
template <typename ValueType, typename IndexType>
void basecase_select(const ValueType* __restrict__ input, IndexType size,
                     IndexType rank, ValueType* __restrict__ out,
                     sycl::nd_item<3> item_ct1, ValueType* sh_local)
{
    constexpr auto sentinel = std::numeric_limits<ValueType>::infinity();
    ValueType local[basecase_local_size];

    for (int i = 0; i < basecase_local_size; ++i) {
        auto idx = item_ct1.get_local_id(2) + i * basecase_block_size;
        local[i] = idx < size ? input[idx] : sentinel;
    }
    bitonic_sort<basecase_size, basecase_local_size>(local, sh_local, item_ct1);
    if (item_ct1.get_local_id(2) == rank / basecase_local_size) {
        *out = local[rank % basecase_local_size];
    }
}

template <typename ValueType, typename IndexType>
void basecase_select(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                     sycl::queue* queue, const ValueType* input, IndexType size,
                     IndexType rank, ValueType* out)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<ValueType, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sh_local_acc_ct1(sycl::range<1>(1024 /*basecase_size*/), cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(config::warp_size)]] {
                    basecase_select(input, size, rank, out, item_ct1,
                                    (ValueType*)sh_local_acc_ct1.get_pointer());
                });
    });
}


/**
 * @internal
 *
 * Finds the bucket that contains the element with the given rank
 * and stores it and the bucket's base rank and size in the place of the prefix
 * sum.
 */
template <typename IndexType>
void find_bucket(IndexType* prefix_sum, IndexType rank,
                 sycl::nd_item<3> item_ct1)
{
    auto warp = group::tiled_partition<config::warp_size>(
        group::this_thread_block(item_ct1));
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

template <typename IndexType>
void find_bucket(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                 sycl::queue* queue, IndexType* prefix_sum, IndexType rank)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(config::warp_size)]] {
                                find_bucket(prefix_sum, rank, item_ct1);
                            });
}


}  // namespace kernel
