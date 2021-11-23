/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "core/factorization/par_ilut_kernels.hpp"


#include <algorithm>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/coo_builder.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/math.hpp"
#include "dpcpp/components/atomic.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/intrinsics.dp.hpp"
#include "dpcpp/components/prefix_sum.dp.hpp"
#include "dpcpp/components/sorting.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/factorization/par_ilut_select_common.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The parallel ILUT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


// subwarp sizes for filter kernels
using compiled_kernels = syn::value_list<int, 1, 8, 16, 32>;


namespace kernel {


template <int subwarp_size, typename IndexType, typename Predicate,
          typename BeginCallback, typename StepCallback,
          typename FinishCallback>
void abstract_filter_impl(const IndexType* row_ptrs, IndexType num_rows,
                          Predicate pred, BeginCallback begin_cb,
                          StepCallback step_cb, FinishCallback finish_cb,
                          sycl::nd_item<3> item_ct1)
{
    auto subwarp = group::tiled_partition<subwarp_size>(
        group::this_thread_block(item_ct1));
    auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>(item_ct1);
    auto lane = subwarp.thread_rank();
    auto lane_prefix_mask = (config::lane_mask_type(1) << lane) - 1;
    if (row >= num_rows) {
        return;
    }

    auto begin = row_ptrs[row];
    auto end = row_ptrs[row + 1];
    begin_cb(row);
    auto num_steps = ceildiv(end - begin, subwarp_size);
    for (IndexType step = 0; step < num_steps; ++step) {
        auto idx = begin + lane + step * subwarp_size;
        auto keep = idx < end && pred(idx, begin, end);
        auto mask = subwarp.ballot(keep);
        step_cb(row, idx, keep, popcnt(mask), popcnt(mask & lane_prefix_mask));
    }
    finish_cb(row, lane);
}


template <int subwarp_size, typename Predicate, typename IndexType>
void abstract_filter_nnz(const IndexType* __restrict__ row_ptrs,
                         IndexType num_rows, Predicate pred,
                         IndexType* __restrict__ nnz, sycl::nd_item<3> item_ct1)
{
    IndexType count{};
    abstract_filter_impl<subwarp_size>(
        row_ptrs, num_rows, pred, [&](IndexType) { count = 0; },
        [&](IndexType, IndexType, bool, IndexType warp_count, IndexType) {
            count += warp_count;
        },
        [&](IndexType row, IndexType lane) {
            if (row < num_rows && lane == 0) {
                nnz[row] = count;
            }
        },
        item_ct1);
}


template <int subwarp_size, typename Predicate, typename IndexType,
          typename ValueType>
void abstract_filter(const IndexType* __restrict__ old_row_ptrs,
                     const IndexType* __restrict__ old_col_idxs,
                     const ValueType* __restrict__ old_vals, IndexType num_rows,
                     Predicate pred, const IndexType* __restrict__ new_row_ptrs,
                     IndexType* __restrict__ new_row_idxs,
                     IndexType* __restrict__ new_col_idxs,
                     ValueType* __restrict__ new_vals,
                     sycl::nd_item<3> item_ct1)
{
    IndexType count{};
    IndexType new_offset{};
    abstract_filter_impl<subwarp_size>(
        old_row_ptrs, num_rows, pred,
        [&](IndexType row) {
            new_offset = new_row_ptrs[row];
            count = 0;
        },
        [&](IndexType row, IndexType idx, bool keep, IndexType warp_count,
            IndexType warp_prefix_sum) {
            if (keep) {
                auto new_idx = new_offset + warp_prefix_sum + count;
                if (new_row_idxs) {
                    new_row_idxs[new_idx] = row;
                }
                new_col_idxs[new_idx] = old_col_idxs[idx];
                new_vals[new_idx] = old_vals[idx];
            }
            count += warp_count;
        },
        [](IndexType, IndexType) {}, item_ct1);
}


template <int subwarp_size, typename ValueType, typename IndexType>
void threshold_filter_nnz(const IndexType* __restrict__ row_ptrs,
                          const ValueType* vals, IndexType num_rows,
                          remove_complex<ValueType> threshold,
                          IndexType* __restrict__ nnz, bool lower,
                          sycl::nd_item<3> item_ct1)
{
    abstract_filter_nnz<subwarp_size>(
        row_ptrs, num_rows,
        [&](IndexType idx, IndexType row_begin, IndexType row_end) {
            auto diag_idx = lower ? row_end - 1 : row_begin;
            return abs(vals[idx]) >= threshold || idx == diag_idx;
        },
        nnz, item_ct1);
}

template <int subwarp_size, typename ValueType, typename IndexType>
void threshold_filter_nnz(dim3 grid, dim3 block,
                          size_type dynamic_shared_memory, sycl::queue* queue,
                          const IndexType* row_ptrs, const ValueType* vals,
                          IndexType num_rows,
                          remove_complex<ValueType> threshold, IndexType* nnz,
                          bool lower)
{
    queue->parallel_for(
        sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
            threshold_filter_nnz<subwarp_size>(row_ptrs, vals, num_rows,
                                               threshold, nnz, lower, item_ct1);
        });
}


template <int subwarp_size, typename ValueType, typename IndexType>
void threshold_filter(const IndexType* __restrict__ old_row_ptrs,
                      const IndexType* __restrict__ old_col_idxs,
                      const ValueType* __restrict__ old_vals,
                      IndexType num_rows, remove_complex<ValueType> threshold,
                      const IndexType* __restrict__ new_row_ptrs,
                      IndexType* __restrict__ new_row_idxs,
                      IndexType* __restrict__ new_col_idxs,
                      ValueType* __restrict__ new_vals, bool lower,
                      sycl::nd_item<3> item_ct1)
{
    abstract_filter<subwarp_size>(
        old_row_ptrs, old_col_idxs, old_vals, num_rows,
        [&](IndexType idx, IndexType row_begin, IndexType row_end) {
            auto diag_idx = lower ? row_end - 1 : row_begin;
            return abs(old_vals[idx]) >= threshold || idx == diag_idx;
        },
        new_row_ptrs, new_row_idxs, new_col_idxs, new_vals, item_ct1);
}

template <int subwarp_size, typename ValueType, typename IndexType>
void threshold_filter(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                      sycl::queue* queue, const IndexType* old_row_ptrs,
                      const IndexType* old_col_idxs, const ValueType* old_vals,
                      IndexType num_rows, remove_complex<ValueType> threshold,
                      const IndexType* new_row_ptrs, IndexType* new_row_idxs,
                      IndexType* new_col_idxs, ValueType* new_vals, bool lower)
{
    queue->parallel_for(
        sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
            threshold_filter<subwarp_size>(old_row_ptrs, old_col_idxs, old_vals,
                                           num_rows, threshold, new_row_ptrs,
                                           new_row_idxs, new_col_idxs, new_vals,
                                           lower, item_ct1);
        });
}


template <int subwarp_size, typename IndexType, typename BucketType>
void bucket_filter_nnz(const IndexType* __restrict__ row_ptrs,
                       const BucketType* buckets, IndexType num_rows,
                       BucketType bucket, IndexType* __restrict__ nnz,
                       sycl::nd_item<3> item_ct1)
{
    abstract_filter_nnz<subwarp_size>(
        row_ptrs, num_rows,
        [&](IndexType idx, IndexType row_begin, IndexType row_end) {
            return buckets[idx] >= bucket || idx == row_end - 1;
        },
        nnz, item_ct1);
}

template <int subwarp_size, typename IndexType, typename BucketType>
void bucket_filter_nnz(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                       sycl::queue* queue, const IndexType* row_ptrs,
                       const BucketType* buckets, IndexType num_rows,
                       BucketType bucket, IndexType* nnz)
{
    queue->parallel_for(
        sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
            bucket_filter_nnz<subwarp_size>(row_ptrs, buckets, num_rows, bucket,
                                            nnz, item_ct1);
        });
}


template <int subwarp_size, typename ValueType, typename IndexType,
          typename BucketType>
void bucket_filter(const IndexType* __restrict__ old_row_ptrs,
                   const IndexType* __restrict__ old_col_idxs,
                   const ValueType* __restrict__ old_vals,
                   const BucketType* buckets, IndexType num_rows,
                   BucketType bucket,
                   const IndexType* __restrict__ new_row_ptrs,
                   IndexType* __restrict__ new_row_idxs,
                   IndexType* __restrict__ new_col_idxs,
                   ValueType* __restrict__ new_vals, sycl::nd_item<3> item_ct1)
{
    abstract_filter<subwarp_size>(
        old_row_ptrs, old_col_idxs, old_vals, num_rows,
        [&](IndexType idx, IndexType row_begin, IndexType row_end) {
            return buckets[idx] >= bucket || idx == row_end - 1;
        },
        new_row_ptrs, new_row_idxs, new_col_idxs, new_vals, item_ct1);
}

template <int subwarp_size, typename ValueType, typename IndexType,
          typename BucketType>
void bucket_filter(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                   sycl::queue* queue, const IndexType* old_row_ptrs,
                   const IndexType* old_col_idxs, const ValueType* old_vals,
                   const BucketType* buckets, IndexType num_rows,
                   BucketType bucket, const IndexType* new_row_ptrs,
                   IndexType* new_row_idxs, IndexType* new_col_idxs,
                   ValueType* new_vals)
{
    queue->parallel_for(
        sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
            bucket_filter<subwarp_size>(
                old_row_ptrs, old_col_idxs, old_vals, buckets, num_rows, bucket,
                new_row_ptrs, new_row_idxs, new_col_idxs, new_vals, item_ct1);
        });
}


}  // namespace kernel
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
    // auto stride = remove_complex<ValueType>(size) / sample_size;
#pragma unroll
    for (int i = 0; i < sampleselect_oversampling; ++i) {
        auto lidx = idx * sampleselect_oversampling + i;
        auto val = input[static_cast<IndexType>(lidx * size / sample_size)];
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

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                build_searchtree(input, size, tree_output, item_ct1,
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
        auto el = abs(input[i]);
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
            warp_sums_acc_ct1(sycl::range<1>(16 /*num_warps*/), cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                block_prefix_sum(counters, totals, num_blocks, item_ct1,
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
            output[ofs] = abs(input[i]);
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
    constexpr auto sentinel = device_numeric_limits<ValueType>::inf;
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
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
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
                        [=](sycl::nd_item<3> item_ct1) {
                            find_bucket(prefix_sum, rank, item_ct1);
                        });
}


}  // namespace kernel


template <int subwarp_size, typename ValueType, typename IndexType>
void threshold_filter_approx(syn::value_list<int, subwarp_size>,
                             std::shared_ptr<const DefaultExecutor> exec,
                             const matrix::Csr<ValueType, IndexType>* m,
                             IndexType rank, Array<ValueType>* tmp,
                             remove_complex<ValueType>* threshold,
                             matrix::Csr<ValueType, IndexType>* m_out,
                             matrix::Coo<ValueType, IndexType>* m_out_coo)
{
    auto values = m->get_const_values();
    IndexType size = m->get_num_stored_elements();
    using AbsType = remove_complex<ValueType>;
    constexpr auto bucket_count = kernel::searchtree_width;
    auto max_num_threads = ceildiv(size, items_per_thread);
    auto max_num_blocks = ceildiv(max_num_threads, default_block_size);

    size_type tmp_size_totals =
        ceildiv((bucket_count + 1) * sizeof(IndexType), sizeof(ValueType));
    size_type tmp_size_partials = ceildiv(
        bucket_count * max_num_blocks * sizeof(IndexType), sizeof(ValueType));
    size_type tmp_size_oracles =
        ceildiv(size * sizeof(unsigned char), sizeof(ValueType));
    size_type tmp_size_tree =
        ceildiv(kernel::searchtree_size * sizeof(AbsType), sizeof(ValueType));
    size_type tmp_size =
        tmp_size_totals + tmp_size_partials + tmp_size_oracles + tmp_size_tree;
    tmp->resize_and_reset(tmp_size);

    auto total_counts = reinterpret_cast<IndexType*>(tmp->get_data());
    auto partial_counts =
        reinterpret_cast<IndexType*>(tmp->get_data() + tmp_size_totals);
    auto oracles = reinterpret_cast<unsigned char*>(
        tmp->get_data() + tmp_size_totals + tmp_size_partials);
    auto tree =
        reinterpret_cast<AbsType*>(tmp->get_data() + tmp_size_totals +
                                   tmp_size_partials + tmp_size_oracles);

    sampleselect_count(exec, values, size, tree, oracles, partial_counts,
                       total_counts);

    // determine bucket with correct rank
    auto bucket = static_cast<unsigned char>(
        sampleselect_find_bucket(exec, total_counts, rank).idx);
    *threshold =
        exec->copy_val_to_host(tree + kernel::searchtree_inner_size + bucket);
    // we implicitly set the first splitter to -inf, but 0 works as well
    if (bucket == 0) {
        *threshold = zero<AbsType>();
    }

    // filter the elements
    auto old_row_ptrs = m->get_const_row_ptrs();
    auto old_col_idxs = m->get_const_col_idxs();
    auto old_vals = m->get_const_values();
    // compute nnz for each row
    auto num_rows = static_cast<IndexType>(m->get_size()[0]);
    auto block_size = default_block_size / subwarp_size;
    auto num_blocks = ceildiv(num_rows, block_size);
    auto new_row_ptrs = m_out->get_row_ptrs();
    kernel::bucket_filter_nnz<subwarp_size>(
        num_blocks, default_block_size, 0, exec->get_queue(), old_row_ptrs,
        oracles, num_rows, bucket, new_row_ptrs);

    // build row pointers
    components::prefix_sum(exec, new_row_ptrs, num_rows + 1);

    // build matrix
    auto new_nnz = exec->copy_val_to_host(new_row_ptrs + num_rows);
    // resize arrays and update aliases
    matrix::CsrBuilder<ValueType, IndexType> builder{m_out};
    builder.get_col_idx_array().resize_and_reset(new_nnz);
    builder.get_value_array().resize_and_reset(new_nnz);
    auto new_col_idxs = m_out->get_col_idxs();
    auto new_vals = m_out->get_values();
    IndexType* new_row_idxs{};
    if (m_out_coo) {
        matrix::CooBuilder<ValueType, IndexType> coo_builder{m_out_coo};
        coo_builder.get_row_idx_array().resize_and_reset(new_nnz);
        coo_builder.get_col_idx_array() =
            Array<IndexType>::view(exec, new_nnz, new_col_idxs);
        coo_builder.get_value_array() =
            Array<ValueType>::view(exec, new_nnz, new_vals);
        new_row_idxs = m_out_coo->get_row_idxs();
    }
    kernel::bucket_filter<subwarp_size>(
        num_blocks, default_block_size, 0, exec->get_queue(), old_row_ptrs,
        old_col_idxs, old_vals, oracles, num_rows, bucket, new_row_ptrs,
        new_row_idxs, new_col_idxs, new_vals);
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_threshold_filter_approx,
                                    threshold_filter_approx);


template <typename ValueType, typename IndexType>
void threshold_filter_approx(std::shared_ptr<const DefaultExecutor> exec,
                             const matrix::Csr<ValueType, IndexType>* m,
                             IndexType rank, Array<ValueType>& tmp,
                             remove_complex<ValueType>& threshold,
                             matrix::Csr<ValueType, IndexType>* m_out,
                             matrix::Coo<ValueType, IndexType>* m_out_coo)
{
    auto num_rows = m->get_size()[0];
    auto total_nnz = m->get_num_stored_elements();
    auto total_nnz_per_row = total_nnz / num_rows;
    select_threshold_filter_approx(
        compiled_kernels(),
        [&](int compiled_subwarp_size) {
            return total_nnz_per_row <= compiled_subwarp_size ||
                   compiled_subwarp_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, m, rank, &tmp,
        &threshold, m_out, m_out_coo);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_APPROX_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
