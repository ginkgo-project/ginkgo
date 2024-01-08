// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

namespace kernel {


template <int subgroup_size, typename IndexType, typename Predicate,
          typename BeginCallback, typename StepCallback,
          typename FinishCallback>
void abstract_filter_impl(const IndexType* row_ptrs, IndexType num_rows,
                          Predicate pred, BeginCallback begin_cb,
                          StepCallback step_cb, FinishCallback finish_cb,
                          sycl::nd_item<3> item_ct1)
{
    auto subwarp = group::tiled_partition<subgroup_size>(
        group::this_thread_block(item_ct1));
    auto row = thread::get_subwarp_id_flat<subgroup_size, IndexType>(item_ct1);
    auto lane = subwarp.thread_rank();
    auto lane_prefix_mask = (config::lane_mask_type(1) << lane) - 1;
    if (row >= num_rows) {
        return;
    }

    auto begin = row_ptrs[row];
    auto end = row_ptrs[row + 1];
    begin_cb(row);
    auto num_steps = ceildiv(end - begin, subgroup_size);
    for (IndexType step = 0; step < num_steps; ++step) {
        auto idx = begin + lane + step * subgroup_size;
        auto keep = idx < end && pred(idx, begin, end);
        auto mask = subwarp.ballot(keep);
        step_cb(row, idx, keep, popcnt(mask), popcnt(mask & lane_prefix_mask));
    }
    finish_cb(row, lane);
}


template <int subgroup_size, typename Predicate, typename IndexType>
void abstract_filter_nnz(const IndexType* __restrict__ row_ptrs,
                         IndexType num_rows, Predicate pred,
                         IndexType* __restrict__ nnz, sycl::nd_item<3> item_ct1)
{
    IndexType count{};
    abstract_filter_impl<subgroup_size>(
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


template <int subgroup_size, typename Predicate, typename IndexType,
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
    abstract_filter_impl<subgroup_size>(
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


template <int subgroup_size, typename ValueType, typename IndexType>
void threshold_filter_nnz(const IndexType* __restrict__ row_ptrs,
                          const ValueType* vals, IndexType num_rows,
                          remove_complex<ValueType> threshold,
                          IndexType* __restrict__ nnz, bool lower,
                          sycl::nd_item<3> item_ct1)
{
    abstract_filter_nnz<subgroup_size>(
        row_ptrs, num_rows,
        [&](IndexType idx, IndexType row_begin, IndexType row_end) {
            auto diag_idx = lower ? row_end - 1 : row_begin;
            return std::abs(vals[idx]) >= threshold || idx == diag_idx;
        },
        nnz, item_ct1);
}

template <int subgroup_size, typename ValueType, typename IndexType>
void threshold_filter_nnz(dim3 grid, dim3 block,
                          size_type dynamic_shared_memory, sycl::queue* queue,
                          const IndexType* row_ptrs, const ValueType* vals,
                          IndexType num_rows,
                          remove_complex<ValueType> threshold, IndexType* nnz,
                          bool lower)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(subgroup_size)]] {
                                threshold_filter_nnz<subgroup_size>(
                                    row_ptrs, vals, num_rows, threshold, nnz,
                                    lower, item_ct1);
                            });
}


template <int subgroup_size, typename ValueType, typename IndexType>
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
    abstract_filter<subgroup_size>(
        old_row_ptrs, old_col_idxs, old_vals, num_rows,
        [&](IndexType idx, IndexType row_begin, IndexType row_end) {
            auto diag_idx = lower ? row_end - 1 : row_begin;
            return std::abs(old_vals[idx]) >= threshold || idx == diag_idx;
        },
        new_row_ptrs, new_row_idxs, new_col_idxs, new_vals, item_ct1);
}

template <int subgroup_size, typename ValueType, typename IndexType>
void threshold_filter(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                      sycl::queue* queue, const IndexType* old_row_ptrs,
                      const IndexType* old_col_idxs, const ValueType* old_vals,
                      IndexType num_rows, remove_complex<ValueType> threshold,
                      const IndexType* new_row_ptrs, IndexType* new_row_idxs,
                      IndexType* new_col_idxs, ValueType* new_vals, bool lower)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(subgroup_size)]] {
                                threshold_filter<subgroup_size>(
                                    old_row_ptrs, old_col_idxs, old_vals,
                                    num_rows, threshold, new_row_ptrs,
                                    new_row_idxs, new_col_idxs, new_vals, lower,
                                    item_ct1);
                            });
}


template <int subgroup_size, typename IndexType, typename BucketType>
void bucket_filter_nnz(const IndexType* __restrict__ row_ptrs,
                       const BucketType* buckets, IndexType num_rows,
                       BucketType bucket, IndexType* __restrict__ nnz,
                       sycl::nd_item<3> item_ct1)
{
    abstract_filter_nnz<subgroup_size>(
        row_ptrs, num_rows,
        [&](IndexType idx, IndexType row_begin, IndexType row_end) {
            return buckets[idx] >= bucket || idx == row_end - 1;
        },
        nnz, item_ct1);
}

template <int subgroup_size, typename IndexType, typename BucketType>
void bucket_filter_nnz(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                       sycl::queue* queue, const IndexType* row_ptrs,
                       const BucketType* buckets, IndexType num_rows,
                       BucketType bucket, IndexType* nnz)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(subgroup_size)]] {
                                bucket_filter_nnz<subgroup_size>(
                                    row_ptrs, buckets, num_rows, bucket, nnz,
                                    item_ct1);
                            });
}


template <int subgroup_size, typename ValueType, typename IndexType,
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
    abstract_filter<subgroup_size>(
        old_row_ptrs, old_col_idxs, old_vals, num_rows,
        [&](IndexType idx, IndexType row_begin, IndexType row_end) {
            return buckets[idx] >= bucket || idx == row_end - 1;
        },
        new_row_ptrs, new_row_idxs, new_col_idxs, new_vals, item_ct1);
}

template <int subgroup_size, typename ValueType, typename IndexType,
          typename BucketType>
void bucket_filter(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                   sycl::queue* queue, const IndexType* old_row_ptrs,
                   const IndexType* old_col_idxs, const ValueType* old_vals,
                   const BucketType* buckets, IndexType num_rows,
                   BucketType bucket, const IndexType* new_row_ptrs,
                   IndexType* new_row_idxs, IndexType* new_col_idxs,
                   ValueType* new_vals)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(subgroup_size)]] {
                                bucket_filter<subgroup_size>(
                                    old_row_ptrs, old_col_idxs, old_vals,
                                    buckets, num_rows, bucket, new_row_ptrs,
                                    new_row_idxs, new_col_idxs, new_vals,
                                    item_ct1);
                            });
}


}  // namespace kernel
