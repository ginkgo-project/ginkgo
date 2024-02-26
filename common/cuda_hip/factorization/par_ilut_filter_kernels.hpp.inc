// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

namespace kernel {


template <int subwarp_size, typename IndexType, typename Predicate,
          typename BeginCallback, typename StepCallback,
          typename FinishCallback>
__device__ void abstract_filter_impl(const IndexType* row_ptrs,
                                     IndexType num_rows, Predicate pred,
                                     BeginCallback begin_cb,
                                     StepCallback step_cb,
                                     FinishCallback finish_cb)
{
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
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
__device__ void abstract_filter_nnz(const IndexType* __restrict__ row_ptrs,
                                    IndexType num_rows, Predicate pred,
                                    IndexType* __restrict__ nnz)
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
        });
}


template <int subwarp_size, typename Predicate, typename IndexType,
          typename ValueType>
__device__ void abstract_filter(const IndexType* __restrict__ old_row_ptrs,
                                const IndexType* __restrict__ old_col_idxs,
                                const ValueType* __restrict__ old_vals,
                                IndexType num_rows, Predicate pred,
                                const IndexType* __restrict__ new_row_ptrs,
                                IndexType* __restrict__ new_row_idxs,
                                IndexType* __restrict__ new_col_idxs,
                                ValueType* __restrict__ new_vals)
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
        [](IndexType, IndexType) {});
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void threshold_filter_nnz(
    const IndexType* __restrict__ row_ptrs, const ValueType* vals,
    IndexType num_rows, remove_complex<ValueType> threshold,
    IndexType* __restrict__ nnz, bool lower)
{
    abstract_filter_nnz<subwarp_size>(
        row_ptrs, num_rows,
        [&](IndexType idx, IndexType row_begin, IndexType row_end) {
            auto diag_idx = lower ? row_end - 1 : row_begin;
            return abs(vals[idx]) >= threshold || idx == diag_idx;
        },
        nnz);
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void threshold_filter(
    const IndexType* __restrict__ old_row_ptrs,
    const IndexType* __restrict__ old_col_idxs,
    const ValueType* __restrict__ old_vals, IndexType num_rows,
    remove_complex<ValueType> threshold,
    const IndexType* __restrict__ new_row_ptrs,
    IndexType* __restrict__ new_row_idxs, IndexType* __restrict__ new_col_idxs,
    ValueType* __restrict__ new_vals, bool lower)
{
    abstract_filter<subwarp_size>(
        old_row_ptrs, old_col_idxs, old_vals, num_rows,
        [&](IndexType idx, IndexType row_begin, IndexType row_end) {
            auto diag_idx = lower ? row_end - 1 : row_begin;
            return abs(old_vals[idx]) >= threshold || idx == diag_idx;
        },
        new_row_ptrs, new_row_idxs, new_col_idxs, new_vals);
}


template <int subwarp_size, typename IndexType, typename BucketType>
__global__ __launch_bounds__(default_block_size) void bucket_filter_nnz(
    const IndexType* __restrict__ row_ptrs, const BucketType* buckets,
    IndexType num_rows, BucketType bucket, IndexType* __restrict__ nnz)
{
    abstract_filter_nnz<subwarp_size>(
        row_ptrs, num_rows,
        [&](IndexType idx, IndexType row_begin, IndexType row_end) {
            return buckets[idx] >= bucket || idx == row_end - 1;
        },
        nnz);
}


template <int subwarp_size, typename ValueType, typename IndexType,
          typename BucketType>
__global__ __launch_bounds__(default_block_size) void bucket_filter(
    const IndexType* __restrict__ old_row_ptrs,
    const IndexType* __restrict__ old_col_idxs,
    const ValueType* __restrict__ old_vals, const BucketType* buckets,
    IndexType num_rows, BucketType bucket,
    const IndexType* __restrict__ new_row_ptrs,
    IndexType* __restrict__ new_row_idxs, IndexType* __restrict__ new_col_idxs,
    ValueType* __restrict__ new_vals)
{
    abstract_filter<subwarp_size>(
        old_row_ptrs, old_col_idxs, old_vals, num_rows,
        [&](IndexType idx, IndexType row_begin, IndexType row_end) {
            return buckets[idx] >= bucket || idx == row_end - 1;
        },
        new_row_ptrs, new_row_idxs, new_col_idxs, new_vals);
}


}  // namespace kernel
