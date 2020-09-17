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

constexpr auto spgemm_block_size = 64;
constexpr auto spgemm_child_count = 4;


template <typename ValueType, typename IndexType>
struct memory_row_accessor {
    __device__ memory_row_accessor(const ValueType *val, const IndexType *idx,
                                   IndexType base_idx, IndexType size)
        : val_{val + base_idx}, idx_{idx + base_idx}, size_{size}
    {}

    __device__ __forceinline__ IndexType col(IndexType i) const
    {
        return idx_[i];
    }

    __device__ __forceinline__ ValueType val(IndexType i) const
    {
        return val_[i];
    }

    const ValueType *val_;
    const IndexType *idx_;
    IndexType size_;
};


template <typename ValueType, typename IndexType>
struct merge_row_accessor {
    __device__ merge_row_accessor(IndexType base_idx, IndexType size)
        : base_idx_{base_idx}, size_{size}
    {}

    __device__ __forceinline__ IndexType col(IndexType i) const
    {
        return base_idx_ + i;
    }

    __device__ __forceinline__ ValueType val(IndexType i) const
    {
        return one<ValueType>();
    }

    IndexType base_idx_;
    IndexType size_;
};


template <typename ValueType, typename IndexType, typename AAccessor>
__device__ void copy_row(AAccessor a, const IndexType *__restrict__ b_row_ptrs,
                         const IndexType *__restrict__ b_cols,
                         const ValueType *__restrict__ b_vals,
                         IndexType c_begin, IndexType *__restrict__ c_cols,
                         ValueType *__restrict__ c_vals)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    auto a_col = a.col(0);
    auto a_val = a.val(0);
    auto b_begin = b_row_ptrs[a_col];
    auto b_size = b_row_ptrs[a_col + 1] - b_begin;
    for (IndexType i = warp.thread_rank(); i < b_size; i += config::warp_size) {
        c_cols[c_begin + i] = b_cols[b_begin + i];
        c_vals[c_begin + i] = a_val * b_vals[b_begin + i];
    }
}


template <typename IndexType, typename AAccessor>
__device__ IndexType count_merge_2way(AAccessor a,
                                      const IndexType *__restrict__ b_row_ptrs,
                                      const IndexType *__restrict__ b_cols)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto a_col1 = a.col(0);
    const auto a_col2 = a.col(1);
    const auto b_begin1 = b_row_ptrs[a_col1];
    const auto b_begin2 = b_row_ptrs[a_col2];
    const auto b_size1 = b_row_ptrs[a_col1 + 1] - b_begin1;
    const auto b_size2 = b_row_ptrs[a_col2 + 1] - b_begin2;
    IndexType nnz{};
    group_merge<config::warp_size>(
        b_cols + b_begin1, b_size1, b_cols + b_begin2, b_size2, warp,
        [&](IndexType, IndexType b_col1, IndexType, IndexType b_col2, IndexType,
            bool valid) {
            nnz += popcnt(warp.ballot(b_col1 != b_col2 && valid));
            return true;
        });
    return nnz;
}


template <typename ValueType, typename IndexType, typename AAccessor>
__device__ void merge_2way(AAccessor a,
                           const IndexType *__restrict__ b_row_ptrs,
                           const IndexType *__restrict__ b_cols,
                           const ValueType *__restrict__ b_vals, IndexType c_nz,
                           IndexType *__restrict__ c_cols,
                           ValueType *__restrict__ c_vals)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto a_col1 = a.col(0);
    const auto a_col2 = a.col(1);
    const auto a_val1 = a.val(0);
    const auto a_val2 = a.val(1);
    const auto b_begin1 = b_row_ptrs[a_col1];
    const auto b_begin2 = b_row_ptrs[a_col2];
    const auto b_size1 = b_row_ptrs[a_col1 + 1] - b_begin1;
    const auto b_size2 = b_row_ptrs[a_col2 + 1] - b_begin2;
    bool skip_first = false;
    const auto lane = static_cast<IndexType>(warp.thread_rank());
    const auto lanemask_eq = config::lane_mask_type{1} << lane;
    const auto lanemask_lt = lanemask_eq - 1;
    group_merge<config::warp_size>(
        b_cols + b_begin1, b_size1, b_cols + b_begin2, b_size2, warp,
        [&](IndexType b_nz1, IndexType b_col1, IndexType b_nz2,
            IndexType b_col2, IndexType, bool valid) {
            auto c_col = min(b_col1, b_col2);
            auto valid_mask = warp.ballot(valid);
            auto equal_mask = warp.ballot(b_col1 == b_col2) & valid_mask;
            // check if the elements in the previous merge step are
            // equal
            auto prev_equal_mask = equal_mask << 1 | skip_first;
            // store the highest bit for the next group_merge_step
            skip_first = bool(equal_mask >> (config::warp_size - 1));
            auto prev_equal = bool(prev_equal_mask & lanemask_eq);
            // only output an entry if the previous cols weren't equal.
            // if they were equal, they were both handled in the
            // previous step
            if (valid && !prev_equal) {
                auto c_ofs = popcnt(~prev_equal_mask & lanemask_lt);
                c_cols[c_nz + c_ofs] = c_col;
                auto b_val1 = b_col1 <= b_col2 ? b_vals[b_nz1 + b_begin1]
                                               : zero<ValueType>();
                auto b_val2 = b_col2 <= b_col1 ? b_vals[b_nz2 + b_begin2]
                                               : zero<ValueType>();
                c_vals[c_nz + c_ofs] = a_val1 * b_val1 + a_val2 * b_val2;
            }
            // advance by the number of merged elements
            c_nz += popcnt(~prev_equal_mask & valid_mask);
            return true;
        });
}


template <typename IndexType, typename AAccessor>
__device__ IndexType count_merge_nway(AAccessor a, int a_size,
                                      const IndexType *__restrict__ b_row_ptrs,
                                      const IndexType *__restrict__ b_cols)
{
    constexpr auto sentinel = device_numeric_limits<IndexType>::max;
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto lane = static_cast<IndexType>(warp.thread_rank());
    IndexType a_col{};
    IndexType b_begin{};
    IndexType b_end{};
    if (lane < a_size) {
        a_col = a.col(lane);
        b_begin = b_row_ptrs[a_col];
        b_end = b_row_ptrs[a_col + 1];
    }
    // the last column from the previous iteration
    IndexType last_col{-1};

    IndexType c_nnz{};
    while (warp.any(b_begin < b_end)) {
        config::lane_mask_type advance_mask;
        auto b_cur_begin = warp.shfl(b_begin, 0);
        auto b_cur_end = warp.shfl(b_end, 0);
        auto col =
            checked_load(b_cols, b_cur_begin + lane, b_cur_end, sentinel);
        for (int i = 1; i < a_size; ++i) {
            auto new_col = checked_load(b_cols, warp.shfl(b_begin, i) + lane,
                                        warp.shfl(b_end, i), sentinel);
            auto merge_result =
                group_merge_step<config::warp_size>(new_col, col, warp);
            if (lane == i) {
                advance_mask = merge_result.a_less_b_mask;
            }
            col = min(merge_result.a_val, merge_result.b_val);
        }
        auto prev_col = warp.shfl_up(col, 1);
        if (lane == 0) {
            prev_col = last_col;
        }
        last_col = warp.shfl(col, config::warp_size - 1);
        c_nnz += popcnt(warp.ballot(col != prev_col && col != sentinel));
        // advance indices by the number of consumed elements
        auto rest_advance = config::warp_size;
        for (int i = a_size - 1; i > 0; --i) {
            auto local_advance =
                popcnt(warp.shfl(advance_mask, i) & prefix_mask(rest_advance));
            rest_advance -= local_advance;
            if (lane == i) {
                b_begin += local_advance;
            }
        }
        if (lane == 0) {
            b_begin += rest_advance;
        }
    }
    return c_nnz;
}


template <typename ValueType, typename IndexType, typename AAccessor>
__device__ void merge_nway(AAccessor a, int a_size,
                           const IndexType *__restrict__ b_row_ptrs,
                           const IndexType *__restrict__ b_cols,
                           const ValueType *__restrict__ b_vals, IndexType c_nz,
                           IndexType *__restrict__ c_cols,
                           ValueType *__restrict__ c_vals)
{
    constexpr auto sentinel = device_numeric_limits<IndexType>::max;
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto lane = static_cast<IndexType>(warp.thread_rank());
    IndexType a_col{};
    ValueType a_val{};
    IndexType b_begin{};
    IndexType b_end{};
    if (lane < a_size) {
        a_col = a.col(lane);
        a_val = a.val(lane);
        b_begin = b_row_ptrs[a_col];
        b_end = b_row_ptrs[a_col + 1];
    }
    // the last column from the previous iteration
    IndexType last_col{-1};

    while (warp.any(b_begin < b_end)) {
        config::lane_mask_type advance_mask{};
        auto b_cur_begin = warp.shfl(b_begin, 0);
        auto b_cur_end = warp.shfl(b_end, 0);
        auto a_cur_val = warp.shfl(a_val, 0);
        auto col =
            checked_load(b_cols, b_cur_begin + lane, b_cur_end, sentinel);
        auto val = a_cur_val * checked_load(b_vals, b_cur_begin + lane,
                                            b_cur_end, zero<ValueType>());
        for (int i = 1; i < a_size; ++i) {
            b_cur_begin = warp.shfl(b_begin, i);
            b_cur_end = warp.shfl(b_end, i);
            a_cur_val = warp.shfl(a_val, i);
            auto new_col =
                checked_load(b_cols, b_cur_begin + lane, b_cur_end, sentinel);
            auto new_val =
                a_cur_val * checked_load(b_vals, b_cur_begin + lane, b_cur_end,
                                         zero<ValueType>());
            auto merge_result =
                group_merge_step<config::warp_size>(new_col, col, warp);
            if (lane == i) {
                advance_mask = merge_result.a_less_b_mask;
            }
            col = min(merge_result.a_val, merge_result.b_val);
            new_val = warp.shfl(new_val, merge_result.a_idx);
            val = warp.shfl(val, merge_result.b_idx);
            auto leq_scale = merge_result.a_val <= merge_result.b_val
                                 ? one<ValueType>()
                                 : zero<ValueType>();
            auto geq_scale = merge_result.a_val >= merge_result.b_val
                                 ? one<ValueType>()
                                 : zero<ValueType>();
            val = geq_scale * val + leq_scale * new_val;
        }
        auto prev_col = warp.shfl_up(col, 1);
        if (lane == 0) {
            prev_col = last_col;
            // take care of the case when a column value was only partially
            // computed within an iteration.
            // this can only happen once to each col since we merge <= 32 rows
            if (prev_col == col) {
                // why is this not necessary?
                // c_vals[c_nz - 1] += val;
            }
        }
        last_col = warp.shfl(col, config::warp_size - 1);
        auto output = col != prev_col && col != sentinel;
        auto output_mask = warp.ballot(output);
        if (output) {
            auto c_idx = c_nz + popcnt(output_mask & prefix_mask(lane));
            c_cols[c_idx] = col;
            c_vals[c_idx] = val;
        }
        c_nz += popcnt(warp.ballot(col != prev_col && col != sentinel));
        // advance indices by the number of consumed elements
        auto rest_advance = config::warp_size;
        for (int i = a_size - 1; i > 0; --i) {
            auto local_advance =
                popcnt(warp.shfl(advance_mask, i) & prefix_mask(rest_advance));
            rest_advance -= local_advance;
            if (lane == i) {
                b_begin += local_advance;
            }
        }
        if (lane == 0) {
            b_begin += rest_advance;
        }
    }
}


template <typename IndexType>
__global__ __launch_bounds__(spgemm_block_size) void spgemm_count_short(
    size_type num_rows, const IndexType *__restrict__ a_row_ptrs,
    const IndexType *__restrict__ a_cols,
    const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, IndexType *__restrict__ c_nnz)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    if (row >= num_rows) {
        return;
    }

    const auto write = threadIdx.x % config::warp_size == 0;
    const auto a_begin = a_row_ptrs[row];
    const auto a_end = a_row_ptrs[row + 1];
    const auto a_size = a_end - a_begin;
    auto null_val = static_cast<float *>(nullptr);
    memory_row_accessor<float, IndexType> a{null_val, a_cols, a_begin, a_size};
    IndexType count{};
    if (a_size == 0) {
        // do nothing
    } else if (a_size == 1) {
        auto a_col = a.col(0);
        count = b_row_ptrs[a_col + 1] - b_row_ptrs[a_col];
    } else if (a_size == 2) {
        count = count_merge_2way(a, b_row_ptrs, b_cols);
    } else {
        count = count_merge_nway(a, a_size, b_row_ptrs, b_cols);
    }
    if (write) {
        c_nnz[row] = count;
    }
}


template <typename IndexType>
__global__ __launch_bounds__(spgemm_block_size) void spgemm_count_merge(
    size_type num_rows, const IndexType *__restrict__ a_row_ptrs,
    const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, IndexType *__restrict__ c_nnz)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    if (row >= num_rows) {
        return;
    }

    const auto write = threadIdx.x % config::warp_size == 0;
    const auto a_begin = a_row_ptrs[row];
    const auto a_end = a_row_ptrs[row + 1];
    const auto a_size = a_end - a_begin;
    auto null_val = static_cast<float *>(nullptr);
    merge_row_accessor<float, IndexType> a{a_begin, a_size};
    IndexType count{};
    if (a_size == 0) {
        // do nothing
    } else if (a_size == 1) {
        auto a_col = a.col(0);
        count = b_row_ptrs[a_col + 1] - b_row_ptrs[a_col];
    } else if (a_size == 2) {
        count = count_merge_2way(a, b_row_ptrs, b_cols);
    } else {
        count = count_merge_nway(a, a_size, b_row_ptrs, b_cols);
    }
    if (write) {
        c_nnz[row] = count;
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spgemm_block_size) void spgemm_kernel_short(
    size_type num_rows, const IndexType *__restrict__ a_row_ptrs,
    const IndexType *__restrict__ a_cols, const ValueType *__restrict__ a_vals,
    const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, const ValueType *__restrict__ b_vals,
    const IndexType *__restrict__ c_row_ptrs, IndexType *__restrict__ c_cols,
    ValueType *__restrict__ c_vals)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    if (row >= num_rows) {
        return;
    }

    const auto write = threadIdx.x % config::warp_size == 0;
    const auto a_begin = a_row_ptrs[row];
    const auto a_end = a_row_ptrs[row + 1];
    const auto a_size = a_end - a_begin;
    memory_row_accessor<ValueType, IndexType> a{a_vals, a_cols, a_begin,
                                                a_size};

    IndexType c_nz = c_row_ptrs[row];
    if (a_size == 0) {
        // do nothing
    } else if (a_size == 1) {
        copy_row(a, b_row_ptrs, b_cols, b_vals, c_nz, c_cols, c_vals);
    } else if (a_size == 2) {
        merge_2way(a, b_row_ptrs, b_cols, b_vals, c_nz, c_cols, c_vals);
    } else {
        merge_nway(a, a_size, b_row_ptrs, b_cols, b_vals, c_nz, c_cols, c_vals);
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spgemm_block_size) void spgemm_kernel_merge(
    size_type num_rows, const IndexType *__restrict__ a_row_ptrs,
    const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, const ValueType *__restrict__ b_vals,
    const IndexType *__restrict__ c_row_ptrs, IndexType *__restrict__ c_cols,
    ValueType *__restrict__ c_vals)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    if (row >= num_rows) {
        return;
    }

    const auto write = threadIdx.x % config::warp_size == 0;
    const auto a_begin = a_row_ptrs[row];
    const auto a_end = a_row_ptrs[row + 1];
    const auto a_size = a_end - a_begin;
    merge_row_accessor<ValueType, IndexType> a{a_begin, a_size};

    IndexType c_nz = c_row_ptrs[row];
    if (a_size == 0) {
        // do nothing
    } else if (a_size == 1) {
        copy_row(a, b_row_ptrs, b_cols, b_vals, c_nz, c_cols, c_vals);
    } else if (a_size == 2) {
        merge_2way(a, b_row_ptrs, b_cols, b_vals, c_nz, c_cols, c_vals);
    } else {
        merge_nway(a, a_size, b_row_ptrs, b_cols, b_vals, c_nz, c_cols, c_vals);
    }
}


template <typename IndexType>
__global__ __launch_bounds__(spgemm_block_size) void spgemm_merge_counts(
    const IndexType *__restrict__ a_row_ptrs, size_type num_rows,
    int merge_size, IndexType *__restrict__ merge_count)
{
    const auto row = thread::get_thread_id_flat();
    if (row >= num_rows) {
        return;
    }

    const auto size = a_row_ptrs[row + 1] - a_row_ptrs[row];
    merge_count[row] = ceildiv(size, merge_size);
}


template <typename IndexType>
__global__ __launch_bounds__(spgemm_block_size) void spgemm_tall_row_ptrs(
    const IndexType *__restrict__ a_row_ptrs,
    const IndexType *__restrict__ row_size_prefixsum, size_type num_rows,
    int merge_size, IndexType *__restrict__ tall_row_ptrs)
{
    const auto row = thread::get_thread_id_flat();
    if (row >= num_rows) {
        return;
    }

    const auto begin = a_row_ptrs[row];
    const auto end = a_row_ptrs[row + 1];
    auto out_idx = row_size_prefixsum[row];

    for (auto i = begin; i < end; i += merge_size) {
        tall_row_ptrs[out_idx] = i;
        ++out_idx;
    }
    if (row == num_rows - 1) {
        tall_row_ptrs[out_idx] = end;
    }
}
