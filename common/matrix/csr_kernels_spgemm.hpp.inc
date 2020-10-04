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


// find the position of the rank'th 1 bit in mask (0-based)
__forceinline__ __device__ int bitmask_select(config::lane_mask_type mask,
                                              int rank)
{
    return synchronous_fixed_binary_search<config::warp_size>(
        [&](int i) { return popcnt(mask & prefix_mask(i + 1)) >= rank + 1; });
}


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
__device__ IndexType count_dense(AAccessor a, IndexType a_size,
                                 IndexType c_first_col,
                                 const IndexType *__restrict__ b_row_ptrs,
                                 const IndexType *__restrict__ b_cols)
{
    constexpr auto sentinel = device_numeric_limits<IndexType>::max;
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto lane = static_cast<IndexType>(warp.thread_rank());

    constexpr auto num_blocks = 32;
    constexpr auto accumulator_size = 32 * num_blocks;
    constexpr auto num_warps = spgemm_block_size / config::warp_size;
    const auto warp_idx = threadIdx.x / config::warp_size;
    __shared__ uint32 sh_sparsity[num_blocks * num_warps];

    auto local_sparsity = sh_sparsity + (warp_idx * num_blocks);
    // vector: ith thread stores begin/end of ith row
    IndexType b_begin{};
    IndexType b_end{};
    // vector: beginning of next dense block column in ith row
    auto b_next_col = sentinel;
    // scalar: non-zero counter for row in C
    IndexType c_nnz{};
    if (lane < a_size) {
        auto a_col = a.col(lane);
        b_begin = b_row_ptrs[a_col];
        b_end = b_row_ptrs[a_col + 1];
    }
    while (warp.any(b_begin < b_end)) {
        // empty local accumulator
        for (int i = lane; i < num_blocks; i += config::warp_size) {
            local_sparsity[i] = 0;
        }
        warp.sync();
        for (IndexType i = 0; i < a_size; ++i) {
            config::lane_mask_type success_mask{};
            IndexType b_col{};
            // load entries from the ith row until we exceed the accumulator
            do {
                auto b_cur_begin = warp.shfl(b_begin, i);
                auto b_cur_end = warp.shfl(b_end, i);
                b_col = checked_load(b_cols, b_cur_begin + lane, b_cur_end,
                                     sentinel);
                auto rel_col = b_col - c_first_col;
                auto mask = uint32{1} << (rel_col % 32);
                auto idx = rel_col / 32;
                if (rel_col < accumulator_size) {
                    atomicOr(local_sparsity + idx, mask);
                }
                success_mask = warp.ballot(rel_col < accumulator_size);
                b_cur_begin += popcnt(success_mask);
                if (lane == i) {
                    b_begin = b_cur_begin;
                }
            } while (success_mask == ~config::lane_mask_type{});
            // capture the first column exceeding the accumulator
            auto new_next_col = warp.shfl(b_col, popcnt(success_mask));
            if (lane == i) {
                b_next_col = new_next_col;
            }
        }
        // find the beginning column of the next dense accumulator block
        c_first_col = reduce(warp, b_next_col, [](IndexType a, IndexType b) {
            return min(a, b);
        });
        c_nnz += reduce(
            warp, popcnt(lane < num_blocks ? local_sparsity[lane] : uint32{}),
            [](IndexType a, IndexType b) { return a + b; });
    }
    return c_nnz;
}


template <typename ValueType, typename IndexType, typename AAccessor>
__device__ void merge_dense(AAccessor a, IndexType a_size,
                            IndexType c_first_col,
                            const IndexType *__restrict__ b_row_ptrs,
                            const IndexType *__restrict__ b_cols,
                            const ValueType *__restrict__ b_vals,
                            IndexType c_nz, IndexType *__restrict__ c_cols,
                            ValueType *__restrict__ c_vals)
{
    constexpr auto sentinel = device_numeric_limits<IndexType>::max;
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto lane = static_cast<IndexType>(warp.thread_rank());
    const auto lane_prefix_mask = (config::lane_mask_type{1} << lane) - 1;

    constexpr auto num_blocks = 4;
    constexpr auto accumulator_size = 32 * num_blocks;
    constexpr auto num_warps = spgemm_block_size / config::warp_size;
    const auto warp_idx = threadIdx.x / config::warp_size;
    __shared__ uint32 sh_sparsity[num_blocks * num_warps];
    __shared__ UninitializedArray<ValueType, accumulator_size * num_warps>
        sh_accumulator;

    auto local_sparsity = sh_sparsity + (warp_idx * num_blocks);
    auto local_accumulator = &sh_accumulator[warp_idx * accumulator_size];
    // vector: ith thread stores begin/end/scale of ith row
    IndexType b_begin{};
    IndexType b_end{};
    ValueType a_val{};
    // vector: beginning of next dense block column in ith row
    auto b_next_col = sentinel;
    if (lane < a_size) {
        auto a_col = a.col(lane);
        a_val = a.val(lane);
        b_begin = b_row_ptrs[a_col];
        b_end = b_row_ptrs[a_col + 1];
    }
    while (warp.any(b_begin < b_end)) {
        // zero local accumulator
        for (int i = lane; i < num_blocks; i += config::warp_size) {
            local_sparsity[i] = 0;
        }
        for (int i = lane; i < accumulator_size; i += config::warp_size) {
            local_accumulator[i] = zero<ValueType>();
        }
        warp.sync();
        for (IndexType i = 0; i < a_size; ++i) {
            config::lane_mask_type success_mask{};
            IndexType b_col{};
            // load entries from the ith row until we exceed the accumulator
            do {
                auto b_cur_begin = warp.shfl(b_begin, i);
                auto b_cur_end = warp.shfl(b_end, i);
                auto a_cur_val = warp.shfl(a_val, i);
                b_col = checked_load(b_cols, b_cur_begin + lane, b_cur_end,
                                     sentinel);
                auto b_val = checked_load(b_vals, b_cur_begin + lane, b_cur_end,
                                          zero<ValueType>());
                auto rel_col = b_col - c_first_col;
                auto mask = uint32{1} << (rel_col % 32);
                auto idx = rel_col / 32;
                if (rel_col < accumulator_size) {
                    atomicOr(local_sparsity + idx, mask);
                    // no atomicity necessary since columns are disjoint
                    local_accumulator[rel_col] += a_cur_val * b_val;
                }
                success_mask = warp.ballot(rel_col < accumulator_size);
                b_cur_begin += popcnt(success_mask);
                if (lane == i) {
                    b_begin = b_cur_begin;
                }
            } while (success_mask == ~config::lane_mask_type{});
            // capture the first column exceeding the accumulator
            auto new_next_col = warp.shfl(b_col, popcnt(success_mask));
            if (lane == i) {
                b_next_col = new_next_col;
            }
        }
        warp.sync();
        // store the contents of the dense accumulator
        for (int block = 0; block < num_blocks; ++block) {
            auto sparsity = local_sparsity[block];
            auto c_ofs = popcnt(sparsity & lane_prefix_mask);
            if ((sparsity >> lane) & 1) {
                c_cols[c_nz + c_ofs] = c_first_col + block * 32 + lane;
                c_vals[c_nz + c_ofs] = local_accumulator[block * 32 + lane];
            }
            c_nz += popcnt(sparsity);
        }
        // find the beginning column of the next dense accumulator block
        c_first_col = reduce(warp, b_next_col, [](IndexType a, IndexType b) {
            return min(a, b);
        });
    }
}


template <typename IndexType, typename AAccessor>
__device__ IndexType count_merge_nway(AAccessor a, IndexType a_size,
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

    IndexType c_nnz{};
    while (warp.any(b_begin < b_end)) {
        config::lane_mask_type new_advance_mask{};
        config::lane_mask_type old_advance_mask{};
        auto b_cur_begin = warp.shfl(b_begin, 0);
        auto b_cur_end = warp.shfl(b_end, 0);
        auto col =
            checked_load(b_cols, b_cur_begin + lane, b_cur_end, sentinel);
        for (int i = 1; i < a_size; ++i) {
            auto new_col = checked_load(b_cols, warp.shfl(b_begin, i) + lane,
                                        warp.shfl(b_end, i), sentinel);
            auto merge_result =
                group_merge_step<config::warp_size>(new_col, col, warp);
            if (max(merge_result.a_val, merge_result.b_val) == sentinel - 1) {
                // spread poison
                merge_result.a_val = sentinel - 1;
                merge_result.b_val = sentinel - 1;
            }
            auto mismatch_mask =
                warp.ballot(merge_result.a_val != merge_result.b_val);
            col = min(merge_result.a_val, merge_result.b_val);
            // 1 bit for every new column index
            auto new_column_mask = (mismatch_mask << 1) | 1;
            auto new_column_count = popcnt(new_column_mask);
            auto compressed_idx = bitmask_select(new_column_mask, lane);
            auto new_le_old_mask = merge_result.a_less_b_mask | ~mismatch_mask;
            auto new_ge_old_mask = ~merge_result.a_less_b_mask | ~mismatch_mask;
            // compress column indices to remove duplicates
            col = warp.shfl(col, compressed_idx);
            // "poison" the remaining entries
            if (lane >= new_column_count) {
                col = sentinel - 1;
            }
            // compress new_?e_old_mask and store
            auto compressed_new_advance =
                warp.ballot(((new_le_old_mask >> compressed_idx) & 1) &&
                            lane < new_column_count && col != sentinel - 1);
            auto compressed_old_advance =
                warp.ballot(((new_ge_old_mask >> compressed_idx) & 1) &&
                            lane < new_column_count && col != sentinel - 1);
            if (lane == i) {
                new_advance_mask = compressed_new_advance;
                old_advance_mask = compressed_old_advance;
            }
        }
        c_nnz += popcnt(warp.ballot(col < sentinel - 1));
        // advance indices by the number of consumed elements
        auto rest_advance = config::warp_size;
        for (int i = a_size - 1; i > 0; --i) {
            auto cur_prefix_mask = prefix_mask(rest_advance);
            auto local_new_advance =
                popcnt(warp.shfl(new_advance_mask, i) & cur_prefix_mask);
            rest_advance =
                popcnt(warp.shfl(old_advance_mask, i) & cur_prefix_mask);
            if (lane == i) {
                b_begin += local_new_advance;
            }
        }
        if (lane == 0) {
            b_begin += rest_advance;
        }
    }
    return c_nnz;
}


template <typename ValueType, typename IndexType, typename AAccessor>
__device__ void merge_nway(AAccessor a, IndexType a_size,
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

    while (warp.any(b_begin < b_end)) {
        config::lane_mask_type new_advance_mask{};
        config::lane_mask_type old_advance_mask{};
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
            if (max(merge_result.a_val, merge_result.b_val) == sentinel - 1) {
                // spread poison
                merge_result.a_val = sentinel - 1;
                merge_result.b_val = sentinel - 1;
            }
            auto mismatch_mask =
                warp.ballot(merge_result.a_val != merge_result.b_val);
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
            // 1 bit for every new column index
            auto new_column_mask = (mismatch_mask << 1) | 1;
            auto new_column_count = popcnt(new_column_mask);
            auto compressed_idx = bitmask_select(new_column_mask, lane);
            auto new_le_old_mask = merge_result.a_less_b_mask | ~mismatch_mask;
            auto new_ge_old_mask = ~merge_result.a_less_b_mask | ~mismatch_mask;
            // compress column indices to remove duplicates
            col = warp.shfl(col, compressed_idx);
            val = warp.shfl(val, compressed_idx);
            // "poison" the remaining entries
            if (lane >= new_column_count) {
                col = sentinel - 1;
            }
            // compress new_?e_old_mask and store
            auto compressed_new_advance =
                warp.ballot(((new_le_old_mask >> compressed_idx) & 1) &&
                            lane < new_column_count && col != sentinel - 1);
            auto compressed_old_advance =
                warp.ballot(((new_ge_old_mask >> compressed_idx) & 1) &&
                            lane < new_column_count && col != sentinel - 1);
            if (lane == i) {
                new_advance_mask = compressed_new_advance;
                old_advance_mask = compressed_old_advance;
            }
        }
        auto output = col < sentinel - 1;
        auto output_mask = warp.ballot(output);
        if (output) {
            auto c_idx = c_nz + popcnt(output_mask & prefix_mask(lane));
            c_cols[c_idx] = col;
            c_vals[c_idx] = val;
        }
        c_nz += popcnt(output_mask);
        // advance indices by the number of consumed elements
        auto rest_advance = config::warp_size;
        for (int i = a_size - 1; i > 0; --i) {
            auto cur_prefix_mask = prefix_mask(rest_advance);
            auto local_new_advance =
                popcnt(warp.shfl(new_advance_mask, i) & cur_prefix_mask);
            rest_advance =
                popcnt(warp.shfl(old_advance_mask, i) & cur_prefix_mask);
            if (lane == i) {
                b_begin += local_new_advance;
            }
        }
        if (lane == 0) {
            b_begin += rest_advance;
        }
    }
}


template <typename IndexType, typename AAccessor>
__device__ IndexType spgemm_count_dispatch(
    AAccessor a, IndexType a_size, const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols)
{
    if (a_size == 0) {
        return IndexType{};
    } else if (a_size == 1) {
        auto a_col = a.col(0);
        return b_row_ptrs[a_col + 1] - b_row_ptrs[a_col];
    } else {
        return count_merge_nway(a, a_size, b_row_ptrs, b_cols);
        // return count_dense(a, a_size, IndexType{}, b_row_ptrs, b_cols);
    }
}


template <typename ValueType, typename IndexType, typename AAccessor>
__device__ void spgemm_merge_dispatch(AAccessor a, IndexType a_size,
                                      const IndexType *__restrict__ b_row_ptrs,
                                      const IndexType *__restrict__ b_cols,
                                      const ValueType *__restrict__ b_vals,
                                      IndexType c_nz,
                                      IndexType *__restrict__ c_cols,
                                      ValueType *__restrict__ c_vals)
{
    if (a_size == 0) {
        // do nothing
    } else if (a_size == 1) {
        copy_row(a, b_row_ptrs, b_cols, b_vals, c_nz, c_cols, c_vals);
    } else {
        merge_nway(a, a_size, b_row_ptrs, b_cols, b_vals, c_nz, c_cols, c_vals);
        // merge_dense(a, a_size, IndexType{}, b_row_ptrs, b_cols, b_vals, c_nz,
        //            c_cols, c_vals);
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
    auto count = spgemm_count_dispatch(a, a_size, b_row_ptrs, b_cols);
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
    merge_row_accessor<float, IndexType> a{a_begin, a_size};
    auto count = spgemm_count_dispatch(a, a_size, b_row_ptrs, b_cols);
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

    const auto a_begin = a_row_ptrs[row];
    const auto a_end = a_row_ptrs[row + 1];
    const auto a_size = a_end - a_begin;
    memory_row_accessor<ValueType, IndexType> a{a_vals, a_cols, a_begin,
                                                a_size};
    spgemm_merge_dispatch(a, a_size, b_row_ptrs, b_cols, b_vals,
                          c_row_ptrs[row], c_cols, c_vals);
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

    const auto a_begin = a_row_ptrs[row];
    const auto a_end = a_row_ptrs[row + 1];
    const auto a_size = a_end - a_begin;
    merge_row_accessor<ValueType, IndexType> a{a_begin, a_size};
    spgemm_merge_dispatch(a, a_size, b_row_ptrs, b_cols, b_vals,
                          c_row_ptrs[row], c_cols, c_vals);
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


template <int subwarp_size, typename IndexType>
__global__ __launch_bounds__(default_block_size) void spgemm_analysis(
    const IndexType *__restrict__ a_row_ptrs, size_type num_rows,
    const IndexType *__restrict__ a_cols,
    const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, IndexType *__restrict__ min_cols,
    IndexType *__restrict__ max_cols, IndexType *__restrict__ max_nnzs,
    IndexType *__restrict__ sum_nnzs, IndexType *__restrict__ tmp_max_nnz,
    IndexType *__restrict__ tmp_sum_nnz)
{
    constexpr auto subwarp_count = default_block_size / subwarp_size;
    __shared__ IndexType total_sum_nnz[subwarp_count];
    auto row = thread::get_subwarp_id_flat<subwarp_size>();
    auto thread_block = group::this_thread_block();
    auto subwarp = group::tiled_partition<subwarp_size>(thread_block);
    const auto lane = subwarp.thread_rank();
    const auto subwarp_id = threadIdx.x / subwarp_size;
    auto min_op = [](IndexType a, IndexType b) { return min(a, b); };
    auto max_op = [](IndexType a, IndexType b) { return max(a, b); };
    auto sum_op = [](IndexType a, IndexType b) { return a + b; };

    if (lane == 0) {
        total_max_nnz[subwarp_id] = 0;
        total_sum_nnz[subwarp_id] = 0;
    }
    if (row < num_rows) {
        auto a_begin = a_row_ptrs[row];
        auto a_end = a_row_ptrs[row + 1];
        constexpr auto sentinel = device_numeric_limits<IndexType>::max();
        IndexType min_col{sentinel};
        IndexType max_col{};
        IndexType max_nnz{};
        IndexType sum_nnz{};
        for (auto a_nz = a_begin + lane; a_nz < a_end; a += subwarp_size) {
            auto a_col = a_cols[a_nz];
            auto b_begin = b_row_ptrs[a_col];
            auto b_end = b_row_ptrs[a_col + 1];
            auto b_size = b_end - b_begin;
            min_col = min(min_col, b_size > 0 ? b_col[b_begin] : sentinel);
            max_col = max(max_col, b_size > 0 ? b_col[b_end - 1] : IndexType{});
            max_nnz = max(max_nnz, b_size);
            sum_nnz += b_size;
        }
        min_col = reduce(subwarp, min_col, min_op);
        max_col = reduce(subwarp, max_col, max_op);
        max_nnz = reduce(subwarp, max_nnz, max_op);
        sum_nnz = reduce(subwarp, sum_nnz, sum_op);
        if (lane == 0) {
            min_cols[row] = min_col;
            max_cols[row] = max_col;
            max_nnzs[row] = max_nnz;
            sum_nnzs[row] = sum_nnz;
            total_sum_nnz[subwarp_id] = max_nnz;
        }
    }
    block.sync();
    if (threadIdx.x >= config::warp_size) {
        return;
    }
    auto warp = group::tiled_partition<config::warp_size>(thread_block);
    static_assert(subwarp_count <= config::warp_size, "block too big");
    auto warp_lane = warp.thread_rank();
    tmp_max_nnz[blockIdx.x] = reduce(warp, total_sum_nnz[warp_lane], max_op);
    tmp_sum_nnz[blockIdx.x] = reduce(warp, total_sum_nnz[warp_lane], sum_op);
}