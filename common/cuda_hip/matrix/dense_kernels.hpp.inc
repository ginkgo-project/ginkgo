/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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


template <typename ValueType, typename IndexType>
__global__
__launch_bounds__(default_block_size) void count_nonzero_blocks_per_row(
    size_type num_block_rows, size_type num_block_cols, size_type stride,
    int block_size, const ValueType* __restrict__ source,
    IndexType* __restrict__ block_row_nnz)
{
    const auto brow =
        thread::get_subwarp_id_flat<config::warp_size, IndexType>();

    if (brow >= num_block_rows) {
        return;
    }

    const auto num_cols = num_block_cols * block_size;
    auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto lane = static_cast<IndexType>(warp.thread_rank());
    constexpr auto full_mask = ~config::lane_mask_type{};
    constexpr auto one_mask = config::lane_mask_type{1};
    bool first_block_nonzero = false;
    IndexType block_count{};
    for (IndexType base_col = 0; base_col < num_cols;
         base_col += config::warp_size) {
        const auto col = base_col + lane;
        const auto block_local_col = col % block_size;
        // which is the first column in the current block?
        const auto block_base_col = col - block_local_col;
        // collect nonzero bitmask
        bool local_nonzero = false;
        for (int local_row = 0; local_row < block_size; local_row++) {
            const auto row = local_row + brow * block_size;
            local_nonzero |=
                col < num_cols && is_nonzero(source[row * stride + col]);
        }
        auto nonzero_mask =
            warp.ballot(local_nonzero) | (first_block_nonzero ? 1u : 0u);
        // only consider threads in the current block
        const auto first_thread = block_base_col - base_col;
        const auto last_thread = first_thread + block_size;
        // HIP compiles these assertions in Release, traps unconditionally
        // assert(first_thread < int(config::warp_size));
        // assert(last_thread >= 0);
        // mask off everything below first_thread
        const auto lower_mask =
            first_thread < 0 ? full_mask : ~((one_mask << first_thread) - 1u);
        // mask off everything from last_thread
        const auto upper_mask = last_thread >= config::warp_size
                                    ? full_mask
                                    : ((one_mask << last_thread) - 1u);
        const auto block_mask = upper_mask & lower_mask;
        const auto local_mask = nonzero_mask & block_mask;
        // last column in the block increments the counter
        block_count +=
            (block_local_col == block_size - 1 && local_mask) ? 1 : 0;
        // if we need to store something for the next iteration
        if ((base_col + config::warp_size) % block_size != 0) {
            // check whether the last block (incomplete) in this warp is nonzero
            auto local_block_nonzero_mask = warp.ballot(local_mask != 0u);
            bool last_block_nonzero =
                (local_block_nonzero_mask >> (config::warp_size - 1)) != 0u;
            first_block_nonzero = last_block_nonzero;
        } else {
            first_block_nonzero = false;
        }
    }
    block_count = reduce(warp, block_count,
                         [](IndexType a, IndexType b) { return a + b; });
    if (lane == 0) {
        block_row_nnz[brow] = block_count;
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void convert_to_fbcsr(
    size_type num_block_rows, size_type num_block_cols, size_type stride,
    int block_size, const ValueType* __restrict__ source,
    const IndexType* __restrict__ block_row_ptrs,
    IndexType* __restrict__ block_cols, ValueType* __restrict__ blocks)
{
    const auto brow =
        thread::get_subwarp_id_flat<config::warp_size, IndexType>();

    if (brow >= num_block_rows) {
        return;
    }

    const auto bs_sq = block_size * block_size;
    const auto num_cols = num_block_cols * block_size;
    auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto lane = static_cast<IndexType>(warp.thread_rank());
    constexpr auto full_mask = ~config::lane_mask_type{};
    constexpr auto one_mask = config::lane_mask_type{1};
    const auto lane_prefix_mask = (one_mask << warp.thread_rank()) - 1u;
    bool first_block_nonzero = false;
    auto block_base_nz = block_row_ptrs[brow];
    for (IndexType base_col = 0; base_col < num_cols;
         base_col += config::warp_size) {
        const auto col = base_col + lane;
        const auto block_local_col = col % block_size;
        // which is the first column in the current block?
        const auto block_base_col = col - block_local_col;
        // collect nonzero bitmask
        bool local_nonzero = false;
        for (int local_row = 0; local_row < block_size; local_row++) {
            const auto row = local_row + brow * block_size;
            local_nonzero |=
                col < num_cols && is_nonzero(source[row * stride + col]);
        }
        auto nonzero_mask =
            warp.ballot(local_nonzero) | (first_block_nonzero ? 1u : 0u);
        // only consider threads in the current block
        const auto first_thread = block_base_col - base_col;
        const auto last_thread = first_thread + block_size;
        // HIP compiles these assertions in Release, traps unconditionally
        // assert(first_thread < int(config::warp_size));
        // assert(last_thread >= 0);
        // mask off everything below first_thread
        const auto lower_mask =
            first_thread < 0 ? full_mask : ~((one_mask << first_thread) - 1u);
        // mask off everything from last_thread
        const auto upper_mask = last_thread >= config::warp_size
                                    ? full_mask
                                    : ((one_mask << last_thread) - 1u);
        const auto block_mask = upper_mask & lower_mask;
        const auto local_mask = nonzero_mask & block_mask;
        const auto block_nonzero_mask =
            warp.ballot(local_mask && (block_local_col == block_size - 1));

        // count how many Fbcsr blocks come before the Fbcsr block handled by
        // the local group of threads
        const auto block_nz =
            block_base_nz + popcnt(block_nonzero_mask & lane_prefix_mask);
        // now in a second sweep, store the actual elements
        if (local_mask) {
            if (block_local_col == block_size - 1) {
                block_cols[block_nz] = col / block_size;
            }
            // only if we encountered elements in this column
            if (local_nonzero) {
                for (int local_row = 0; local_row < block_size; local_row++) {
                    const auto row = local_row + brow * block_size;
                    blocks[local_row + block_local_col * block_size +
                           block_nz * bs_sq] = source[row * stride + col];
                }
            }
        }
        // if we need to store something for the next iteration
        if ((base_col + config::warp_size) % block_size != 0) {
            // check whether the last block (incomplete) in this warp is nonzero
            auto local_block_nonzero_mask = warp.ballot(local_mask != 0u);
            bool last_block_nonzero =
                (local_block_nonzero_mask >> (config::warp_size - 1)) != 0u;
            first_block_nonzero = last_block_nonzero;
        } else {
            first_block_nonzero = false;
        }
        // advance by the completed blocks
        block_base_nz += popcnt(block_nonzero_mask);
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_coo(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType* __restrict__ source, const int64* __restrict__ row_ptrs,
    IndexType* __restrict__ row_idxs, IndexType* __restrict__ col_idxs,
    ValueType* __restrict__ values)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();

    if (row < num_rows) {
        auto warp = group::tiled_partition<config::warp_size>(
            group::this_thread_block());
        auto lane_prefix_mask =
            (config::lane_mask_type(1) << warp.thread_rank()) - 1;
        auto base_out_idx = row_ptrs[row];
        for (size_type i = 0; i < num_cols; i += config::warp_size) {
            const auto col = i + warp.thread_rank();
            const auto pred =
                col < num_cols ? is_nonzero(source[stride * row + col]) : false;
            const auto mask = warp.ballot(pred);
            const auto out_idx = base_out_idx + popcnt(mask & lane_prefix_mask);
            if (pred) {
                values[out_idx] = source[stride * row + col];
                col_idxs[out_idx] = col;
                row_idxs[out_idx] = row;
            }
            base_out_idx += popcnt(mask);
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_csr(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType* __restrict__ source, IndexType* __restrict__ row_ptrs,
    IndexType* __restrict__ col_idxs, ValueType* __restrict__ values)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();

    if (row < num_rows) {
        auto warp = group::tiled_partition<config::warp_size>(
            group::this_thread_block());
        auto lane_prefix_mask =
            (config::lane_mask_type(1) << warp.thread_rank()) - 1;
        auto base_out_idx = row_ptrs[row];
        for (size_type i = 0; i < num_cols; i += config::warp_size) {
            const auto col = i + warp.thread_rank();
            const auto pred =
                col < num_cols ? is_nonzero(source[stride * row + col]) : false;
            const auto mask = warp.ballot(pred);
            const auto out_idx = base_out_idx + popcnt(mask & lane_prefix_mask);
            if (pred) {
                values[out_idx] = source[stride * row + col];
                col_idxs[out_idx] = col;
            }
            base_out_idx += popcnt(mask);
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_sparsity_csr(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType* __restrict__ source, IndexType* __restrict__ row_ptrs,
    IndexType* __restrict__ col_idxs)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();

    if (row < num_rows) {
        auto warp = group::tiled_partition<config::warp_size>(
            group::this_thread_block());
        auto lane_prefix_mask =
            (config::lane_mask_type(1) << warp.thread_rank()) - 1;
        auto base_out_idx = row_ptrs[row];
        for (size_type i = 0; i < num_cols; i += config::warp_size) {
            const auto col = i + warp.thread_rank();
            const auto pred =
                col < num_cols ? is_nonzero(source[stride * row + col]) : false;
            const auto mask = warp.ballot(pred);
            const auto out_idx = base_out_idx + popcnt(mask & lane_prefix_mask);
            if (pred) {
                col_idxs[out_idx] = col;
            }
            base_out_idx += popcnt(mask);
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_ell(
    size_type num_rows, size_type num_cols, size_type source_stride,
    const ValueType* __restrict__ source, size_type max_nnz_per_row,
    size_type result_stride, IndexType* __restrict__ col_idxs,
    ValueType* __restrict__ values)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();

    if (row < num_rows) {
        auto warp = group::tiled_partition<config::warp_size>(
            group::this_thread_block());
        auto lane_prefix_mask =
            (config::lane_mask_type(1) << warp.thread_rank()) - 1;
        size_type base_out_idx{};
        for (size_type i = 0; i < num_cols; i += config::warp_size) {
            const auto col = i + warp.thread_rank();
            const auto pred =
                col < num_cols ? is_nonzero(source[source_stride * row + col])
                               : false;
            const auto mask = warp.ballot(pred);
            const auto out_idx =
                row + (base_out_idx + popcnt(mask & lane_prefix_mask)) *
                          result_stride;
            if (pred) {
                values[out_idx] = source[source_stride * row + col];
                col_idxs[out_idx] = col;
            }
            base_out_idx += popcnt(mask);
        }
        for (size_type i = base_out_idx + warp.thread_rank();
             i < max_nnz_per_row; i += config::warp_size) {
            const auto out_idx = row + i * result_stride;
            values[out_idx] = zero<ValueType>();
            col_idxs[out_idx] = invalid_index<IndexType>();
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_hybrid(
    size_type num_rows, size_type num_cols, size_type source_stride,
    const ValueType* __restrict__ source, size_type ell_max_nnz_per_row,
    size_type ell_stride, IndexType* __restrict__ ell_col_idxs,
    ValueType* __restrict__ ell_values, const int64* __restrict__ coo_row_ptrs,
    IndexType* __restrict__ coo_row_idxs, IndexType* __restrict__ coo_col_idxs,
    ValueType* __restrict__ coo_values)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();

    if (row < num_rows) {
        auto warp = group::tiled_partition<config::warp_size>(
            group::this_thread_block());
        auto lane_prefix_mask =
            (config::lane_mask_type(1) << warp.thread_rank()) - 1;
        size_type base_out_idx{};
        const auto coo_out_begin = coo_row_ptrs[row];
        for (size_type i = 0; i < num_cols; i += config::warp_size) {
            const auto col = i + warp.thread_rank();
            const auto pred =
                col < num_cols ? is_nonzero(source[source_stride * row + col])
                               : false;
            const auto mask = warp.ballot(pred);
            const auto cur_out_idx =
                base_out_idx + popcnt(mask & lane_prefix_mask);
            if (pred) {
                if (cur_out_idx < ell_max_nnz_per_row) {
                    const auto out_idx = row + cur_out_idx * ell_stride;
                    ell_values[out_idx] = source[source_stride * row + col];
                    ell_col_idxs[out_idx] = col;
                } else {
                    const auto out_idx =
                        cur_out_idx - ell_max_nnz_per_row + coo_out_begin;
                    coo_values[out_idx] = source[source_stride * row + col];
                    coo_col_idxs[out_idx] = col;
                    coo_row_idxs[out_idx] = row;
                }
            }
            base_out_idx += popcnt(mask);
        }
        for (size_type i = base_out_idx + warp.thread_rank();
             i < ell_max_nnz_per_row; i += config::warp_size) {
            const auto out_idx = row + i * ell_stride;
            ell_values[out_idx] = zero<ValueType>();
            ell_col_idxs[out_idx] = invalid_index<IndexType>();
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_sellp(
    size_type num_rows, size_type num_cols, size_type slice_size,
    size_type stride, const ValueType* __restrict__ source,
    size_type* __restrict__ slice_sets, IndexType* __restrict__ col_idxs,
    ValueType* __restrict__ values)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    const auto local_row = row % slice_size;
    const auto slice = row / slice_size;

    if (row < num_rows) {
        auto warp = group::tiled_partition<config::warp_size>(
            group::this_thread_block());
        const auto lane = warp.thread_rank();
        const auto prefix_mask = (config::lane_mask_type{1} << lane) - 1;
        const auto slice_end = slice_sets[slice + 1] * slice_size;
        auto base_idx = slice_sets[slice] * slice_size + local_row;
        for (size_type i = 0; i < num_cols; i += config::warp_size) {
            const auto col = i + lane;
            const auto val = checked_load(source + stride * row, col, num_cols,
                                          zero<ValueType>());
            const auto pred = is_nonzero(val);
            const auto mask = warp.ballot(pred);
            const auto idx = base_idx + popcnt(mask & prefix_mask) * slice_size;
            if (pred) {
                values[idx] = val;
                col_idxs[idx] = col;
            }
            base_idx += popcnt(mask) * slice_size;
        }
        for (auto i = base_idx + lane * slice_size; i < slice_end;
             i += config::warp_size * slice_size) {
            values[i] = zero<ValueType>();
            col_idxs[i] = invalid_index<IndexType>();
        }
    }
}


}  // namespace kernel
