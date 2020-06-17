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

template <int warps_per_block>
__global__
__launch_bounds__(warps_per_block *config::warp_size) void duplicate_array(
    const precision_reduction *__restrict__ source, size_type source_size,
    precision_reduction *__restrict__ dest, size_type dest_size)
{
    auto grid = group::this_grid();
    if (grid.thread_rank() >= dest_size) {
        return;
    }
    for (auto i = grid.thread_rank(); i < dest_size; i += grid.size()) {
        dest[i] = source[i % source_size];
    }
}


template <typename IndexType>
__global__ void compare_adjacent_rows(size_type num_rows, int32 max_block_size,
                                      const IndexType *__restrict__ row_ptrs,
                                      const IndexType *__restrict__ col_idx,
                                      bool *__restrict__ matching_next_row)
{
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto local_tid = warp.thread_rank();
    const auto warp_id = thread::get_subwarp_id_flat<config::warp_size>();

    if (warp_id >= num_rows - 1) {
        return;
    }

    const auto curr_row_start = row_ptrs[warp_id];
    const auto next_row_start = row_ptrs[warp_id + 1];
    const auto next_row_end = row_ptrs[warp_id + 2];

    const auto nz_this_row = next_row_end - next_row_start;
    const auto nz_prev_row = next_row_start - curr_row_start;

    if (nz_this_row != nz_prev_row) {
        matching_next_row[warp_id] = false;
        return;
    }
    size_type steps = ceildiv(nz_this_row, config::warp_size);
    for (size_type i = 0; i < steps; i++) {
        auto j = local_tid + i * config::warp_size;
        auto prev_col = (curr_row_start + j < next_row_start)
                            ? col_idx[curr_row_start + j]
                            : 0;
        auto this_col = (curr_row_start + j < next_row_start)
                            ? col_idx[next_row_start + j]
                            : 0;
        if (warp.any(prev_col != this_col)) {
            matching_next_row[warp_id] = false;
            return;
        }
    }
    matching_next_row[warp_id] = true;
}


template <typename IndexType>
__global__ void generate_natural_block_pointer(
    size_type num_rows, int32 max_block_size,
    const bool *__restrict__ matching_next_row,
    IndexType *__restrict__ block_ptrs, size_type *__restrict__ num_blocks_arr)
{
    block_ptrs[0] = 0;
    if (num_rows == 0) {
        return;
    }
    size_type num_blocks = 1;
    int32 current_block_size = 1;
    for (size_type i = 0; i < num_rows - 1; ++i) {
        if ((matching_next_row[i]) && (current_block_size < max_block_size)) {
            ++current_block_size;
        } else {
            block_ptrs[num_blocks] =
                block_ptrs[num_blocks - 1] + current_block_size;
            ++num_blocks;
            current_block_size = 1;
        }
    }
    block_ptrs[num_blocks] = block_ptrs[num_blocks - 1] + current_block_size;
    num_blocks_arr[0] = num_blocks;
}


template <typename IndexType>
__global__ void agglomerate_supervariables_kernel(
    int32 max_block_size, size_type num_natural_blocks,
    IndexType *__restrict__ block_ptrs, size_type *__restrict__ num_blocks_arr)
{
    num_blocks_arr[0] = 0;
    if (num_natural_blocks == 0) {
        return;
    }
    size_type num_blocks = 1;
    int32 current_block_size = block_ptrs[1] - block_ptrs[0];
    for (size_type i = 1; i < num_natural_blocks; ++i) {
        const int32 block_size = block_ptrs[i + 1] - block_ptrs[i];
        if (current_block_size + block_size <= max_block_size) {
            current_block_size += block_size;
        } else {
            block_ptrs[num_blocks] = block_ptrs[i];
            ++num_blocks;
            current_block_size = block_size;
        }
    }
    block_ptrs[num_blocks] = block_ptrs[num_natural_blocks];
    num_blocks_arr[0] = num_blocks;
}


template <bool conjugate, int max_block_size, int subwarp_size,
          int warps_per_block, typename ValueType, typename IndexType>
__global__ void __launch_bounds__(warps_per_block *config::warp_size)
    transpose_jacobi(const ValueType *__restrict__ blocks,
                     preconditioner::block_interleaved_storage_scheme<IndexType>
                         storage_scheme,
                     const IndexType *__restrict__ block_ptrs,
                     size_type num_blocks, ValueType *__restrict__ out_blocks)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    if (block_id >= num_blocks) {
        return;
    }
    const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];

    const auto block_ofs = storage_scheme.get_global_block_offset(block_id);
    const auto block_stride = storage_scheme.get_stride();
    const auto rank = subwarp.thread_rank();
    if (rank < block_size) {
        for (IndexType i = 0; i < block_size; ++i) {
            auto val = blocks[block_ofs + i * block_stride + rank];
            out_blocks[block_ofs + i + rank * block_stride] =
                conjugate ? conj(val) : val;
        }
    }
}


template <bool conjugate, int max_block_size, int subwarp_size,
          int warps_per_block, typename ValueType, typename IndexType>
__global__ void
__launch_bounds__(warps_per_block *config::warp_size) adaptive_transpose_jacobi(
    const ValueType *__restrict__ blocks,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    const precision_reduction *__restrict__ block_precisions,
    const IndexType *__restrict__ block_ptrs, size_type num_blocks,
    ValueType *__restrict__ out_blocks)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    if (block_id >= num_blocks) {
        return;
    }
    const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];

    const auto block_stride = storage_scheme.get_stride();
    const auto rank = subwarp.thread_rank();
    if (rank < block_size) {
        GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(
            ValueType, block_precisions[block_id],
            auto local_block =
                reinterpret_cast<const resolved_precision *>(
                    blocks + storage_scheme.get_group_offset(block_id)) +
                storage_scheme.get_block_offset(block_id);
            auto local_out_block =
                reinterpret_cast<resolved_precision *>(
                    out_blocks + storage_scheme.get_group_offset(block_id)) +
                storage_scheme.get_block_offset(block_id);
            for (IndexType i = 0; i < block_size; ++i) {
                auto val = local_block[i * block_stride + rank];
                local_out_block[i + rank * block_stride] =
                    conjugate ? conj(val) : val;
            });
    }
}
