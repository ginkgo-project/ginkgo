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

__global__
    __launch_bounds__(default_block_size) void extract_common_block_pattern_kernel(
        const int nrows, const int* const __restrict__ sys_row_ptrs,
        const int* const __restrict__ sys_col_idxs,
        const gko::size_type num_blocks,
        const gko::preconditioner::batched_blocks_storage_scheme storage_scheme,
        const int* const __restrict__ block_pointers,
        const int* const __restrict__ row_in_which_block_info,
        int* const __restrict__ blocks_pattern)
{
    constexpr auto tile_size =
        config::warp_size;  // use full warp for coalesced memory accesses
    auto thread_block = group::this_thread_block();
    auto warp_grp = group::tiled_partition<tile_size>(thread_block);
    const int warp_id_in_grid = thread::get_subwarp_id_flat<tile_size, int>();
    const int total_num_warps_in_grid =
        thread::get_subwarp_num_flat<tile_size, int>();
    const int id_within_warp = warp_grp.thread_rank();

    // one warp per row of the matrix
    for (int row_idx = warp_id_in_grid; row_idx < nrows;
         row_idx += total_num_warps_in_grid) {
        const int block_idx = row_in_which_block_info[row_idx];
        const int idx_start = block_pointers[block_idx];
        const int idx_end = block_pointers[block_idx + 1];
        int* pattern_ptr =
            blocks_pattern + storage_scheme.get_block_offset(block_idx);

        for (int i = sys_row_ptrs[row_idx] + id_within_warp;
             i < sys_row_ptrs[row_idx + 1]; i += tile_size) {
            const int col_idx = sys_col_idxs[i];  // coalesced accesses

            if (col_idx >= idx_start && col_idx < idx_end) {
                // element at (row_idx, col_idx) is part of the diagonal block
                // store it into the pattern
                const int dense_block_row = row_idx - idx_start;
                const int dense_block_col = col_idx - idx_start;

                // The pattern is stored in row-major order
                pattern_ptr[dense_block_row * storage_scheme.get_stride() +
                            dense_block_col] = i;  // coalesced accesses
            }
        }
    }
}

template <typename BatchMatrixType, typename ValueType>
void batch_jacobi_apply_helper(
    const BatchMatrixType& sys_mat_batch, const size_type num_blocks,
    const uint32 max_block_size,
    const gko::preconditioner::batched_blocks_storage_scheme& storage_scheme,
    const ValueType* const blocks_array, const int* const block_ptrs,
    const int* const row_part_of_which_block_info,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z)
{
    const auto nbatch = sys_mat_batch.num_batch;
    const auto nrows = sys_mat_batch.num_rows;

    const auto r_ub = get_batch_struct(r);
    const auto z_ub = get_batch_struct(z);

    if (max_block_size == 1u) {
        const auto shared_size =
            BatchScalarJacobi<ValueType>::dynamic_work_size(
                sys_mat_batch.num_rows, sys_mat_batch.num_nnz) *
            sizeof(ValueType);
        auto prec_scalar_jacobi = BatchScalarJacobi<device_type<ValueType>>();

        batch_scalar_jacobi_apply<<<nbatch, default_block_size, shared_size>>>(
            prec_scalar_jacobi, sys_mat_batch, nbatch, nrows,
            as_device_type(r->get_const_values()),
            as_device_type(z->get_values()));

    } else {
        const auto shared_size =
            BatchBlockJacobi<ValueType>::dynamic_work_size(
                sys_mat_batch.num_rows, sys_mat_batch.num_nnz) *
            sizeof(ValueType);
        auto prec_block_jacobi = BatchBlockJacobi<device_type<ValueType>>(
            num_blocks, storage_scheme, as_device_type(blocks_array),
            block_ptrs, row_part_of_which_block_info);

        batch_block_jacobi_apply<<<nbatch, default_block_size, shared_size>>>(
            prec_block_jacobi, nbatch, nrows,
            as_device_type(r->get_const_values()),
            as_device_type(z->get_values()));
    }
}