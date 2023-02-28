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


__dpct_inline__ void find_row_is_part_of_which_block_kernel(
    const size_type num_blocks, const int* const __restrict__ block_pointers,
    int* const __restrict__ row_part_of_which_block_info,
    sycl::nd_item<3> item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_size = sg.get_local_range().size();
    const int sg_tid = sg.get_local_id();
    const int sg_global_id =
        item_ct1.get_group_linear_id() * num_blocks + sg.get_group_id();

    auto block_start = block_pointers[sg_global_id];
    auto block_end = block_pointers[sg_global_id + 1];
    auto block_size = block_end - block_start;

    if (sg_tid < block_size) {
        for (int i = block_start; i < block_end; i += sg_size) {
            row_part_of_which_block_info[i + sg_tid] = sg_global_id;
        }
    }
}


__dpct_inline__ void extract_common_block_pattern_kernel(
    const int nrows, const int* const __restrict__ sys_row_ptrs,
    const int* const __restrict__ sys_col_idxs, const size_type num_blocks,
    const preconditioner::batched_jacobi_blocks_storage_scheme<int>
        storage_scheme,
    const int* const __restrict__ blocks_cumulative_storage,
    const int* const __restrict__ block_pointers,
    const int* const __restrict__ row_in_which_block_info,
    int* const blocks_pattern, sycl::nd_item<3> item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_size = sg.get_local_range().size();
    const int sg_tid = sg.get_local_id();
    const int sg_global_id =
        item_ct1.get_group_linear_id() * num_blocks + sg.get_group_id();
    const int num_sg_total = item_ct1.get_group_range().size() * num_blocks;

    // one warp per row of the matrix
    for (int row_idx = sg_global_id; row_idx < nrows; row_idx += num_sg_total) {
        const int block_idx = row_in_which_block_info[row_idx];
        const int idx_start = block_pointers[block_idx];
        const int idx_end = block_pointers[block_idx + 1];
        int* __restrict__ pattern_ptr =
            blocks_pattern + storage_scheme.get_block_offset(
                                 block_idx, blocks_cumulative_storage);
        const auto stride =
            storage_scheme.get_stride(block_idx, block_pointers);

        for (int i = sys_row_ptrs[row_idx] + sg_tid;
             i < sys_row_ptrs[row_idx + 1]; i += sg_size) {
            const int col_idx = sys_col_idxs[i];  // coalesced accesses

            if (col_idx >= idx_start && col_idx < idx_end) {
                // element at (row_idx, col_idx) is part of the diagonal block
                // store it into the pattern
                const int dense_block_row = row_idx - idx_start;
                const int dense_block_col = col_idx - idx_start;

                // The pattern is stored in row-major order
                pattern_ptr[dense_block_row * stride + dense_block_col] =
                    i;  // coalesced accesses
            }
        }
    }
}


template <typename ValueType>
__dpct_inline__ int choose_pivot(const int block_size,
                                 const ValueType* const __restrict__ block_row,
                                 const int& perm, int k,
                                 sycl::nd_item<3> item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_size = sg.get_local_range().size();
    const int sg_tid = sg.get_local_id();

    auto my_abs_ele = abs(block_row[k]);
    if (perm > -1) {
        my_abs_ele = -1;
    }
    if (sg_tid >= block_size) {
        my_abs_ele = -1;
    }
    sg.barrier();
    int my_piv_idx = sg_tid;
    for (int a = sg_size / 2; a > 0; a /= 2) {
        const auto abs_ele_other = sg.shuffle_down(my_abs_ele, a);
        const int piv_idx_other = sg.shuffle_down(my_piv_idx, a);
        if (my_abs_ele < abs_ele_other) {
            my_abs_ele = abs_ele_other;
            my_piv_idx = piv_idx_other;
        }
    }
    sg.barrier();
    const int ipiv = sg.shuffle(my_piv_idx, 0);
    return ipiv;
}


template <typename ValueType>
__dpct_inline__ void invert_dense_block(const int block_size,
                                        ValueType* const block_row, int& perm,
                                        sycl::nd_item<3> item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_size = sg.get_local_range().size();
    const int sg_tid = sg.get_local_id();
    // Gauss Jordan Elimination with implicit pivoting
    for (int k = 0; k < block_size; k++) {
        // implicit pivoting
        const int ipiv = choose_pivot(block_size, block_row, perm, k,
                                      item_ct1);  // pivot index
        if (sg_tid == ipiv) {
            perm = k;
        }
        const ValueType d = sg.shuffle(block_row[k], ipiv);
        // scale kth col
        block_row[k] /= -d;
        if (sg_tid == ipiv) {
            block_row[k] = zero<ValueType>();
        }
        const ValueType row_val = block_row[k];
        // GER
        for (int col = 0; col < block_size; col++) {
            const ValueType col_val = sg.shuffle(block_row[col], ipiv);
            block_row[col] += row_val * col_val;
        }
        // Computations for the threads of the subwarp having local id >=
        // block_size are meaningless.
        // scale ipiv th row
        if (sg_tid == ipiv) {
            for (int i = 0; i < block_size; i++) {
                block_row[i] /= d;
            }
            block_row[k] = one<ValueType>() / d;
        }
    }
}


template <typename ValueType>
__dpct_inline__ void compute_block_jacobi_kernel(
    const size_type nbatch, const int nnz, const ValueType* const A_vals,
    const size_type num_blocks,
    const preconditioner::batched_jacobi_blocks_storage_scheme<int>
        storage_scheme,
    const int* const __restrict__ blocks_cumulative_storage,
    const int* const __restrict__ block_pointers,
    const int* const blocks_pattern, ValueType* const blocks,
    sycl::nd_item<3> item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    constexpr int sg_size = config::warp_size;
    const int sg_tid = sg.get_local_id();
    const int sg_global_id =
        item_ct1.get_group_linear_id() * num_blocks + sg.get_group_id();
    const int num_sg_total = item_ct1.get_group_range().size() * num_blocks;

    // one subwarp per small diagonal block
    const auto batch_idx = sg_global_id / num_blocks;
    const auto block_idx = sg_global_id % num_blocks;

    ValueType block_row[sg_size];
    const auto block_size =
        block_pointers[block_idx + 1] - block_pointers[block_idx];
    assert(block_size <= sg_size);

    const int* __restrict__ current_block_pattern =
        blocks_pattern +
        storage_scheme.get_block_offset(block_idx, blocks_cumulative_storage);
    ValueType* __restrict__ current_block_data =
        blocks +
        storage_scheme.get_global_block_offset(batch_idx, num_blocks, block_idx,
                                               blocks_cumulative_storage);
    const auto stride = storage_scheme.get_stride(block_idx, block_pointers);

    // each thread of the subwarp stores the column of the dense block/row
    // of the transposed block in its local memory
    if (sg_tid < block_size) {
        for (int a = 0; a < block_size; a++) {
            const auto idx =
                current_block_pattern[a * storage_scheme.get_stride(
                                              block_idx, block_pointers) +
                                      sg_tid];  // coalseced accesses
            ValueType val_to_fill = zero<ValueType>();
            if (idx >= 0) {
                assert(idx < nnz);
                val_to_fill = A_vals[idx + nnz * batch_idx];
            }
            block_row[a] = val_to_fill;
        }
    }

    int perm = -1;
    // invert
    invert_dense_block(
        block_size, block_row, perm,
        item_ct1);  // invert the transpose of the dense block.
                    // Note: Each thread of the subwarp has a row of the block
                    // to be inverted. (local id: 0 thread has 0th row, 1st
                    // thread has 1st row and so on..) If block_size <
                    // subwarp_size, then threads with local id >= block_size do
                    // not mean anything. Also, values in the block_row for
                    // index >= block_size are meaningless
    sg.barrier();

    // write back the tranpose of the transposed inverse matrix to block
    // array
    for (int a = 0; a < block_size; a++) {
        const int col_inv_transposed_mat = a;
        const int col = sg.shuffle(perm, a);  // column permutation
        const int row_inv_transposed_mat =
            perm;  // accumulated row swaps during pivoting
        const auto val_to_write = block_row[col];

        const int row_diag_block = col_inv_transposed_mat;
        const int col_diag_block = row_inv_transposed_mat;

        if (sg_tid < block_size) {
            current_block_data[row_diag_block * stride + col_diag_block] =
                val_to_write;  // non-coalesced accesses due to pivoting
        }
    }
    sg.barrier();
}


template <typename ValueType>
__dpct_inline__ void transpose_block_jacobi_kernel(
    const size_type nbatch, const int nrows, const size_type num_blocks,
    const int* const __restrict__ block_pointers,
    const ValueType* const blocks_array,
    const preconditioner::batched_jacobi_blocks_storage_scheme<int>
        storage_scheme,
    const int* const __restrict__ blocks_cumulative_storage,
    const int* const row_part_of_which_block_info,
    ValueType* const out_blocks_array, const bool to_conjugate,
    sycl::nd_item<3> item_ct1)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_size = sg.get_local_range().size();
    const int sg_tid = sg.get_local_id();
    const int sg_global_id =
        item_ct1.get_group_linear_id() * num_blocks + sg.get_group_id();
    const int num_sg_total = item_ct1.get_group_range().size() * num_blocks;

    // one subwarp per row of the small diagonal block
    const auto batch_idx = sg_global_id / nrows;
    const auto row_idx = sg_global_id % nrows;

    const auto block_idx = row_part_of_which_block_info[row_idx];
    const auto row_start = block_pointers[block_idx];
    const auto row_end = block_pointers[block_idx + 1];  // exclusive
    const auto bsize = row_end - row_start;

    const auto row_dense_block = row_idx - row_start;

    const ValueType* __restrict__ dense_block_ptr =
        blocks_array +
        storage_scheme.get_global_block_offset(batch_idx, num_blocks, block_idx,
                                               blocks_cumulative_storage);

    ValueType* __restrict__ out_dense_block_ptr =
        out_blocks_array +
        storage_scheme.get_global_block_offset(batch_idx, num_blocks, block_idx,
                                               blocks_cumulative_storage);

    const auto stride = storage_scheme.get_stride(block_idx, block_pointers);

    for (int c = sg_tid; c < bsize; c += sg_size) {
        const auto val =
            to_conjugate == true
                ? conj(dense_block_ptr[row_dense_block * stride + c])
                : dense_block_ptr[row_dense_block * stride + c];
        out_dense_block_ptr[c * stride + row_dense_block] = val;
        // coalesced reads, non-coalesced writes
    }
}
