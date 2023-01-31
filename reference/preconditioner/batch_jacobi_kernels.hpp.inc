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

template <typename ValueType>
inline void extract_block_pattern_impl(
    const size_type k, const matrix::Csr<ValueType, int>* const first_sys_csr,
    const preconditioner::batched_blocks_storage_scheme& storage_scheme,
    const int* const block_pointers, int* const blocks_pattern)
{
    const int* row_ptrs = first_sys_csr->get_const_row_ptrs();
    const int* col_idxs = first_sys_csr->get_const_col_idxs();

    const int row_idx_st = block_pointers[k];
    const int row_idx_end = block_pointers[k + 1];  // exclusive

    // pattern is stored in row-major order
    for (int row_idx = row_idx_st; row_idx < row_idx_end; row_idx++) {
        for (int i = row_ptrs[row_idx]; i < row_ptrs[row_idx + 1]; i++) {
            const auto col_idx = col_idxs[i];

            if (col_idx >= row_idx_st && col_idx < row_idx_end) {
                // (row_idx, col_idx) is the part of the diagonal block
                // storing the position/index of this element at place: (row_idx
                // - row_idx_st, col_idx - row_idx_st)
                const auto dense_block_row = row_idx - row_idx_st;
                const auto dense_block_col = col_idx - row_idx_st;
                const auto offset = storage_scheme.get_block_offset(k);
                auto& place = blocks_pattern[offset +
                                             dense_block_row *
                                                 storage_scheme.get_stride() +
                                             dense_block_col];  // data-locality
                place = i;
            }
        }
    }
}


template <typename T>
inline void swap_numbers(T& a, T& b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

template <typename ValueType>
inline void swap_rows(const int size, std::vector<ValueType>& mat,
                      const int row1, const int row2)
{
    for (int c = 0; c < size; c++) {
        swap_numbers(mat[row1 * size + c], mat[row2 * size + c]);
    }
}

template <typename ValueType>
inline int choose_pivot(const int k, const int block_size,
                        const std::vector<ValueType>& temp_dense_block)
{
    int piv = k;
    ValueType piv_val = temp_dense_block[k * block_size + k];
    for (int r = k + 1; r < block_size; r++)  // uncoalesced accesses
    {
        ValueType curr_val = temp_dense_block[r * block_size + k];
        if (abs(piv_val) < abs(curr_val)) {
            piv = r;
            piv_val = curr_val;
        }
    }
    return piv;
}

template <typename ValueType>
inline void invert_dense_block(const int block_size,
                               std::vector<ValueType>& temp_dense_block,
                               std::vector<int>& col_perm)
{
    // the block is stored in the row-major order, with row stride = block_size

    // Gauss-Jordan elimination with pivoting

    for (int k = 0; k < block_size; k++) {
        const int piv = choose_pivot(k, block_size, temp_dense_block);
        if (k != piv) {
            swap_rows(block_size, temp_dense_block, k, piv);
            swap_numbers(col_perm[piv], col_perm[k]);
        }


        const ValueType d = temp_dense_block[k * block_size + k];

        // SCAL (scaling the kth col)
        for (int r = 0; r < block_size; r++) {
            temp_dense_block[r * block_size + k] /= -d;
        }  // uncoalesced accesses
        temp_dense_block[k * block_size + k] = zero<ValueType>();

        // GER (Note: the kth row and kth col do not change)
        for (int r = 0; r < block_size; r++) {
            for (int c = 0; c < block_size; c++) {
                temp_dense_block[r * block_size + c] +=
                    temp_dense_block[r * block_size + k] *
                    temp_dense_block[k * block_size + c];
            }
        }  // data-locality

        // SCAL (scaling the kth row)
        for (int c = 0; c < block_size; c++) {
            temp_dense_block[k * block_size + c] /= d;
        }  // data-locality
        temp_dense_block[k * block_size + k] = one<ValueType>() / d;
    }
}

template <typename ValueType>
inline void compute_block_jacobi_impl(
    const size_type batch_idx, const size_type block_idx,
    const batch_csr::BatchEntry<const ValueType>& A_entry,
    const size_type num_blocks,
    const preconditioner::batched_blocks_storage_scheme& storage_scheme,
    const int* const block_pointers, const int* const blocks_pattern,
    ValueType* const blocks)
{
    const auto offset_batch =
        storage_scheme.get_batch_offset(num_blocks, batch_idx);
    const auto offset_indiv = storage_scheme.get_block_offset(block_idx);
    ValueType* dense_block_ptr =
        blocks + offset_batch +
        offset_indiv;  // the blocks are in row-major order

    const auto bsize =
        block_pointers[block_idx + 1] - block_pointers[block_idx];

    std::vector<ValueType> temp_dense_block(bsize *
                                            bsize);  // also in row-major order
    std::vector<int> col_perm(bsize);
    for (int i = 0; i < bsize; i++) {
        col_perm[i] = i;
    }

    for (int row = 0; row < bsize; row++) {
        for (int col = 0; col < bsize; col++) {
            auto to_write = zero<ValueType>();
            const auto idx = blocks_pattern[offset_indiv +
                                            row * storage_scheme.get_stride() +
                                            col];  // data locality
            if (idx >= 0) {
                assert(idx < A_entry.num_nnz);
                to_write = A_entry.values[idx];
            }
            temp_dense_block[row * bsize + col] = to_write;  // data locality
        }
    }

    // invert the diagonal block
    invert_dense_block(bsize, temp_dense_block, col_perm);

    // store the dense block (take into account the col permutations)
    for (int row = 0; row < bsize; row++) {
        for (int col = 0; col < bsize; col++) {
            dense_block_ptr[row * storage_scheme.get_stride() + col_perm[col]] =
                temp_dense_block[row * bsize + col];
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
    const auto rub = gko::kernels::host::get_batch_struct(r);
    const auto zub = gko::kernels::host::get_batch_struct(z);

    using scalar_jacobi_prec = gko::kernels::host::BatchScalarJacobi<ValueType>;
    using block_jacobi_prec = gko::kernels::host::BatchBlockJacobi<ValueType>;

    if (max_block_size == 1) {
        scalar_jacobi_prec prec;
        apply_jacobi(sys_mat_batch, prec, rub, zub);
    } else {
        block_jacobi_prec prec(max_block_size, num_blocks, storage_scheme,
                               blocks_array, block_ptrs,
                               row_part_of_which_block_info);
        apply_jacobi(sys_mat_batch, prec, rub, zub);
    }
}

template <typename ValueType>
void tranpose_dense_block_impl(
    const size_type batch_id, const size_type block_id,
    const size_type num_blocks, const int* const block_pointers,
    const ValueType* const blocks_array,
    const gko::preconditioner::batched_blocks_storage_scheme& storage_scheme,
    ValueType* const out_blocks_array, const bool to_conjugate)
{
    const ValueType* dense_block_ptr =
        blocks_array +
        storage_scheme.get_global_block_offset(num_blocks, batch_id, block_id);
    ValueType* out_dense_block_ptr =
        out_blocks_array +
        storage_scheme.get_global_block_offset(num_blocks, batch_id, block_id);

    const auto bsize = block_pointers[block_id + 1] - block_pointers[block_id];
    const auto stride = storage_scheme.get_stride();

    for (int r = 0; r < bsize; r++) {
        for (int c = 0; c < bsize; c++) {
            out_dense_block_ptr[c * stride + r] =
                to_conjugate == true ? conj(dense_block_ptr[r * stride + c])
                                     : dense_block_ptr[r * stride + c];
        }
    }
}