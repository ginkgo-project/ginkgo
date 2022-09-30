/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


void match_and_write(const int size, const int dense_sys_row,
                     const int aiA_row_st, const int aiA_row_end,
                     const int* const aiA_col_idxs, const int A_row_st,
                     const int A_row_end, const int* const A_col_idxs,
                     int* const dense_mat_pattern_ptr)
{
    using gko::preconditioner::batch_isai::row_size_limit;
    int* const dense_mat_row_ptr =
        dense_mat_pattern_ptr + dense_sys_row * row_size_limit;
    // row-major order

    int ptr_aiA = aiA_row_st;
    int ptr_A = A_row_st;

    while (ptr_aiA < aiA_row_end && ptr_A < A_row_end) {
        const int val_aiA = aiA_col_idxs[ptr_aiA];
        const int val_A = A_col_idxs[ptr_A];

        if (val_aiA < val_A) {
            ptr_aiA++;
        } else if (val_aiA > val_A) {
            ptr_A++;
        } else  // val_aiA == val_A
        {
            dense_mat_row_ptr[ptr_aiA - aiA_row_st] = ptr_A;
            ptr_aiA++;
            ptr_A++;
        }
    }
}


void extract_pattern_for_dense_sys_corr_to_current_row_impl(
    const int curr_row_idx, const int num_rows, const int* const A_row_ptrs,
    const int* const A_col_idxs, const int* const aiA_row_ptrs,
    const int* const aiA_col_idxs, int* const dense_mat_pattern,
    int* const rhs_one_idxs, int* const sizes)
{
    using gko::preconditioner::batch_isai::row_size_limit;

    int* const dense_mat_pattern_ptr =
        dense_mat_pattern + curr_row_idx * row_size_limit * row_size_limit;
    int& size = sizes[curr_row_idx];
    int& rhs_one_loc = rhs_one_idxs[curr_row_idx];

    const int aiA_row_st = aiA_row_ptrs[curr_row_idx];
    const int aiA_row_end = aiA_row_ptrs[curr_row_idx + 1];

    size = aiA_row_end - aiA_row_st;
    assert(size <= gko::preconditioner::batch_isai::row_size_limit);

    for (int i = aiA_row_st; i < aiA_row_end; i++) {
        const int col = aiA_col_idxs[i];
        if (curr_row_idx == col) {
            rhs_one_loc = i - aiA_row_st;
        }

        const int A_row = col;
        const int A_row_st = A_row_ptrs[A_row];
        const int A_row_end = A_row_ptrs[A_row + 1];

        // match row: curr_row_idx of aiA with row: A_row of A (both the arrays
        // are in sorted order)
        match_and_write(size, i - aiA_row_st, aiA_row_st, aiA_row_end,
                        aiA_col_idxs, A_row_st, A_row_end, A_col_idxs,
                        dense_mat_pattern_ptr);
    }
}

template <typename ValueType>
void solve_lower_tri_system(const int size, const ValueType* const dense_mat,
                            const int rhs_one_loc, ValueType* const sol_arr)
{
    using gko::preconditioner::batch_isai::row_size_limit;

    for (int r = 0; r < size; r++) {
        const auto rhs =
            rhs_one_loc == r ? one<ValueType>() : zero<ValueType>();
        ValueType sum = zero<ValueType>();
        for (int c = 0; c < r; c++) {
            const auto val = dense_mat[r * row_size_limit + c];
            sum += val * sol_arr[c];
        }
        sol_arr[r] = (rhs - sum) / dense_mat[r * row_size_limit + r];
    }
}


template <typename ValueType>
void solve_upper_tri_system(const int size, const ValueType* const dense_mat,
                            const int rhs_one_loc, ValueType* const sol_arr)
{
    using gko::preconditioner::batch_isai::row_size_limit;

    for (int r = size - 1; r >= 0; r++) {
        const auto rhs =
            rhs_one_loc == r ? one<ValueType>() : zero<ValueType>();
        ValueType sum = zero<ValueType>();
        for (int c = size - 1; c > r; c++) {
            const auto val = dense_mat[r * row_size_limit + c];
            sum += val * sol_arr[c];
        }
        sol_arr[r] = (rhs - sum) / dense_mat[r * row_size_limit + r];
    }
}


template <typename ValueType>
void solve_gen_square_system(const int size, const ValueType* const dense_mat,
                             const int rhs_one_loc, ValueType* const sol_arr)
{
    using gko::preconditioner::batch_isai::row_size_limit;

    GKO_NOT_IMPLEMENTED;
}


template <typename ValueType>
void fill_values_dense_mat_and_solve_batch_entry_impl(
    const batch_csr::BatchEntry<const ValueType>& A_entry,
    const batch_csr::BatchEntry<ValueType>& aiA_entry,
    const int* const dense_mat_pattern, const int* const rhs_one_idxs,
    const int* const sizes,
    const gko::preconditioner::batch_isai_input_matrix_type&
        input_matrix_type_isai)
{
    using gko::preconditioner::batch_isai::row_size_limit;

    for (int row_idx = 0; row_idx < aiA_entry.num_rows; row_idx++) {
        const int size = sizes[row_idx];
        const int rhs_one_loc = rhs_one_idxs[row_idx];
        const int* const dense_mat_pattern_ptr =
            dense_mat_pattern + row_idx * row_size_limit * row_size_limit;

        std::vector<ValueType> dense_mat(row_size_limit * row_size_limit, 0);

        // use the stored pattern to fill the dense matrix
        for (int r = 0; r < size; r++) {
            for (int c = 0; c < size; c++) {
                const auto A_idx =
                    dense_mat_pattern_ptr[r * row_size_limit + c];
                if (A_idx >= 0) {
                    // write at c,r (implcit transpose) (non-coalseced accesses)
                    // row-major order
                    dense_mat[c * row_size_limit + r] = A_entry.values[A_idx];
                }
            }
        }

        ValueType* const sol_arr =
            &aiA_entry.values[aiA_entry.row_ptrs[row_idx]];
        // solve the dense system
        if (input_matrix_type_isai ==
            gko::preconditioner::batch_isai_input_matrix_type::lower_tri) {
            solve_upper_tri_system(size, dense_mat.data(), rhs_one_loc,
                                   sol_arr);
        } else if (input_matrix_type_isai ==
                   gko::preconditioner::batch_isai_input_matrix_type::
                       upper_tri) {
            solve_lower_tri_system(size, dense_mat.data(), rhs_one_loc,
                                   sol_arr);
        } else if (input_matrix_type_isai ==
                   gko::preconditioner::batch_isai_input_matrix_type::general) {
            solve_gen_square_system(size, dense_mat.data(), rhs_one_loc,
                                    sol_arr);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }
}
