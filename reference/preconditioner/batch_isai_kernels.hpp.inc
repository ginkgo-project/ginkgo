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

template <typename Callback>
inline void match(const int aiA_row_st, const int aiA_row_end,
                  const int* const aiA_col_idxs, const int A_row_st,
                  const int A_row_end, const int* const A_col_idxs,
                  Callback do_this_on_match)
{
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
            do_this_on_match(ptr_aiA, ptr_A);
            ptr_aiA++;
            ptr_A++;
        }
    }
}


inline void extract_pattern_for_dense_sys_corr_to_current_row_impl(
    const int curr_row_idx, const int num_rows, const int* const A_row_ptrs,
    const int* const A_col_idxs, const int* const aiA_row_ptrs,
    const int* const aiA_col_idxs, int* const dense_mat_pattern,
    int* const rhs_one_idxs, int* const sizes,
    int* num_matches_per_row_for_each_csr_sys)
{
    using gko::preconditioner::batch_isai::row_size_limit;

    int* const dense_mat_pattern_ptr =
        dense_mat_pattern + curr_row_idx * row_size_limit * row_size_limit;
    int& size = sizes[curr_row_idx];
    int& rhs_one_loc = rhs_one_idxs[curr_row_idx];

    const int aiA_row_st = aiA_row_ptrs[curr_row_idx];
    const int aiA_row_end = aiA_row_ptrs[curr_row_idx + 1];

    size = aiA_row_end - aiA_row_st;

    for (int i = aiA_row_st; i < aiA_row_end; i++) {
        const int col = aiA_col_idxs[i];
        if (curr_row_idx == col) {
            rhs_one_loc = i - aiA_row_st;
        }

        const int A_row = col;
        const int A_row_st = A_row_ptrs[A_row];
        const int A_row_end = A_row_ptrs[A_row + 1];


        if (size <= row_size_limit) {
            // match row: curr_row_idx of aiA with row: A_row of A (both the
            // arrays are in sorted order)
            const int dense_sys_row = i - aiA_row_st;
            int* const dense_ptr_for_row =
                dense_mat_pattern_ptr + dense_sys_row * row_size_limit;
            // row-major order

            auto write_on_match = [dense_ptr_for_row, aiA_row_st](
                                      int idx_match_aiA_arr,
                                      int idx_match_A_arr) {
                dense_ptr_for_row[idx_match_aiA_arr - aiA_row_st] =
                    idx_match_A_arr;
            };

            match(aiA_row_st, aiA_row_end, aiA_col_idxs, A_row_st, A_row_end,
                  A_col_idxs, write_on_match);
        } else {
            // match row: curr_row_idx of aiA with row: A_row of A (both the
            // arrays are in sorted order)
            int& count = num_matches_per_row_for_each_csr_sys[i];
            count = 0;

            auto count_on_match = [&count](int idx_match_aiA_arr,
                                           int idx_match_A_arr) {
                count = count + 1;
            };

            match(aiA_row_st, aiA_row_end, aiA_col_idxs, A_row_st, A_row_end,
                  A_col_idxs, count_on_match);
        }
    }
}

template <typename ValueType>
inline void solve_lower_tri_system(const int size,
                                   const ValueType* const dense_mat,
                                   const int rhs_one_loc,
                                   ValueType* const sol_arr)
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
inline void solve_upper_tri_system(const int size,
                                   const ValueType* const dense_mat,
                                   const int rhs_one_loc,
                                   ValueType* const sol_arr)
{
    using gko::preconditioner::batch_isai::row_size_limit;

    for (int r = size - 1; r >= 0; r--) {
        const auto rhs =
            rhs_one_loc == r ? one<ValueType>() : zero<ValueType>();
        ValueType sum = zero<ValueType>();
        for (int c = size - 1; c > r; c--) {
            const auto val = dense_mat[r * row_size_limit + c];
            sum += val * sol_arr[c];
        }
        sol_arr[r] = (rhs - sum) / dense_mat[r * row_size_limit + r];
    }
}

template <typename ValueType>
inline int choose_pivot_row(const int diag_pos,
                            const ValueType* const dense_mat, const int size)
{
    using gko::preconditioner::batch_isai::row_size_limit;

    const int col_idx = diag_pos;
    int piv_row_idx = diag_pos;
    ValueType ele = dense_mat[piv_row_idx * row_size_limit + col_idx];
    for (int row_idx = diag_pos + 1; row_idx < size; row_idx++) {
        const ValueType curr_ele =
            dense_mat[row_idx * row_size_limit + col_idx];
        if (abs(curr_ele) > abs(ele)) {
            piv_row_idx = row_idx;
            ele = curr_ele;
        }
    }
    return piv_row_idx;
}

template <typename ValueType>
inline void swap_row_and_rhs(const int diag_pos, const int piv_row_idx,
                             const int size, ValueType* const dense_mat,
                             ValueType* const rhs_arr)
{
    using gko::preconditioner::batch_isai::row_size_limit;

    for (int col = 0; col < size; col++) {
        const ValueType val = dense_mat[diag_pos * row_size_limit + col];
        dense_mat[diag_pos * row_size_limit + col] =
            dense_mat[piv_row_idx * row_size_limit + col];
        dense_mat[piv_row_idx * row_size_limit + col] = val;
    }

    const ValueType rhs = rhs_arr[diag_pos];
    rhs_arr[diag_pos] = rhs_arr[piv_row_idx];
    rhs_arr[piv_row_idx] = rhs;
}

template <typename ValueType>
inline void row_transformation(const int diag_pos, const int row_idx,
                               const int size, ValueType* const dense_mat,
                               ValueType* const rhs_arr)
{
    using gko::preconditioner::batch_isai::row_size_limit;

    const ValueType diag_ele = dense_mat[diag_pos * row_size_limit + diag_pos];
    assert(diag_ele != zero<ValueType>());
    const ValueType multiplier =
        dense_mat[row_idx * row_size_limit + diag_pos] / diag_ele;
    const ValueType rhs_key_val = rhs_arr[diag_pos];

    for (int col = 0; col < size; col++) {
        const ValueType col_key_val =
            dense_mat[diag_pos * row_size_limit + col];
        dense_mat[row_idx * row_size_limit + col] -= multiplier * col_key_val;
    }

    rhs_arr[row_idx] -= multiplier * rhs_key_val;
}

template <typename ValueType>
inline void solve_gen_square_system(const int size, ValueType* const dense_mat,
                                    const int rhs_one_loc,
                                    ValueType* const sol_arr)
{
    using gko::preconditioner::batch_isai::row_size_limit;

    std::vector<ValueType> rhs_arr(size, 0);
    rhs_arr[rhs_one_loc] = one<ValueType>();

    for (int diag_pos = 0; diag_pos < size; diag_pos++) {
        const int piv_row_idx = choose_pivot_row(diag_pos, dense_mat, size);

        if (diag_pos != piv_row_idx) {
            swap_row_and_rhs(diag_pos, piv_row_idx, size, dense_mat,
                             rhs_arr.data());
        }

        for (int row_idx = 0; row_idx < diag_pos; row_idx++) {
            row_transformation(diag_pos, row_idx, size, dense_mat,
                               rhs_arr.data());
        }

        for (int row_idx = diag_pos + 1; row_idx < size; row_idx++) {
            row_transformation(diag_pos, row_idx, size, dense_mat,
                               rhs_arr.data());
        }
    }
    // system in reduced row echelon form (rref)

    for (int i = 0; i < size; i++) {
        sol_arr[i] = rhs_arr[i] / dense_mat[i * row_size_limit + i];
    }
}


template <typename ValueType>
inline void fill_values_dense_mat_and_solve_batch_entry_impl(
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

        if (size > row_size_limit) {
            continue;
        }
        const int rhs_one_loc = rhs_one_idxs[row_idx];
        const int* const dense_mat_pattern_ptr =
            dense_mat_pattern + row_idx * row_size_limit * row_size_limit;

        std::array<ValueType, row_size_limit * row_size_limit> dense_mat;
        dense_mat.fill(0);

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


template <typename ValueType>
inline void match_for_csr(
    const int* const inv_col_idxs, const int inv_row_st, const int inv_row_end,
    const int* const sys_col_idxs, const int sys_row_st, const int sys_row_end,
    const int csr_pattern_row_start, int* const csr_pattern_col_idxs,
    gko::remove_complex<ValueType>* const csr_pattern_values)
{
    int l = inv_row_st;
    int m = sys_row_st;
    int j = csr_pattern_row_start;

    while (l < inv_row_end && m < sys_row_end) {
        if (inv_col_idxs[l] == sys_col_idxs[m]) {
            csr_pattern_values[j] =
                static_cast<gko::remove_complex<ValueType>>(m);
            csr_pattern_col_idxs[j] = l - inv_row_st;
            j++;

            l++;
            m++;
        } else if (inv_col_idxs[l] > sys_col_idxs[m]) {
            m++;
        } else {
            l++;
        }
    }
}


template <typename ValueType>
void extract_csr_row_impl(
    const int lin_sys_row, const int inv_nz, const int* const inv_row_ptrs,
    const int* const inv_col_idxs, const int* const sys_row_ptrs,
    const int* const sys_col_idxs, const int* const csr_pattern_row_ptrs,
    int* const csr_pattern_col_idxs,
    gko::remove_complex<ValueType>* const csr_pattern_values)
{
    const int col_idx = inv_col_idxs[inv_nz];
    const int sys_row = col_idx;
    const int sys_row_st = sys_row_ptrs[sys_row];
    const int sys_row_end = sys_row_ptrs[sys_row + 1];

    const int inv_row_st = inv_row_ptrs[lin_sys_row];
    const int inv_row_end = inv_row_ptrs[lin_sys_row + 1];

    const int csr_pattern_row_start = csr_pattern_row_ptrs[inv_nz - inv_row_st];

    match_for_csr<ValueType>(inv_col_idxs, inv_row_st, inv_row_end,
                             sys_col_idxs, sys_row_st, sys_row_end,
                             csr_pattern_row_start, csr_pattern_col_idxs,
                             csr_pattern_values);
}
