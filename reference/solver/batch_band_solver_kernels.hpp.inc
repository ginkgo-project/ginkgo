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

namespace unblocked_LU_band_factorization {

template <typename ValueType>
inline void set_possible_fill_in_U_part_zero(const int N, const int KL,
                                             const int KU,
                                             ValueType* const AB_array_entry)
{
    const int KV = KL + KU;
    const int AB_nrows = 2 * KL + KU + 1;

    // set all fill-in elements: zero
    // col: 0,1,2,..., KU : no fill-in possible in U part
    // for(int AB_col_idx = KU + 1; AB_col_idx <= std::min(KV, N-1);
    // AB_col_idx++)
    // {
    //     for(int AB_row_idx = KV - AB_col_idx; AB_row_idx <= KL - 1;
    //     AB_row_idx++)
    //     {
    //         AB_array_entry[AB_row_idx +  AB_nrows * AB_col_idx] =
    //         zero<ValueType>();
    //     }
    // }

    // for(int AB_col_idx = KV + 1; AB_col_idx <= N-1; AB_col_idx++)
    // {
    //     for(int AB_row_idx = 0; AB_row_idx <= KL - 1; AB_row_idx++)
    //     {
    //         AB_array_entry[AB_row_idx + AB_nrows * AB_col_idx] =
    //         zero<ValueType>();
    //     }
    // }

    for (int AB_col_idx = KU + 1; AB_col_idx <= N - 1; AB_col_idx++) {
        for (int AB_row_idx = KL - 1;
             AB_row_idx >= std::max(0, KV - AB_col_idx); AB_row_idx--) {
            AB_array_entry[AB_row_idx + AB_nrows * AB_col_idx] =
                zero<ValueType>();
        }
    }
}

template <typename ValueType>
inline int find_pivot_row_index(const int N, const int KL, const int KU,
                                const ValueType* const AB_array_entry,
                                const int AB_col_idx)
{
    const int AB_nrows = 2 * KL + KU + 1;
    const int KV = KU + KL;
    int AB_piv_row_idx = KV;
    const int AB_row_idx_limit = KL + KU + (N - AB_col_idx - 1);
    ValueType piv_val = AB_array_entry[AB_piv_row_idx + AB_nrows * AB_col_idx];
    for (int AB_row_idx = KV; AB_row_idx <= std::min(AB_row_idx_limit, KV + KL);
         AB_row_idx++) {
        const auto val = AB_array_entry[AB_row_idx + AB_nrows * AB_col_idx];
        if (abs(piv_val) < abs(val)) {
            piv_val = val;
            AB_piv_row_idx = AB_row_idx;
        }
    }

    return AB_piv_row_idx;
}

template <typename ValueType>
inline void swap_rows(const int N, const int KL, const int KU,
                      ValueType* const AB_array_entry,
                      const int A_affected_col_idx_last,
                      const int A_diag_row_idx, const int A_pivot_row_idx)
{
    const int AB_nrows = 2 * KL + KU + 1;
    for (int A_col_idx = A_diag_row_idx; A_col_idx <= A_affected_col_idx_last;
         A_col_idx++) {
        const int AB_row_idx_1 = KL + KU + A_diag_row_idx - A_col_idx;
        const int AB_row_idx_2 = KL + KU + A_pivot_row_idx - A_col_idx;
        const ValueType temp =
            AB_array_entry[AB_row_idx_1 + AB_nrows * A_col_idx];
        AB_array_entry[AB_row_idx_1 + AB_nrows * A_col_idx] =
            AB_array_entry[AB_row_idx_2 + AB_nrows * A_col_idx];
        AB_array_entry[AB_row_idx_2 + AB_nrows * A_col_idx] = temp;
    }
}

template <typename ValueType>
inline void scale_column(const int N, const int KL, const int KU,
                         ValueType* const AB_array_entry, const int AB_col_idx)
{
    const int AB_nrows = 2 * KL + KU + 1;
    const int AB_diag_row_idx = KL + KU;

    const int AB_row_idx_limit = KL + KU + (N - 1 - AB_col_idx);
    const ValueType diag_ele =
        AB_array_entry[AB_diag_row_idx + AB_nrows * AB_col_idx];
    for (int AB_row_idx = AB_diag_row_idx + 1;
         AB_row_idx <= std::min(AB_row_idx_limit, AB_diag_row_idx + KL);
         AB_row_idx++) {
        AB_array_entry[AB_row_idx + AB_nrows * AB_col_idx] /= diag_ele;
    }
}

template <typename ValueType>
inline void update_trailing_submatrix(const int N, const int KL, const int KU,
                                      ValueType* const AB_array_entry,
                                      const int AB_col_idx,
                                      const int A_diag_row_idx,
                                      const int A_affected_col_idx_last)
{
    const int AB_nrows = 2 * KL + KU + 1;
    for (int A_col_idx = A_diag_row_idx + 1;
         A_col_idx <= A_affected_col_idx_last; A_col_idx++) {
        const int AB_main_row = KL + KU + A_diag_row_idx - A_col_idx;
        const int AB_col = A_col_idx;
        const ValueType ele_above =
            AB_array_entry[AB_main_row + AB_nrows * AB_col];

        for (int A_row_idx = A_diag_row_idx + 1;
             A_row_idx <= std::min(N - 1, A_diag_row_idx + KL); A_row_idx++) {
            const int AB_row_idx_for_side_ele =
                KL + KU + A_row_idx - AB_col_idx;
            const ValueType ele_side =
                AB_array_entry[AB_row_idx_for_side_ele + AB_nrows * AB_col_idx];

            const int AB_row_idx_for_this_col_ele =
                KL + KU + A_row_idx - A_col_idx;
            AB_array_entry[AB_row_idx_for_this_col_ele +
                           AB_nrows * A_col_idx] -= ele_above * ele_side;
        }
    }
}


template <typename ValueType>
inline void factorize_band(const int N, const int KL, const int KU,
                           ValueType* const AB_array_entry,
                           int* const ipiv_entry)
{
    const int KV = KL + KU;

    set_possible_fill_in_U_part_zero(N, KL, KU, AB_array_entry);

    // think of A as mat with KL subdiags, KU super diags (actually KL + KU
    // superdiags, the outer KL are zero...
    // --> this way after factorizn no fill-in... )

    int A_affected_col_idx_last = 0;

    for (int AB_col_idx = 0; AB_col_idx <= N - 1; AB_col_idx++) {
        const int AB_diag_row_idx = KV;
        const int AB_pivot_row_idx =
            find_pivot_row_index(N, KL, KU, AB_array_entry, AB_col_idx);
        const int A_pivot_row_idx = AB_pivot_row_idx - KV + AB_col_idx;
        const int A_diag_row_idx = AB_col_idx;

        // assert piv not equal to zero
        assert(AB_array_entry[AB_pivot_row_idx + AB_nrows * AB_col_idx] !=
               zero<ValueType>());

        // THINK...: reference- LAPACK
        A_affected_col_idx_last = std::max(
            A_affected_col_idx_last, std::min(A_pivot_row_idx + KU, N - 1));

        // swap rows
        if (AB_diag_row_idx != AB_pivot_row_idx) {
            swap_rows(N, KL, KU, AB_array_entry, A_affected_col_idx_last,
                      A_diag_row_idx, A_pivot_row_idx);
        }

        // record the swap row info...
        ipiv_entry[AB_col_idx] = A_pivot_row_idx;

        // scale A's column
        scale_column(N, KL, KU, AB_array_entry, AB_col_idx);

        // update trailing submatrix
        update_trailing_submatrix(N, KL, KU, AB_array_entry, AB_col_idx,
                                  A_diag_row_idx, A_affected_col_idx_last);
    }
}

}  // namespace unblocked_LU_band_factorization

namespace blocked_LU_band_factorization {


}  // namespace blocked_LU_band_factorization


namespace unblocked_and_blocked_band_solver_common {

template <typename ValueType>
void accommodate_partial_row_swaps_lower_trsv(
    const int N, const int KL, const int KU, const int* const ipiv_entry,
    const ValueType* const AB_array_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& b_entry,
    const gko::batch_dense::BatchEntry<ValueType>& temp_entry)
{
    // initialize temp with b
    copy(b_entry, temp_entry);

    const int AB_nrows = 2 * KL + KU + 1;

    for (int rhs_idx = 0; rhs_idx < b_entry.num_rhs; rhs_idx++) {
        for (int j = 0; j <= N - 2; j++) {
            // swap temp[j] and temp[ipiv[j]]
            const ValueType tmp =
                temp_entry.values[j + temp_entry.stride * rhs_idx];
            temp_entry.values[j + temp_entry.stride * rhs_idx] =
                temp_entry.values[ipiv_entry[j]];
            temp_entry.values[ipiv_entry[j] + temp_entry.stride * rhs_idx] =
                tmp;

            // Now temp[j] holds the correct value.
            const int A_col_idx = j;
            const ValueType temp_at_row_j =
                temp_entry.values[A_col_idx + temp_entry.stride * rhs_idx];

            for (int A_row_idx = j + 1; A_row_idx <= std::min(N - 1, j + KL);
                 A_row_idx++) {
                // temp[A_row_idx] += -1 * A(A_row_idx, A_col_idx) *
                // temp[A_col_idx]

                const int AB_row_idx = KL + KU + A_row_idx - A_col_idx;

                temp_entry.values[A_row_idx + temp_entry.stride * rhs_idx] +=
                    -one<ValueType>() *
                    AB_array_entry[AB_row_idx + A_col_idx * AB_nrows] *
                    temp_at_row_j;
            }
        }
    }
}

template <typename ValueType>
void upper_trsv(const int N, const int KL, const int KU,
                const ValueType* const AB_array_entry,
                const gko::batch_dense::BatchEntry<const ValueType>& y_entry,
                const gko::batch_dense::BatchEntry<ValueType>& x_entry)
{
    const int AB_nrows = 2 * KL + KU + 1;
    const int KV = KL + KU;

    for (int rhs_idx = 0; rhs_idx < x_entry.num_rhs; rhs_idx++) {
        for (int A_row_idx = N - 1; A_row_idx >= 0; A_row_idx--) {
            const int A_col_idx_last = std::min(A_row_idx + KL + KU, N - 1);
            ValueType sum = zero<ValueType>();
            for (int A_col_idx = A_col_idx_last; A_col_idx > A_row_idx;
                 A_col_idx--) {
                // sum += A(A_row_idx, A_col_idx) * x(A_col_idx)
                const int AB_corr_row_idx = KL + KU + A_row_idx - A_col_idx;
                sum += AB_array_entry[AB_corr_row_idx + A_col_idx * AB_nrows] *
                       x_entry.values[A_col_idx + x_entry.stride * rhs_idx];
            }

            // x(A_row_idx) = (y(A_row_idx) - sum)/A(A_row_idx, A_row_idx);
            const int AB_row_idx_diag_ele = KL + KU;
            const ValueType diag_ele =
                AB_array_entry[AB_row_idx_diag_ele + A_row_idx * AB_nrows];
            x_entry.values[A_row_idx + x_entry.stride * rhs_idx] =
                (y_entry.values[A_row_idx + y_entry.stride * rhs_idx] - sum) /
                diag_ele;
        }
    }
}

}  // namespace unblocked_and_blocked_band_solver_common

template <typename ValueType>
inline void batch_entry_band_unblocked_solve_impl(
    const gko::size_type ibatch, const gko::size_type nbatch, const int KL,
    const int KU, ValueType* const batch_band_mat_array,
    const gko::batch_dense::UniformBatch<const ValueType>& b_batch,
    const gko::batch_dense::UniformBatch<ValueType>& x_batch,
    unsigned char* const local_space)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const int N = b_batch.num_rows;
    const int nrhs = b_batch.num_rhs;

    const gko::batch_dense::BatchEntry<const ValueType> b_entry =
        gko::batch::batch_entry(b_batch, ibatch);

    const gko::batch_dense::BatchEntry<ValueType> x_entry =
        gko::batch::batch_entry(x_batch, ibatch);

    unsigned char* const shared_space = local_space;
    ValueType* const temp = reinterpret_cast<ValueType*>(shared_space);
    const gko::batch_dense::BatchEntry<ValueType> temp_entry{
        temp, static_cast<size_type>(nrhs), N, nrhs};

    int* const ipiv_entry = reinterpret_cast<int*>(temp + nrhs * N);

    const int AB_nrows = 2 * KL + KU + 1;
    const int AB_ncols = N;

    ValueType* const AB_array_entry =
        batch_band_mat_array + ibatch * AB_nrows * AB_ncols;

    unblocked_LU_band_factorization::factorize_band(N, KL, KU, AB_array_entry,
                                                    ipiv_entry);

    unblocked_and_blocked_band_solver_common::
        accommodate_partial_row_swaps_lower_trsv(
            N, KL, KU, ipiv_entry, AB_array_entry, b_entry, temp_entry);

    unblocked_and_blocked_band_solver_common::upper_trsv(
        N, KL, KU, AB_array_entry, gko::batch::to_const(temp_entry), x_entry);
}


template <typename ValueType>
inline void batch_entry_band_blocked_solve_impl(
    const gko::size_type ibatch, const gko::size_type nbatch, const int KL,
    const int KU, ValueType* const batch_band_mat_array,
    const int blocked_solve_panel_size,
    const gko::batch_dense::UniformBatch<const ValueType>& b_batch,
    const gko::batch_dense::UniformBatch<ValueType>& x_batch,
    unsigned char* const local_space)
{
    GKO_NOT_IMPLEMENTED;
}
