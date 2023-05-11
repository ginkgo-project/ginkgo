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

namespace common_band_factorization {

template <typename ValueType>
inline void set_possible_fill_in_U_part_zero(const int N, const int KL,
                                             const int KU,
                                             ValueType* const AB_array_entry)
{
    const int KV = KL + KU;
    const int AB_nrows = 2 * KL + KU + 1;

    // set all fill-in elements: zero
    // col: 0,1,2,..., KU : no fill-in possible in U part

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
                      const int A_col_idx_start_inclusive,
                      const int A_col_idx_end_inclusive, const int A_row_idx_1,
                      const int A_row_idx_2)
{
    const int AB_nrows = 2 * KL + KU + 1;
    for (int A_col_idx = A_col_idx_start_inclusive;
         A_col_idx <= A_col_idx_end_inclusive; A_col_idx++) {
        const int AB_row_idx_1 = KL + KU + A_row_idx_1 - A_col_idx;
        const int AB_row_idx_2 = KL + KU + A_row_idx_2 - A_col_idx;
        const ValueType temp =
            AB_array_entry[AB_row_idx_1 + AB_nrows * A_col_idx];
        AB_array_entry[AB_row_idx_1 + AB_nrows * A_col_idx] =
            AB_array_entry[AB_row_idx_2 + AB_nrows * A_col_idx];
        AB_array_entry[AB_row_idx_2 + AB_nrows * A_col_idx] = temp;
    }
}

template <typename ValueType>
inline void scale_column(const int N, const int KL, const int KU,
                         ValueType* const AB_array_entry, const int AB_col_idx,
                         ValueType diag_ele)
{
    const int AB_nrows = 2 * KL + KU + 1;
    const int AB_diag_row_idx = KL + KU;

    const int AB_row_idx_limit = KL + KU + (N - 1 - AB_col_idx);

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
                                      const int A_col_idx_end_inclusive)
{
    const int AB_nrows = 2 * KL + KU + 1;
    const int A_diag_row_idx = AB_col_idx;

    for (int A_col_idx = A_diag_row_idx + 1;
         A_col_idx <= A_col_idx_end_inclusive; A_col_idx++) {
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

}  // namespace common_band_factorization

namespace unblocked_LU_band_factorization {

template <typename ValueType>
inline void factorize_band(const int N, const int KL, const int KU,
                           ValueType* const AB_array_entry,
                           int* const ipiv_entry)
{
    const int AB_nrows = 2 * KL + KU + 1;
    const int KV = KL + KU;

    common_band_factorization::set_possible_fill_in_U_part_zero(N, KL, KU,
                                                                AB_array_entry);

    // think of A as mat with KL subdiags, KU super diags (actually KL + KU
    // superdiags, the outer KL are zero...
    // Take care not to access any location outside the stored band

    /*
    Implementing partial row-pivoting is tricky
    because of the fill-in generated by row-interchanges:

    > Extra KL super-diagonals to accommodate fill-in in the upper-triangular

    > The part of the row to the left of the diagonal element is not swapped as
    there is no extra space in the lower triangular part to store the fill-in;
    it is dealt with directly during the lower triangular solve phase
    */

    int A_affected_col_idx_last = 0;

    for (int AB_col_idx = 0; AB_col_idx <= N - 1; AB_col_idx++) {
        const int AB_diag_row_idx = KV;
        const int AB_pivot_row_idx =
            common_band_factorization::find_pivot_row_index(
                N, KL, KU, AB_array_entry, AB_col_idx);
        const int A_pivot_row_idx = AB_pivot_row_idx - KV + AB_col_idx;
        const int A_diag_row_idx = AB_col_idx;

        // assert piv not equal to zero
        const ValueType piv_ele =
            AB_array_entry[AB_pivot_row_idx + AB_nrows * AB_col_idx];
        assert(piv_ele != zero<ValueType>());

        A_affected_col_idx_last = std::max(
            A_affected_col_idx_last, std::min(A_pivot_row_idx + KU, N - 1));

        // swap rows
        if (AB_diag_row_idx != AB_pivot_row_idx) {
            common_band_factorization::swap_rows(
                N, KL, KU, AB_array_entry, AB_col_idx, A_affected_col_idx_last,
                A_diag_row_idx, A_pivot_row_idx);
        }

        // record the swap row info...
        ipiv_entry[AB_col_idx] = A_pivot_row_idx;

        // scale A's column
        common_band_factorization::scale_column(N, KL, KU, AB_array_entry,
                                                AB_col_idx, piv_ele);

        // update trailing submatrix
        common_band_factorization::update_trailing_submatrix(
            N, KL, KU, AB_array_entry, AB_col_idx, A_affected_col_idx_last);
    }
}

}  // namespace unblocked_LU_band_factorization

namespace blocked_LU_band_factorization {

template <typename ValueType>
inline void apply_row_interchanges_A01_A11_A21(
    const int N, const int KL, const int KU, ValueType* const AB_array_entry,
    const int j, const int JB, const int J1, const int* ipiv_entry)
{
    const int AB_nrows = 2 * KL + KU + 1;

    for (int a = j; a < j + JB; a++) {
        const int A_row_1 = a;
        const int A_row_2 = ipiv_entry[a];

        if (A_row_1 != A_row_2) {
            for (int col_idx = j + JB; col_idx < j + JB + J1; col_idx++) {
                const int AB_row_1 = KL + KU + A_row_1 - col_idx;
                const int AB_row_2 = KL + KU + A_row_2 - col_idx;
                const ValueType temp =
                    AB_array_entry[AB_row_1 + AB_nrows * col_idx];
                AB_array_entry[AB_row_1 + AB_nrows * col_idx] =
                    AB_array_entry[AB_row_2 + AB_nrows * col_idx];
                AB_array_entry[AB_row_2 + AB_nrows * col_idx] = temp;
            }
        }
    }
}

template <typename ValueType>
inline void apply_row_interchanges_A02_A12_A22(const int N, const int KL,
                                               const int KU,
                                               ValueType* const AB_array_entry,
                                               const int j, const int JB,
                                               const int J1, const int J2,
                                               const int* ipiv_entry)
{
    const int AB_nrows = 2 * KL + KU + 1;

    // apply row interchanges to A02, A12, and A22 columnwise
    for (int A_col = j + JB + J1; A_col < j + JB + J1 + J2; A_col++) {
        for (int a = std::max(j, A_col - KL - KU); a < j + JB; a++) {
            const int A_row_1 = a;
            const int A_row_2 = ipiv_entry[a];

            if (A_row_1 != A_row_2) {
                const int AB_row_1 = KL + KU + A_row_1 - A_col;
                const int AB_row_2 = KL + KU + A_row_2 - A_col;
                const ValueType temp =
                    AB_array_entry[AB_row_1 + AB_nrows * A_col];
                AB_array_entry[AB_row_1 + AB_nrows * A_col] =
                    AB_array_entry[AB_row_2 + AB_nrows * A_col];
                AB_array_entry[AB_row_2 + AB_nrows * A_col] = temp;
            }
        }
    }
}

template <typename ValueType>
inline void lower_trsv_update_A01(const int N, const int KL, const int KU,
                                  ValueType* const AB_array_entry, const int j,
                                  const int JB, const int J1)
{
    const int AB_nrows = 2 * KL + KU + 1;

    for (int A_col = j + JB; A_col < j + JB + J1; A_col++) {
        for (int A_row = j; A_row < j + JB; A_row++) {
            ValueType sum = zero<ValueType>();

            for (int i = 0; i < A_row - j; i++) {
                // sum += inner product(A01 col , A00 row)
                sum += AB_array_entry[(KL + KU + i + j - A_col) +
                                      AB_nrows * A_col] *
                       AB_array_entry[(KL + KU + A_row - (i + j)) +
                                      AB_nrows * (i + j)];
            }

            AB_array_entry[(KL + KU + A_row - A_col) + AB_nrows * A_col] -= sum;
        }
    }
}

template <typename ValueType, typename Callable>
void update_part_of_trailing_matrix(const int N, const int KL, const int KU,
                                    ValueType* const AB_array_entry,
                                    const int trail_row_start,
                                    const int trail_row_end_exclusive,
                                    const int trail_col_start,
                                    const int trail_col_end_exclusive,
                                    Callable compute_sum)
{
    const int AB_nrows = 2 * KL + KU + 1;

    for (int trail_row = trail_row_start; trail_row < trail_row_end_exclusive;
         trail_row++) {
        for (int trail_col = trail_col_start;
             trail_col < trail_col_end_exclusive; trail_col++) {
            const ValueType sum = compute_sum(trail_row, trail_col);

            AB_array_entry[(KL + KU + trail_row - trail_col) +
                           trail_col * AB_nrows] -= sum;
        }
    }
}

template <typename ValueType>
void update_A11(const int N, const int KL, const int KU,
                ValueType* const AB_array_entry, const int j, const int JB,
                const int I1, const int J1)
{
    const int AB_nrows = 2 * KL + KU + 1;

    auto compute_sum = [=](const int trail_row, const int trail_col) {
        ValueType sum = zero<ValueType>();
        for (int i = 0; i < JB; i++) {
            const int c = i + j;
            const int r = i + j;
            sum += AB_array_entry[(KL + KU + trail_row - c) + c * AB_nrows] *
                   AB_array_entry[(KL + KU + r - trail_col) +
                                  trail_col * AB_nrows];
        }

        return sum;
    };

    update_part_of_trailing_matrix(N, KL, KU, AB_array_entry, j + JB,
                                   j + JB + I1, j + JB, j + JB + J1,
                                   compute_sum);
}

template <typename ValueType>
void update_A21(const int N, const int KL, const int KU,
                ValueType* const AB_array_entry, const int j, const int JB,
                const int I1, const int I2, const int J1, const int NB,
                const ValueType* work20_entry)
{
    const int AB_nrows = 2 * KL + KU + 1;

    auto compute_sum = [=](const int trail_row, const int trail_col) {
        ValueType sum = zero<ValueType>();

        for (int i = 0; i < JB; i++) {
            const int r = i + j;
            sum += work20_entry[trail_row - (j + JB + I1) + i * NB] *
                   AB_array_entry[(KL + KU + r - trail_col) +
                                  trail_col * AB_nrows];
        }

        return sum;
    };

    update_part_of_trailing_matrix(N, KL, KU, AB_array_entry, j + JB + I1,
                                   j + JB + I1 + I2, j + JB, j + JB + J1,
                                   compute_sum);
}

template <typename ValueType>
void update_A12(const int N, const int KL, const int KU,
                ValueType* const AB_array_entry, const int j, const int JB,
                const int I1, const int J1, const int J2, const int NB,
                const ValueType* work02_entry)
{
    const int AB_nrows = 2 * KL + KU + 1;

    auto compute_sum = [=](const int trail_row, const int trail_col) {
        ValueType sum = zero<ValueType>();

        for (int i = 0; i < JB; i++) {
            const int c = i + j;
            const int offset_col = j + JB + J1;
            sum += AB_array_entry[(KL + KU + trail_row - c) + c * AB_nrows] *
                   work02_entry[i + (trail_col - offset_col) * NB];
        }
        return sum;
    };

    // update A12
    update_part_of_trailing_matrix(N, KL, KU, AB_array_entry, j + JB,
                                   j + JB + I1, j + JB + J1, j + JB + J1 + J2,
                                   compute_sum);
}

template <typename ValueType>
void update_A22(const int N, const int KL, const int KU,
                ValueType* const AB_array_entry, const int j, const int JB,
                const int I1, const int I2, const int J1, const int J2,
                const int NB, const ValueType* work20_entry,
                const ValueType* work02_entry)
{
    const int AB_nrows = 2 * KL + KU + 1;

    auto compute_sum = [=](const int trail_row, const int trail_col) {
        ValueType sum = zero<ValueType>();

        for (int i = 0; i < JB; i++) {
            const int offset_col = j + JB + J1;
            const int offset_row = j + JB + I1;
            sum += work20_entry[(trail_row - offset_row) + i * NB] *
                   work02_entry[i + (trail_col - offset_col) * NB];
        }

        return sum;
    };
    // update A22
    update_part_of_trailing_matrix(N, KL, KU, AB_array_entry, j + JB + I1,
                                   j + JB + I1 + I2, j + JB + J1,
                                   j + JB + J1 + J2, compute_sum);
}

template <typename ValueType>
void lower_trsv_update_work02(const int N, const int KL, const int KU,
                              const ValueType* AB_array_entry, const int j,
                              const int J2, const int JB, const int NB,
                              ValueType* const work02_entry)
{
    const int AB_nrows = 2 * KL + KU + 1;

    for (int c = 0; c < J2; c++) {
        for (int r = 0; r < JB; r++) {
            ValueType sum = zero<ValueType>();

            for (int i = 0; i < r; i++) {
                const int A_col = i + j;
                sum += AB_array_entry[(KL + KU + (r + j) - A_col) +
                                      AB_nrows * A_col] *
                       work02_entry[i + c * NB];
            }
            work02_entry[r + c * NB] -= sum;
        }
    }
}


template <typename ValueType>
inline void factorize_band(const int N, const int KL, const int KU,
                           const int NB, ValueType* const AB_array_entry,
                           int* const ipiv_entry, ValueType* const work02_entry,
                           ValueType* const work20_entry)
{
    const int AB_nrows = 2 * KL + KU + 1;
    const int KV = KL + KU;

    common_band_factorization::set_possible_fill_in_U_part_zero(N, KL, KU,
                                                                AB_array_entry);

    for (int i = 0; i < NB * NB; i++) {
        work20_entry[i] = zero<ValueType>();
        work02_entry[i] = zero<ValueType>();
    }

    /*
    Main steps- blocked LU factorization: Iterate over the
    column panels and perform the following steps-

    > Partition the rows- finding I2, I3

    > Unblocked factorization of column
    panel (with full row swaps - this is
    where Work31 is needed)

    > Partition the columns - finding J2, J3

    > Applying row interchanges to the set
    of J2 columns and J3 columns

    > Lower triangular solves to update
    A12, A13

    > Matrix-Matrix multiplications to
    update A22, A32, A23, A33

    > Column panel- undo the “full” row
    interchanges by partial row swaps
    */

    int A_affected_col_idx_last = 0;

    for (int j = 0; j < N; j += NB) {
        const int JB = std::min(NB, N - j);

        /*
         *          The active part of the matrix is partitioned:
         *
         *              A00   A01   A02
         *              A10   A11   A12
         *              A20   A21   A22
         *
         *           Here A00, A10 and A20 denote the current block of JB
         *           columns which is about to be factorized. The number of rows
         *           in the partitions are JB, I1, I2 respectively, and the
         *           number of columns are JB, J1, J1. The superdiagonal
         *           elements of A02 and the subdiagonal elements of A20
         *           lie outside the band.
         *
         */

        const int I1 = std::min(KL - JB, N - j - JB);
        const int I2 = std::min(JB, N - j - KL);

        // J1 and J1 are computed after A_affected_col_idx_last has been
        // updated.

        // Factorize the current block of JB columns
        for (int jj = j; jj < j + JB; jj++) {
            const int AB_pivot_row_idx =
                common_band_factorization::find_pivot_row_index(
                    N, KL, KU, AB_array_entry, jj);
            const int A_pivot_row_idx = AB_pivot_row_idx - KV + jj;
            ipiv_entry[jj] = A_pivot_row_idx;

            const ValueType piv_val =
                AB_array_entry[AB_pivot_row_idx + AB_nrows * jj];
            assert(piv_val != zero<ValueType>());

            // the index of the last column affected by the current stage of the
            // factorization
            A_affected_col_idx_last = std::max(
                A_affected_col_idx_last, std::min(A_pivot_row_idx + KU, N - 1));

            if (A_pivot_row_idx != jj) {
                // swap: row swaps for columns j to j + JB - 1(inclusive)
                if (A_pivot_row_idx <= j + JB - 1 + I1) {
                    // swap in band array
                    common_band_factorization::swap_rows(
                        N, KL, KU, AB_array_entry, j, j + JB - 1, jj,
                        A_pivot_row_idx);
                } else {
                    // for cols. j to jj - 1 --> do in work20
                    for (int A_col_idx = j; A_col_idx <= jj - 1; A_col_idx++) {
                        const int AB_row_idx1 = KL + KU + jj - A_col_idx;
                        const ValueType temp =
                            AB_array_entry[AB_row_idx1 + AB_nrows * A_col_idx];
                        const int r = A_pivot_row_idx - (j + JB + I1);
                        AB_array_entry[AB_row_idx1 + AB_nrows * A_col_idx] =
                            work20_entry[r + (A_col_idx - j) * NB];
                        work20_entry[r + (A_col_idx - j) * NB] = temp;
                    }

                    // for cols. jj to j + JB -1 --> do in band array
                    common_band_factorization::swap_rows(
                        N, KL, KU, AB_array_entry, jj, j + JB - 1, jj,
                        A_pivot_row_idx);
                }
            }

            common_band_factorization::scale_column(N, KL, KU, AB_array_entry,
                                                    jj, piv_val);

            /*
                Update trailing submatrix within the band and within
                the current block. JM is the index of the last column
                which needs to be updated.

                jm = min( ju, j+jb-1 ) // No need since we ensured KL >= NB

            */

            // Update trailing submatrix within the band and within the current
            // block
            common_band_factorization::update_trailing_submatrix(
                N, KL, KU, AB_array_entry, jj, j + JB - 1);

            // Copy current column of A20 into the work array work20
            for (int A_row = j + JB + I1; A_row <= std::min(jj + KL, N - 1);
                 A_row++) {
                const int AB_row = KL + KU + A_row - jj;
                const ValueType val = AB_array_entry[AB_row + AB_nrows * jj];
                const int r = A_row - (j + JB + I1);
                work20_entry[r + (jj - j) * NB] = val;
            }
        }

        // Apply row interchanges to other blocks
        const int J1 = std::min(A_affected_col_idx_last - j + 1, KV) - JB;
        const int J2 = std::max(0, A_affected_col_idx_last - j - KV + 1);

        // apply the row interchanges to A01, A11, and A21
        apply_row_interchanges_A01_A11_A21(N, KL, KU, AB_array_entry, j, JB, J1,
                                           ipiv_entry);

        // apply the row interchanges to A02, A12, and A22 columnwise
        apply_row_interchanges_A02_A12_A22(N, KL, KU, AB_array_entry, j, JB, J1,
                                           J2, ipiv_entry);

        // Update the relevant part of the trailing submatrix
        if (J1 > 0) {
            // lower trsv - update A01
            lower_trsv_update_A01(N, KL, KU, AB_array_entry, j, JB, J1);

            if (I1 > 0) {
                update_A11(N, KL, KU, AB_array_entry, j, JB, I1, J1);
            }

            if (I2 > 0) {
                update_A21(N, KL, KU, AB_array_entry, j, JB, I1, I2, J1, NB,
                           work20_entry);
            }
        }

        if (J2 > 0) {
            // Copy part of A02 into the work array WORK02
            for (int A_col = j + JB + J1; A_col < j + JB + J1 + J2; A_col++) {
                const int A_row_start = std::max(0, A_col - KL - KU);
                for (int A_row = A_row_start; A_row < j + JB; A_row++) {
                    const int AB_row = KL + KU + A_row - A_col;
                    const ValueType val =
                        AB_array_entry[AB_row + AB_nrows * A_col];
                    const int r = A_row - j;
                    const int c = A_col - (j + JB + J1);
                    work02_entry[r + c * NB] = val;
                }
            }

            lower_trsv_update_work02(N, KL, KU, AB_array_entry, j, J2, JB, NB,
                                     work02_entry);

            if (I1 > 0) {
                update_A12(N, KL, KU, AB_array_entry, j, JB, I1, J1, J2, NB,
                           work02_entry);
            }

            if (I2 > 0) {
                update_A22(N, KL, KU, AB_array_entry, j, JB, I1, I2, J1, J2, NB,
                           work20_entry, work02_entry);
            }

            // Copy part of A02 back into place
            for (int A_col = j + JB + J1; A_col < j + JB + J1 + J2; A_col++) {
                const int A_row_start = std::max(0, A_col - KL - KU);
                for (int A_row = A_row_start; A_row < j + JB; A_row++) {
                    const int AB_row = KL + KU + A_row - A_col;
                    const int r = A_row - j;
                    const int c = A_col - (j + JB + J1);
                    const ValueType val = work02_entry[r + c * NB];
                    AB_array_entry[AB_row + AB_nrows * A_col] = val;
                }
            }
        }

        // undo partial row interchanges in the column panel - A00, A10, A20
        for (int a = j + JB - 1; a >= j; a--) {
            const int A_row_1 = a;
            const int A_row_2 = ipiv_entry[a];

            // swap rows for cols. j to a - 1 (inclusive)
            if (A_row_2 <= j + JB + I1 - 1) {
                common_band_factorization::swap_rows(
                    N, KL, KU, AB_array_entry, j, a - 1, A_row_1, A_row_2);
            } else {
                // swap from col. j to a-1 (inclusive)
                for (int A_col = j; A_col <= a - 1; A_col++) {
                    const int AB_row_idx1 = KL + KU + A_row_1 - A_col;
                    const ValueType temp =
                        AB_array_entry[AB_row_idx1 + AB_nrows * A_col];
                    const int r = A_row_2 - (j + JB + I1);
                    AB_array_entry[AB_row_idx1 + AB_nrows * A_col] =
                        work20_entry[r + (A_col - j) * NB];
                    work20_entry[r + (A_col - j) * NB] = temp;
                }
            }
        }

        // copy relevant part of work20 to A20
        for (int A_col = j; A_col < j + JB; A_col++) {
            for (int A_row = j + JB + I1; A_row < std::min(A_col + KL + 1, N);
                 A_row++) {
                const int AB_row = KL + KU + A_row - A_col;
                const int r = A_row - (j + JB + I1);
                const int c = A_col - j;
                const ValueType val = work20_entry[r + c * NB];
                AB_array_entry[AB_row + A_col * AB_nrows] = val;
            }
        }
    }
}

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
    // Based on unblocked LU factorization and subsequent lower and upper
    // triangular solves
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
    // Based on column panel LU factorization and subsequent lower and upper
    // triangular solves
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

    ValueType* const work02_entry = temp + nrhs * N;
    ValueType* const work20_entry =
        work02_entry + blocked_solve_panel_size * blocked_solve_panel_size;
    int* const ipiv_entry = reinterpret_cast<int*>(
        work20_entry + blocked_solve_panel_size * blocked_solve_panel_size);

    const int AB_nrows = 2 * KL + KU + 1;
    const int AB_ncols = N;

    ValueType* const AB_array_entry =
        batch_band_mat_array + ibatch * AB_nrows * AB_ncols;

    blocked_LU_band_factorization::factorize_band(
        N, KL, KU, blocked_solve_panel_size, AB_array_entry, ipiv_entry,
        work02_entry, work20_entry);

    unblocked_and_blocked_band_solver_common::
        accommodate_partial_row_swaps_lower_trsv(
            N, KL, KU, ipiv_entry, AB_array_entry, b_entry, temp_entry);

    unblocked_and_blocked_band_solver_common::upper_trsv(
        N, KL, KU, AB_array_entry, gko::batch::to_const(temp_entry), x_entry);
}
