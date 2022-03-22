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

/**
 * 'Device' kernel for SpMV of one CSR matrix in a batch.
 *
 * Assumes the input and output multi-vectors are stored row-major.
 */
template <typename ValueType>
inline void matvec_kernel(
    const gko::batch_ell::BatchEntry<const ValueType>& a,
    const gko::batch_dense::BatchEntry<const ValueType>& b,
    const gko::batch_dense::BatchEntry<ValueType>& c)
{
    for (int row = 0; row < a.num_rows; ++row) {
        for (int j = 0; j < b.num_rhs; ++j) {
            c.values[row * c.stride + j] = zero<ValueType>();
        }
        for (auto k = 0; k < a.num_stored_elems_per_row; ++k) {
            auto val = a.values[row + k * a.stride];
            auto col = a.col_idxs[row + k * a.stride];
            for (int j = 0; j < b.num_rhs; ++j) {
                c.values[row * c.stride + j] +=
                    val * b.values[col * b.stride + j];
            }
        }
    }
}


/**
 * 'Device' kernel for 'advanced' SpMV of one CSR matrix in a batch.
 *
 * Assumes the input and output multi-vectors are stored row-major.
 */
template <typename ValueType>
inline void advanced_matvec_kernel(
    const ValueType alpha, const gko::batch_ell::BatchEntry<const ValueType>& a,
    const gko::batch_dense::BatchEntry<const ValueType>& b,
    const ValueType beta, const gko::batch_dense::BatchEntry<ValueType>& c)
{
    for (int row = 0; row < a.num_rows; ++row) {
        for (int j = 0; j < c.num_rhs; ++j) {
            c.values[row * c.stride + j] *= beta;
        }
        for (auto k = 0; k < a.num_stored_elems_per_row; ++k) {
            auto val = a.values[row + k * a.stride];
            auto col = a.col_idxs[row + k * a.stride];
            for (int j = 0; j < b.num_rhs; ++j) {
                c.values[row * c.stride + j] +=
                    alpha * val * b.values[col * b.stride + j];
            }
        }
    }
}


inline void check_diagonal_entries_exist(const int nrows, const int stride,
                                         const int max_nnz_per_row,
                                         const int* const col_idxs,
                                         bool& has_all_diags)
{
    has_all_diags = true;
    if (col_idxs[0] != 0) {
        has_all_diags = false;
        return;
    }
    for (int irow = 1; irow < nrows; irow++) {
        bool has_diag = false;
        for (int iz = 0; iz < max_nnz_per_row; iz++) {
            if (col_idxs[iz * stride + irow] == irow) {
                has_diag = true;
                break;
            }
        }
        if (!has_diag) {
            has_all_diags = false;
            break;
        }
    }
}


template <typename ValueType>
inline void add_scaled_identity(const int nrows, const int stride,
                                const int max_nnz_per_row,
                                const int* const col_idxs,
                                ValueType* const values, const ValueType& alpha,
                                const ValueType& beta)
{
    for (int jz = 0; jz < max_nnz_per_row; jz++) {
        values[jz * stride] *= beta;
        if (col_idxs[jz * stride] == 0 && jz == 0) {
            values[jz * stride] += alpha;
        }
    }
    for (int i = 1; i < nrows; i++) {
        for (int jz = 0; jz < max_nnz_per_row; jz++) {
            values[jz * stride + i] *= beta;
            if (i == col_idxs[jz * stride + i]) {
                values[jz * stride + i] += alpha;
            }
        }
    }
}
