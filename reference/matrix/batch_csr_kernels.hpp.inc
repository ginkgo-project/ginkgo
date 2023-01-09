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

/**
 * 'Device' kernel for SpMV of one CSR matrix in a batch.
 *
 * Assumes the input and output multi-vectors are stored row-major.
 */
template <typename ValueType>
inline void matvec_kernel(
    const gko::batch_csr::BatchEntry<const ValueType>& a,
    const gko::batch_dense::BatchEntry<const ValueType>& b,
    const gko::batch_dense::BatchEntry<ValueType>& c)
{
    for (int row = 0; row < a.num_rows; ++row) {
        for (int j = 0; j < b.num_rhs; ++j) {
            c.values[row * c.stride + j] = zero<ValueType>();
        }
        for (auto k = a.row_ptrs[row]; k < a.row_ptrs[row + 1]; ++k) {
            auto val = a.values[k];
            auto col = a.col_idxs[k];
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
    const ValueType alpha, const gko::batch_csr::BatchEntry<const ValueType>& a,
    const gko::batch_dense::BatchEntry<const ValueType>& b,
    const ValueType beta, const gko::batch_dense::BatchEntry<ValueType>& c)
{
    for (int row = 0; row < a.num_rows; ++row) {
        for (int j = 0; j < c.num_rhs; ++j) {
            c.values[row * c.stride + j] *= beta;
        }
        for (int k = a.row_ptrs[row]; k < a.row_ptrs[row + 1]; ++k) {
            const auto val = a.values[k];
            const auto col = a.col_idxs[k];
            for (int j = 0; j < c.num_rhs; ++j) {
                c.values[row * c.stride + j] +=
                    alpha * val * b.values[col * b.stride + j];
            }
        }
    }
}


/**
 * Scales a uniform CSR matrix with dense vectors for row and column scaling.
 *
 * One warp is assigned to each row.
 */
template <typename ValueType>
inline void batch_scale(const ValueType* const left_scale,
                        const ValueType* const right_scale,
                        const gko::batch_csr::BatchEntry<ValueType>& a)
{
    for (int i_row = 0; i_row < a.num_rows; i_row++) {
        const ValueType rowscale = left_scale[i_row];
        for (int iz = a.row_ptrs[i_row]; iz < a.row_ptrs[i_row + 1]; iz++) {
            a.values[iz] *= rowscale * right_scale[a.col_idxs[iz]];
        }
    }
}


/**
 * Scales a matrix from the left and right, and a vector from the right.
 */
template <typename ValueType>
inline void pre_diag_scale_system(
    const size_type batch_id, const size_type a_batch_stride,
    const int num_rows, const int num_cols, ValueType* const a_values,
    const int* const col_idxs, const int* const row_ptrs, const int num_rhs,
    const size_type b_stride, ValueType* const b,
    const ValueType* const left_scale, const ValueType* const right_scale)
{
    auto ab = a_values + a_batch_stride * batch_id;
    auto bb = gko::batch::batch_entry_ptr(b, b_stride, num_rows, batch_id);
    auto left_scaleb =
        gko::batch::batch_entry_ptr(left_scale, 1, num_rows, batch_id);
    auto right_scaleb =
        gko::batch::batch_entry_ptr(right_scale, 1, num_cols, batch_id);
    for (int irow = 0; irow < num_rows; irow++) {
        const ValueType left_scale_value = left_scaleb[irow];
        for (int iz = row_ptrs[irow]; iz < row_ptrs[irow + 1]; iz++) {
            const int jcol = col_idxs[iz];
            ab[iz] *= left_scale_value * right_scaleb[jcol];
        }
        for (int irhs = 0; irhs < num_rhs; irhs++) {
            bb[irow * b_stride + irhs] *= left_scale_value;
        }
    }
}


template <typename ValueType>
inline void convert_csr_to_dense(const int num_rows, const int num_cols,
                                 const int* const row_ptrs,
                                 const int* const col_idxs,
                                 const ValueType* const values,
                                 const size_type dense_stride,
                                 ValueType* const dense_vals)
{
    for (int i_row = 0; i_row < num_rows; i_row++) {
        for (int j = 0; j < num_cols; j++) {
            dense_vals[i_row * dense_stride + j] = zero<ValueType>();
        }
        for (int iz = row_ptrs[i_row]; iz < row_ptrs[i_row + 1]; iz++) {
            const int colidx = col_idxs[iz];
            dense_vals[i_row * dense_stride + colidx] = values[iz];
        }
    }
}


inline void check_diagonal_entries_exist(const int nrows,
                                         const int* const row_ptrs,
                                         const int* const col_idxs,
                                         bool& has_all_diags)
{
    has_all_diags = true;
    for (int irow = 0; irow < nrows; irow++) {
        bool has_diag = false;
        for (int iz = row_ptrs[irow]; iz < row_ptrs[irow + 1]; iz++) {
            if (col_idxs[iz] == irow) {
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
inline void add_scaled_identity(const int nrows, const int* const row_ptrs,
                                const int* const col_idxs,
                                ValueType* const values, const ValueType& alpha,
                                const ValueType& beta)
{
    for (int i = 0; i < nrows; i++) {
        for (int jz = row_ptrs[i]; jz < row_ptrs[i + 1]; jz++) {
            values[jz] *= beta;
            if (i == col_idxs[jz]) {
                values[jz] += alpha;
            }
        }
    }
}
