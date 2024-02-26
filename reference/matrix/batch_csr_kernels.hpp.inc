// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename ValueType, typename IndexType>
inline void simple_apply_kernel(
    const gko::batch::matrix::csr::batch_item<const ValueType, IndexType>& a,
    const gko::batch::multi_vector::batch_item<const ValueType>& b,
    const gko::batch::multi_vector::batch_item<ValueType>& c)
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


template <typename ValueType, typename IndexType>
inline void advanced_apply_kernel(
    const ValueType alpha,
    const gko::batch::matrix::csr::batch_item<const ValueType, IndexType>& a,
    const gko::batch::multi_vector::batch_item<const ValueType>& b,
    const ValueType beta,
    const gko::batch::multi_vector::batch_item<ValueType>& c)
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


template <typename ValueType, typename IndexType>
inline void scale(
    const ValueType* const col_scale, const ValueType* const row_scale,
    const gko::batch::matrix::csr::batch_item<ValueType, IndexType>& mat)
{
    for (int row = 0; row < mat.num_rows; row++) {
        const ValueType r_scalar = row_scale[row];
        for (int col = mat.row_ptrs[row]; col < mat.row_ptrs[row + 1]; col++) {
            mat.values[col] *= r_scalar * col_scale[mat.col_idxs[col]];
        }
    }
}


template <typename ValueType, typename IndexType>
inline void add_scaled_identity_kernel(
    const ValueType alpha, const ValueType beta,
    const gko::batch::matrix::csr::batch_item<ValueType, IndexType>& mat)
{
    for (int row = 0; row < mat.num_rows; row++) {
        for (int nnz = mat.row_ptrs[row]; nnz < mat.row_ptrs[row + 1]; nnz++) {
            mat.values[nnz] *= beta;
            if (row == mat.col_idxs[nnz]) {
                mat.values[nnz] += alpha;
            }
        }
    }
}
