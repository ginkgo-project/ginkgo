// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename ValueType, typename IndexType>
inline void simple_apply_kernel(
    const gko::batch::matrix::ell::batch_item<const ValueType, IndexType>& a,
    const gko::batch::multi_vector::batch_item<const ValueType>& b,
    const gko::batch::multi_vector::batch_item<ValueType>& c)
{
    for (int row = 0; row < c.num_rows; ++row) {
        for (int j = 0; j < c.num_rhs; ++j) {
            c.values[row * c.stride + j] = zero<ValueType>();
        }
        for (auto k = 0; k < a.num_stored_elems_per_row; ++k) {
            auto val = a.values[row + k * a.stride];
            auto col = a.col_idxs[row + k * a.stride];
            if (col != invalid_index<IndexType>()) {
                for (int j = 0; j < c.num_rhs; ++j) {
                    c.values[row * c.stride + j] +=
                        val * b.values[col * b.stride + j];
                }
            }
        }
    }
}


template <typename ValueType, typename IndexType>
inline void advanced_apply_kernel(
    const ValueType alpha,
    const gko::batch::matrix::ell::batch_item<const ValueType, IndexType>& a,
    const gko::batch::multi_vector::batch_item<const ValueType>& b,
    const ValueType beta,
    const gko::batch::multi_vector::batch_item<ValueType>& c)
{
    for (int row = 0; row < a.num_rows; ++row) {
        for (int j = 0; j < c.num_rhs; ++j) {
            c.values[row * c.stride + j] *= beta;
        }
        for (auto k = 0; k < a.num_stored_elems_per_row; ++k) {
            auto val = a.values[row + k * a.stride];
            auto col = a.col_idxs[row + k * a.stride];
            if (col != invalid_index<IndexType>()) {
                for (int j = 0; j < b.num_rhs; ++j) {
                    c.values[row * c.stride + j] +=
                        alpha * val * b.values[col * b.stride + j];
                }
            }
        }
    }
}


template <typename ValueType, typename IndexType>
inline void scale(
    const ValueType* const col_scale, const ValueType* const row_scale,
    const gko::batch::matrix::ell::batch_item<ValueType, IndexType>& mat)
{
    for (int row = 0; row < mat.num_rows; row++) {
        const ValueType r_scalar = row_scale[row];
        for (auto k = 0; k < mat.num_stored_elems_per_row; ++k) {
            auto col_idx = mat.col_idxs[row + mat.stride * k];
            if (col_idx == invalid_index<IndexType>()) {
                break;
            } else {
                mat.values[row + mat.stride * k] *=
                    r_scalar * col_scale[col_idx];
            }
        }
    }
}


template <typename ValueType, typename IndexType>
inline void add_scaled_identity_kernel(
    const ValueType alpha, const ValueType beta,
    const gko::batch::matrix::ell::batch_item<ValueType, IndexType>& mat)
{
    for (int row = 0; row < mat.num_rows; row++) {
        for (int k = 0; k < mat.num_stored_elems_per_row; k++) {
            mat.values[row + k * mat.stride] *= beta;
            auto col_idx = mat.col_idxs[row + mat.stride * k];
            if (col_idx == invalid_index<IndexType>()) {
                break;
            } else {
                if (row == col_idx) {
                    mat.values[row + k * mat.stride] += alpha;
                }
            }
        }
    }
}
