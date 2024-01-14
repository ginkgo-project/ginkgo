// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename ValueType>
inline void simple_apply_kernel(
    const gko::batch::matrix::dense::batch_item<const ValueType>& a,
    const gko::batch::multi_vector::batch_item<const ValueType>& b,
    const gko::batch::multi_vector::batch_item<ValueType>& c)
{
    for (int row = 0; row < c.num_rows; ++row) {
        for (int col = 0; col < c.num_rhs; ++col) {
            c.values[row * c.stride + col] = gko::zero<ValueType>();
        }
    }

    for (int row = 0; row < c.num_rows; ++row) {
        for (int inner = 0; inner < a.num_cols; ++inner) {
            for (int col = 0; col < c.num_rhs; ++col) {
                c.values[row * c.stride + col] +=
                    a.values[row * a.stride + inner] *
                    b.values[inner * b.stride + col];
            }
        }
    }
}


template <typename ValueType>
inline void advanced_apply_kernel(
    const ValueType alpha,
    const gko::batch::matrix::dense::batch_item<const ValueType>& a,
    const gko::batch::multi_vector::batch_item<const ValueType>& b,
    const ValueType beta,
    const gko::batch::multi_vector::batch_item<ValueType>& c)
{
    if (beta != gko::zero<ValueType>()) {
        for (int row = 0; row < c.num_rows; ++row) {
            for (int col = 0; col < c.num_rhs; ++col) {
                c.values[row * c.stride + col] *= beta;
            }
        }
    } else {
        for (int row = 0; row < c.num_rows; ++row) {
            for (int col = 0; col < c.num_rhs; ++col) {
                c.values[row * c.stride + col] = gko::zero<ValueType>();
            }
        }
    }

    for (int row = 0; row < c.num_rows; ++row) {
        for (int inner = 0; inner < a.num_cols; ++inner) {
            for (int col = 0; col < c.num_rhs; ++col) {
                c.values[row * c.stride + col] +=
                    alpha * a.values[row * a.stride + inner] *
                    b.values[inner * b.stride + col];
            }
        }
    }
}


template <typename ValueType>
inline void scale(const int num_rows, const int num_cols,
                  const size_type stride, const ValueType* const col_scale,
                  const ValueType* const row_scale, ValueType* const mat)
{
    for (int row = 0; row < num_rows; row++) {
        const ValueType row_scalar = row_scale[row];
        for (int col = 0; col < num_cols; col++) {
            mat[row * stride + col] *= row_scalar * col_scale[col];
        }
    }
}


template <typename ValueType>
inline void scale_add_kernel(
    const ValueType alpha,
    const gko::batch::matrix::dense::batch_item<const ValueType>& b,
    const gko::batch::matrix::dense::batch_item<ValueType>& in_out)
{
    for (int row = 0; row < b.num_rows; row++) {
        for (int col = 0; col < b.num_cols; col++) {
            in_out.values[row * in_out.stride + col] =
                alpha * in_out.values[row * in_out.stride + col] +
                b.values[row * b.stride + col];
        }
    }
}


template <typename ValueType>
inline void add_scaled_identity_kernel(
    const ValueType alpha, const ValueType beta,
    const gko::batch::matrix::dense::batch_item<ValueType>& mat)
{
    for (int row = 0; row < mat.num_rows; row++) {
        for (int col = 0; col < mat.num_cols; col++) {
            auto nnz = row * mat.stride + col;
            mat.values[nnz] *= beta;
            if (row == col) {
                mat.values[nnz] += alpha;
            }
        }
    }
}
