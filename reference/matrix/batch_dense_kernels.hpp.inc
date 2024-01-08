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
