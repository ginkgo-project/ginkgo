// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
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
