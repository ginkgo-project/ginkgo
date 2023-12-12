// Copyright (c) 2017-2023, the Ginkgo authors
// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename ValueType>
inline void simple_apply_in_place(
    const ValueType* const diag_vec,
    const gko::batch::multi_vector::batch_item<ValueType>& a)
{
    for (int i_row = 0; i_row < a.num_rows; i_row++) {
        const ValueType scale = diag_vec[i_row];
        for (int j = 0; j < a.num_rhs; j++) {
            a.values[i_row * a.stride + j] *= scale;
        }
    }
}


template <typename ValueType>
inline void simple_apply(const int nrows, const ValueType* const diag,
                         const int nrhs, const int x_stride,
                         const ValueType* const x, const int y_stride,
                         ValueType* const y)
{
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < nrhs; j++) {
            y[i * y_stride + j] = diag[i] * x[i * x_stride + j];
        }
    }
}
