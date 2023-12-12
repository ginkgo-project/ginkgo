// Copyright (c) 2017-2023, the Ginkgo authors
// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename ValueType>
__dpct_inline__ void simple_apply_kernel(
    const int nrows, const ValueType* const diag, const int nrhs,
    const size_type b_stride, const ValueType* const b,
    const size_type x_stride, ValueType* const x, sycl::nd_item<3>& item_ct1)
{
    const int local_id = item_ct1.get_local_linear_id();
    const int local_range = item_ct1.get_local_range().size();

    for (int iz = local_id; iz < nrows * nrhs; iz += local_range) {
        const int row = iz / nrhs;
        const int col = iz % nrhs;
        x[row * x_stride + col] = diag[row] * b[row * b_stride + col];
    }
}


template <typename ValueType>
__dpct_inline__ void simple_apply_in_place_kernel(
    const int num_rows, const size_type stride, const int num_rhs,
    const ValueType* const diag_vec, ValueType* const a,
    sycl::nd_item<3>& item_ct1)
{
    for (int iz = item_ct1.get_local_linear_id(); iz < num_rows * num_rhs;
         iz += item_ct1.get_local_range().size()) {
        const int row = iz / num_rhs;
        const int col = iz % num_rhs;
        a[row * stride + col] *= diag_vec[row];
    }
}
