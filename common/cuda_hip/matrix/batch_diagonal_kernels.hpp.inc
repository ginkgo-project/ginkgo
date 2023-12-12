// Copyright (c) 2017-2023, the Ginkgo authors
// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename ValueType>
__device__ __forceinline__ void simple_apply_inplace(
    const int num_rows, const size_type stride, const int num_rhs,
    const ValueType* const diag_vec, ValueType* const a)
{
    for (int iz = threadIdx.x; iz < num_rows * num_rhs; iz += blockDim.x) {
        const int row = iz / num_rhs;
        const int col = iz % num_rhs;
        a[row * stride + col] *= diag_vec[row];
    }
}

template <typename ValueType>
__global__ void simple_apply_inplace_kernel(
    const int nrows, const size_type stride, const int nrhs,
    const size_type num_batch_items,
    const ValueType* const __restrict__ diag_vals,
    ValueType* const __restrict__ a)
{
    for (size_type batch_id = blockIdx.x; batch_id < num_batch_items;
         batch_id += gridDim.x) {
        int offset = batch_id * nrows;
        simple_apply_inplace(nrows, stride, nrhs, diag_vals + offset,
                             a + offset * stride);
    }
}


template <typename ValueType>
__device__ __forceinline__ void simple_apply(
    const int nrows, const ValueType* const diag, const int nrhs,
    const size_type b_stride, const ValueType* const b,
    const size_type x_stride, ValueType* const x)
{
    for (int iz = threadIdx.x; iz < nrows * nrhs; iz += blockDim.x) {
        const int row = iz / nrhs;
        const int col = iz % nrhs;
        x[row * x_stride + col] = diag[row] * b[row * b_stride + col];
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void simple_apply_kernel(
    const size_type num_batch_items, const int nrows,
    const ValueType* const __restrict__ diag_vals, const int nrhs,
    const size_type b_stride, const ValueType* const __restrict__ b,
    const size_type x_stride, ValueType* const __restrict__ x)
{
    for (size_type batch_id = blockIdx.x; batch_id < num_batch_items;
         batch_id += gridDim.x) {
        int offset = batch_id * nrows;
        simple_apply(nrows, diag_vals + offset, nrhs, b_stride,
                     b + offset * b_stride, x_stride, x + offset * x_stride);
    }
}
