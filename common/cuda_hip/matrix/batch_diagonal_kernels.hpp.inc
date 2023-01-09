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
 * Diagonal-scales a dense matrix or set of vectors with a diagonal matrix.
 * It is equivalent to multiplication by a diagonal matrix whose diagonal
 * entries are given by the scaling vector.
 *
 * One thread is assigned to each entry.
 */
template <typename ValueType>
__device__ __forceinline__ void batch_diag_apply_inplace(
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
__global__ void uniform_batch_diag_apply_inplace(
    const int nrows, const size_type stride, const int nrhs,
    const size_type num_batch, const ValueType* const __restrict__ diag_vals,
    ValueType* const __restrict__ a)
{
    for (size_type batch_id = blockIdx.x; batch_id < num_batch;
         batch_id += gridDim.x) {
        const auto a_ptr =
            gko::batch::batch_entry_ptr(a, stride, nrows, batch_id);
        const auto d_ptr =
            gko::batch::batch_entry_ptr(diag_vals, 1, nrows, batch_id);
        batch_diag_apply_inplace(nrows, stride, nrhs, d_ptr, a_ptr);
    }
}


template <typename ValueType>
__device__ __forceinline__ void batch_diag_apply(
    const int nrows, const int ncols, const ValueType* const diag,
    const int nrhs, const size_type b_stride, const ValueType* const b,
    const size_type x_stride, ValueType* const x)
{
    const int mindim = min(nrows, ncols);
    for (int iz = threadIdx.x; iz < mindim * nrhs; iz += blockDim.x) {
        const int row = iz / nrhs;
        const int col = iz % nrhs;
        x[row * x_stride + col] = diag[row] * b[row * b_stride + col];
    }
    for (int iz = threadIdx.x + mindim * nrhs; iz < nrows * nrhs;
         iz += blockDim.x) {
        const int row = iz / nrhs;
        const int col = iz % nrhs;
        x[row * x_stride + col] = zero<ValueType>();
    }
}

template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void uniform_batch_diag_apply(
    const size_type num_batch, const int nrows, const int ncols,
    const ValueType* const __restrict__ diag_vals, const int nrhs,
    const size_type b_stride, const ValueType* const __restrict__ b,
    const size_type x_stride, ValueType* const __restrict__ x)
{
    const int mindim = min(nrows, ncols);
    for (size_type batch_id = blockIdx.x; batch_id < num_batch;
         batch_id += gridDim.x) {
        const auto b_ptr =
            gko::batch::batch_entry_ptr(b, b_stride, ncols, batch_id);
        const auto x_ptr =
            gko::batch::batch_entry_ptr(x, x_stride, nrows, batch_id);
        const auto d_ptr =
            gko::batch::batch_entry_ptr(diag_vals, 1, mindim, batch_id);
        batch_diag_apply(nrows, ncols, d_ptr, nrhs, b_stride, b_ptr, x_stride,
                         x_ptr);
    }
}
