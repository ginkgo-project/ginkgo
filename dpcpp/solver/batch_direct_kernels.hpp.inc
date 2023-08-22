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

template <typename ValueType>
__dpct_inline__ void transpose_scale_copy_kernel(
    const size_type nbatch, const int nrows, const int nrhs,
    const size_type in_stride, const size_type out_stride,
    const ValueType* const __restrict__ scaling,
    const ValueType* const __restrict__ in, ValueType* const __restrict__ out,
    sycl::nd_item<3> item_ct1)
{
    for (int iz = item_ct1.get_local_linear_id(); iz < nrows * nrhs;
         iz += item_ct1.get_local_range().size()) {
        const int irow = iz / nrhs;
        const int jcol = iz % nrhs;
        out[irow * out_stride + jcol] =
            in[jcol * in_stride + irow] * scaling[irow];
    }
}


template <typename ValueType>
__dpct_inline__ void pre_diag_scale_system_transpose(
    const int num_rows, const int num_cols, const size_type a_stride,
    const ValueType* const __restrict__ a, const int num_rhs,
    const size_type b_stride, const ValueType* const __restrict__ b,
    const ValueType* const __restrict__ left_scale,
    const ValueType* const __restrict__ right_scale,
    const size_type a_scaled_stride, ValueType* const __restrict__ a_scaled_t,
    const size_type b_scaled_stride, ValueType* const __restrict__ b_scaled_t,
    sycl::nd_item<3> item_ct1)
{
    auto tid = item_ct1.get_local_linear_id();
    auto group_size = item_ct1.get_local_range().size();

    for (int iz = tid; iz < num_rows * num_cols; iz += group_size) {
        const int row = iz / num_cols;
        const int col = iz % num_cols;
        a_scaled_t[col * a_scaled_stride + row] =
            a[row * a_stride + col] * left_scale[row] * right_scale[col];
    }
    for (int iz = tid; iz < num_rows * num_rhs; iz += group_size) {
        const int row = iz / num_rhs;
        const int col = iz % num_rhs;
        b_scaled_t[col * b_scaled_stride + row] =
            b[row * b_stride + col] * left_scale[row];
    }
}
