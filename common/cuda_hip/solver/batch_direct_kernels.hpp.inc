/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
__global__ void setup_batch_pointers(
    const size_type nbatch, const int nrows, const size_type mat_stride,
    ValueType* const __restrict__ entire_batch_mats,
    ValueType** const __restrict__ mat_batches, const int nrhs,
    const size_type vec_stride, ValueType* const __restrict__ entire_batch_vecs,
    ValueType** const __restrict__ vec_batches)
{
    for (size_type ib = thread::get_thread_id_flat(); ib < nbatch;
         ib += thread::get_thread_num_flat()) {
        mat_batches[ib] = gko::batch::batch_entry_ptr(entire_batch_mats,
                                                      mat_stride, nrows, ib);
        vec_batches[ib] = gko::batch::batch_entry_ptr(entire_batch_vecs,
                                                      vec_stride, nrhs, ib);
    }
}


template <typename ValueType>
__global__ void transpose_scale_copy(
    const size_type nbatch, const int nrows, const int nrhs,
    const size_type in_stride, const size_type out_stride,
    const ValueType* const __restrict__ scaling,
    const ValueType* const __restrict__ in, ValueType* const __restrict__ out)
{
    for (size_type ib = blockIdx.x; ib < nbatch; ib += gridDim.x) {
        auto outb = gko::batch::batch_entry_ptr(out, out_stride, nrows, ib);
        auto inb = gko::batch::batch_entry_ptr(in, in_stride, nrhs, ib);
        auto scaleb = gko::batch::batch_entry_ptr(scaling, 1, nrows, ib);
        for (int iz = threadIdx.x; iz < nrows * nrhs; iz += blockDim.x) {
            const int irow = iz / nrhs;
            const int jcol = iz % nrhs;
            outb[irow * out_stride + jcol] =
                inb[jcol * in_stride + irow] * scaleb[irow];
        }
    }
}


template <typename ValueType>
__device__ __forceinline__ void pre_diag_scale_system_transpose(
    const int num_rows, const int num_cols, const size_type a_stride,
    const ValueType* const __restrict__ a, const int num_rhs,
    const size_type b_stride, const ValueType* const __restrict__ b,
    const ValueType* const __restrict__ left_scale,
    const ValueType* const __restrict__ right_scale,
    const size_type a_scaled_stride, ValueType* const __restrict__ a_scaled_t,
    const size_type b_scaled_stride, ValueType* const __restrict__ b_scaled_t)
{
    for (int iz = threadIdx.x; iz < num_rows * num_cols; iz += blockDim.x) {
        const int row = iz / num_cols;
        const int col = iz % num_cols;
        a_scaled_t[col * a_scaled_stride + row] =
            a[row * a_stride + col] * left_scale[row] * right_scale[col];
    }
    for (int iz = threadIdx.x; iz < num_rows * num_rhs; iz += blockDim.x) {
        const int row = iz / num_rhs;
        const int col = iz % num_rhs;
        b_scaled_t[col * b_scaled_stride + row] =
            b[row * b_stride + col] * left_scale[row];
    }
}

template <typename ValueType>
__global__ void pre_diag_scale_system_transpose(
    const size_type nbatch, const int num_rows, const int num_cols,
    const size_type a_stride, const ValueType* const __restrict__ a,
    const int num_rhs, const size_type b_stride,
    const ValueType* const __restrict__ b, const size_type left_scale_stride,
    const ValueType* const __restrict__ left_scale,
    const size_type rght_scale_stride,
    const ValueType* const __restrict__ rght_scale,
    const size_type a_scaled_stride, ValueType* const __restrict__ a_scaled_t,
    const size_type b_scaled_stride, ValueType* const __restrict__ b_scaled_t)
{
    for (size_type ib = blockIdx.x; ib < nbatch; ib += gridDim.x) {
        auto ab = gko::batch::batch_entry_ptr(a, a_stride, num_rows, ib);
        auto astb = gko::batch::batch_entry_ptr(a_scaled_t, a_scaled_stride,
                                                num_cols, ib);
        auto bb = gko::batch::batch_entry_ptr(b, b_stride, num_rows, ib);
        auto bstb = gko::batch::batch_entry_ptr(b_scaled_t, b_scaled_stride,
                                                num_rhs, ib);
        auto lscaleb = gko::batch::batch_entry_ptr(
            left_scale, left_scale_stride, num_rows, ib);
        auto rscaleb = gko::batch::batch_entry_ptr(
            rght_scale, rght_scale_stride, num_cols, ib);
        pre_diag_scale_system_transpose(
            num_rows, num_cols, a_stride, ab, num_rhs, b_stride, bb, lscaleb,
            rscaleb, a_scaled_stride, astb, b_scaled_stride, bstb);
    }
}
