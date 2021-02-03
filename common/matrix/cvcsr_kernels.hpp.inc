/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

namespace kernel {
namespace {


template <int num_thread_per_worker, bool atomic, typename ValueType,
          typename IndexType, typename Closure, typename const_accessor>
__device__ void spmv_kernel(
    const size_type num_rows, const int num_worker_per_row,
    range<const_accessor> val, const IndexType *__restrict__ col,
    const size_type stride, const size_type num_stored_elements_per_row,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride, Closure op)
{
    const auto tidx = thread::get_thread_id_flat();
    const auto column_id = blockIdx.y;
    if (num_thread_per_worker == 1) {
        // Specialize the num_thread_per_worker = 1. It doesn't need the shared
        // memory, __syncthreads, and atomic_add
        if (tidx < num_rows) {
            ValueType temp = zero<ValueType>();
            for (size_type idx = 0; idx < num_stored_elements_per_row; idx++) {
                const auto ind = tidx + idx * stride;
                const auto col_idx = col[ind];
                if (col_idx < idx) {
                    break;
                } else {
                    temp += val(ind) * b[col_idx * b_stride + column_id];
                }
            }
            const auto c_ind = tidx * c_stride + column_id;
            c[c_ind] = op(temp, c[c_ind]);
        }
    } else {
        if (tidx < num_worker_per_row * num_rows) {
            const auto idx_in_worker = threadIdx.y;
            const auto x = tidx % num_rows;
            const auto worker_id = tidx / num_rows;
            const auto step_size = num_worker_per_row * num_thread_per_worker;
            __shared__ UninitializedArray<ValueType, default_block_size /
                                                         num_thread_per_worker>
                storage;
            if (idx_in_worker == 0) {
                storage[threadIdx.x] = 0;
            }
            __syncthreads();
            ValueType temp = zero<ValueType>();
            for (size_type idx =
                     worker_id * num_thread_per_worker + idx_in_worker;
                 idx < num_stored_elements_per_row; idx += step_size) {
                const auto ind = x + idx * stride;
                const auto col_idx = col[ind];
                if (col_idx < idx) {
                    break;
                } else {
                    temp += val(ind) * b[col_idx * b_stride + column_id];
                }
            }
            atomic_add(&storage[threadIdx.x], temp);
            __syncthreads();
            if (idx_in_worker == 0) {
                const auto c_ind = x * c_stride + column_id;
                if (atomic) {
                    atomic_add(&(c[c_ind]), op(storage[threadIdx.x], c[c_ind]));
                } else {
                    c[c_ind] = op(storage[threadIdx.x], c[c_ind]);
                }
            }
        }
    }
}


template <int num_thread_per_worker, bool atomic = false, typename ValueType,
          typename IndexType, typename const_accessor>
__global__ __launch_bounds__(default_block_size) void spmv(
    const size_type num_rows, const int num_worker_per_row,
    range<const_accessor> val, const IndexType *__restrict__ col,
    const size_type stride, const size_type num_stored_elements_per_row,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride)
{
    spmv_kernel<num_thread_per_worker, atomic>(
        num_rows, num_worker_per_row, val, col, stride,
        num_stored_elements_per_row, b, b_stride, c, c_stride,
        [](const ValueType &x, const ValueType &y) { return x; });
}


template <int num_thread_per_worker, bool atomic = false, typename ValueType,
          typename IndexType, typename const_accessor>
__global__ __launch_bounds__(default_block_size) void spmv(
    const size_type num_rows, const int num_worker_per_row,
    const ValueType *__restrict__ alpha, range<const_accessor> val,
    const IndexType *__restrict__ col, const size_type stride,
    const size_type num_stored_elements_per_row,
    const ValueType *__restrict__ b, const size_type b_stride,
    const ValueType *__restrict__ beta, ValueType *__restrict__ c,
    const size_type c_stride)
{
    const ValueType alpha_val = alpha[0];
    const ValueType beta_val = beta[0];
    // Because the atomic operation changes the values of c during computation,
    // it can not do the right alpha * a * b + beta * c operation.
    // Thus, the cuda kernel only computes alpha * a * b when it uses atomic
    // operation.
    if (atomic) {
        spmv_kernel<num_thread_per_worker, atomic>(
            num_rows, num_worker_per_row, val, col, stride,
            num_stored_elements_per_row, b, b_stride, c, c_stride,
            [&alpha_val](const ValueType &x, const ValueType &y) {
                return alpha_val * x;
            });
    } else {
        spmv_kernel<num_thread_per_worker, atomic>(
            num_rows, num_worker_per_row, val, col, stride,
            num_stored_elements_per_row, b, b_stride, c, c_stride,
            [&alpha_val, &beta_val](const ValueType &x, const ValueType &y) {
                return alpha_val * x + beta_val * y;
            });
    }
}


}  // namespace
}  // namespace kernel
