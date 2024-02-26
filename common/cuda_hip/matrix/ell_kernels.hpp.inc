// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

namespace kernel {


template <int num_thread_per_worker, bool atomic, typename b_accessor,
          typename a_accessor, typename OutputValueType, typename IndexType,
          typename Closure>
__device__ void spmv_kernel(
    const size_type num_rows, const int num_worker_per_row,
    acc::range<a_accessor> val, const IndexType* __restrict__ col,
    const size_type stride, const size_type num_stored_elements_per_row,
    acc::range<b_accessor> b, OutputValueType* __restrict__ c,
    const size_type c_stride, Closure op)
{
    using arithmetic_type = typename a_accessor::arithmetic_type;
    const auto tidx = thread::get_thread_id_flat();
    const decltype(tidx) column_id = blockIdx.y;
    if (num_thread_per_worker == 1) {
        // Specialize the num_thread_per_worker = 1. It doesn't need the shared
        // memory, __syncthreads, and atomic_add
        if (tidx < num_rows) {
            auto temp = zero<arithmetic_type>();
            for (size_type idx = 0; idx < num_stored_elements_per_row; idx++) {
                const auto ind = tidx + idx * stride;
                const auto col_idx = col[ind];
                if (col_idx == invalid_index<IndexType>()) {
                    break;
                } else {
                    temp += val(ind) * b(col_idx, column_id);
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
            __shared__ uninitialized_array<
                arithmetic_type, default_block_size / num_thread_per_worker>
                storage;
            if (idx_in_worker == 0) {
                storage[threadIdx.x] = 0;
            }
            __syncthreads();
            auto temp = zero<arithmetic_type>();
            for (size_type idx =
                     worker_id * num_thread_per_worker + idx_in_worker;
                 idx < num_stored_elements_per_row; idx += step_size) {
                const auto ind = x + idx * stride;
                const auto col_idx = col[ind];
                if (col_idx == invalid_index<IndexType>()) {
                    break;
                } else {
                    temp += val(ind) * b(col_idx, column_id);
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


template <int num_thread_per_worker, bool atomic = false, typename b_accessor,
          typename a_accessor, typename OutputValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void spmv(
    const size_type num_rows, const int num_worker_per_row,
    acc::range<a_accessor> val, const IndexType* __restrict__ col,
    const size_type stride, const size_type num_stored_elements_per_row,
    acc::range<b_accessor> b, OutputValueType* __restrict__ c,
    const size_type c_stride)
{
    spmv_kernel<num_thread_per_worker, atomic>(
        num_rows, num_worker_per_row, val, col, stride,
        num_stored_elements_per_row, b, c, c_stride,
        [](const auto& x, const OutputValueType& y) {
            return static_cast<OutputValueType>(x);
        });
}


template <int num_thread_per_worker, bool atomic = false, typename b_accessor,
          typename a_accessor, typename OutputValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void spmv(
    const size_type num_rows, const int num_worker_per_row,
    acc::range<a_accessor> alpha, acc::range<a_accessor> val,
    const IndexType* __restrict__ col, const size_type stride,
    const size_type num_stored_elements_per_row, acc::range<b_accessor> b,
    const OutputValueType* __restrict__ beta, OutputValueType* __restrict__ c,
    const size_type c_stride)
{
    using arithmetic_type = typename a_accessor::arithmetic_type;
    const auto alpha_val = alpha(0);
    const OutputValueType beta_val = beta[0];
    if (atomic) {
        // Because the atomic operation changes the values of c during
        // computation, it can not directly do alpha * a * b + beta * c
        // operation. The beta * c needs to be done before calling this kernel.
        // Then, this kernel only adds alpha * a * b when it uses atomic
        // operation.
        spmv_kernel<num_thread_per_worker, atomic>(
            num_rows, num_worker_per_row, val, col, stride,
            num_stored_elements_per_row, b, c, c_stride,
            [&alpha_val](const auto& x, const OutputValueType& y) {
                return static_cast<OutputValueType>(alpha_val * x);
            });
    } else {
        spmv_kernel<num_thread_per_worker, atomic>(
            num_rows, num_worker_per_row, val, col, stride,
            num_stored_elements_per_row, b, c, c_stride,
            [&alpha_val, &beta_val](const auto& x, const OutputValueType& y) {
                return static_cast<OutputValueType>(
                    alpha_val * x + static_cast<arithmetic_type>(beta_val * y));
            });
    }
}


}  // namespace kernel
