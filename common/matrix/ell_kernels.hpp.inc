/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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


template <int subwarp_size, bool atomic, typename ValueType, typename IndexType,
          typename Closure>
__device__ void spmv_kernel(const size_type num_rows, const int nwarps_per_row,
                            const ValueType *__restrict__ val,
                            const IndexType *__restrict__ col,
                            const size_type stride,
                            const size_type num_stored_elements_per_row,
                            const ValueType *__restrict__ b,
                            const size_type b_stride, ValueType *__restrict__ c,
                            const size_type c_stride, Closure op)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    const auto width = blockDim.y;
    const auto column_id = blockIdx.y;
    if (tidx < nwarps_per_row * num_rows) {
        const auto idx_in_warp = threadIdx.y;
        const auto x = tidx % num_rows;
        const auto warp_id = tidx / num_rows;
        const auto step_size = nwarps_per_row * width;
        __shared__ UninitializedArray<ValueType, default_block_size> storage;
        if (threadIdx.y == 0) {
            storage[threadIdx.x] = 0;
        }
        __syncthreads();
        ValueType temp = zero<ValueType>();
        for (auto idx = warp_id * width + idx_in_warp;
             idx < num_stored_elements_per_row; idx += step_size) {
            const auto ind = x + idx * stride;
            const auto col_idx = col[ind];
            if (col_idx < idx) {
                break;
            } else {
                temp += val[ind] * b[col_idx * b_stride + column_id];
            }
        }
        atomic_add(&storage[threadIdx.x], temp);
        __syncthreads();
        if (threadIdx.y == 0) {
            if (atomic) {
                atomic_add(
                    &(c[x * c_stride + column_id]),
                    op(storage[threadIdx.x], c[x * c_stride + column_id]));
            } else {
                c[x * c_stride + column_id] =
                    op(storage[threadIdx.x], c[x * c_stride + column_id]);
            }
        }
    }
}


template <int subwarp_size, bool atomic = false, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void spmv(
    const size_type num_rows, const int nwarps_per_row,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const size_type stride, const size_type num_stored_elements_per_row,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride)
{
    spmv_kernel<subwarp_size, atomic>(
        num_rows, nwarps_per_row, val, col, stride, num_stored_elements_per_row,
        b, b_stride, c, c_stride,
        [](const ValueType &x, const ValueType &y) { return x; });
}


template <int subwarp_size, bool atomic = false, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void spmv(
    const size_type num_rows, const int nwarps_per_row,
    const ValueType *__restrict__ alpha, const ValueType *__restrict__ val,
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
        spmv_kernel<subwarp_size, atomic>(
            num_rows, nwarps_per_row, val, col, stride,
            num_stored_elements_per_row, b, b_stride, c, c_stride,
            [&alpha_val](const ValueType &x, const ValueType &y) {
                return alpha_val * x;
            });
    } else {
        spmv_kernel<subwarp_size, atomic>(
            num_rows, nwarps_per_row, val, col, stride,
            num_stored_elements_per_row, b, b_stride, c, c_stride,
            [&alpha_val, &beta_val](const ValueType &x, const ValueType &y) {
                return alpha_val * x + beta_val * y;
            });
    }
}


}  // namespace


template <typename ValueType>
__global__ __launch_bounds__(config::max_block_size) void initialize_zero_dense(
    size_type num_rows, size_type num_cols, size_type stride,
    ValueType *__restrict__ result)
{
    const auto tidx_x = threadIdx.x + blockDim.x * blockIdx.x;
    const auto tidx_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (tidx_x < num_cols && tidx_y < num_rows) {
        result[tidx_y * stride + tidx_x] = zero<ValueType>();
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_dense(
    size_type num_rows, size_type nnz, size_type source_stride,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values, size_type result_stride,
    ValueType *__restrict__ result)
{
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx < num_rows) {
        for (auto col = 0; col < nnz; col++) {
            result[tidx * result_stride +
                   col_idxs[tidx + col * source_stride]] +=
                values[tidx + col * source_stride];
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void count_nnz_per_row(
    size_type num_rows, size_type max_nnz_per_row, size_type stride,
    const ValueType *__restrict__ values, IndexType *__restrict__ result)
{
    constexpr auto warp_size = config::warp_size;
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto row_idx = tidx / warp_size;

    if (row_idx < num_rows) {
        IndexType part_result{};
        for (auto i = threadIdx.x % warp_size; i < max_nnz_per_row;
             i += warp_size) {
            if (values[stride * i + row_idx] != zero<ValueType>()) {
                part_result += 1;
            }
        }

        auto warp_tile =
            group::tiled_partition<warp_size>(group::this_thread_block());
        result[row_idx] = reduce(
            warp_tile, part_result,
            [](const size_type &a, const size_type &b) { return a + b; });
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_csr(
    size_type num_rows, size_type max_nnz_per_row, size_type stride,
    const ValueType *__restrict__ source_values,
    const IndexType *__restrict__ source_col_idxs,
    IndexType *__restrict__ result_row_ptrs,
    IndexType *__restrict__ result_col_idxs,
    ValueType *__restrict__ result_values)
{
    const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;

    if (tidx < num_rows) {
        auto write_to = result_row_ptrs[tidx];
        for (auto i = 0; i < max_nnz_per_row; i++) {
            const auto source_idx = tidx + stride * i;
            if (source_values[source_idx] != zero<ValueType>()) {
                result_values[write_to] = source_values[source_idx];
                result_col_idxs[write_to] = source_col_idxs[source_idx];
                write_to++;
            }
        }
    }
}


}  // namespace kernel