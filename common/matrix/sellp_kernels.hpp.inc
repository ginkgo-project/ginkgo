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

namespace {


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(matrix::default_slice_size) void spmv_kernel(
    size_type num_rows, size_type num_right_hand_sides, size_type b_stride,
    size_type c_stride, const size_type *__restrict__ slice_lengths,
    const size_type *__restrict__ slice_sets, const ValueType *__restrict__ a,
    const IndexType *__restrict__ col, const ValueType *__restrict__ b,
    ValueType *__restrict__ c)
{
    const auto slice_id = blockIdx.x;
    const auto slice_size = blockDim.x;
    const auto row_in_slice = threadIdx.x;
    const auto global_row =
        static_cast<size_type>(slice_size) * slice_id + row_in_slice;
    const auto column_id = blockIdx.y;
    ValueType val = 0;
    IndexType ind = 0;
    if (global_row < num_rows && column_id < num_right_hand_sides) {
        for (size_type i = 0; i < slice_lengths[slice_id]; i++) {
            ind = row_in_slice + (slice_sets[slice_id] + i) * slice_size;
            val += a[ind] * b[col[ind] * b_stride + column_id];
        }
        c[global_row * c_stride + column_id] = val;
    }
}


template <typename ValueType, typename IndexType>
__global__
    __launch_bounds__(matrix::default_slice_size) void advanced_spmv_kernel(
        size_type num_rows, size_type num_right_hand_sides, size_type b_stride,
        size_type c_stride, const size_type *__restrict__ slice_lengths,
        const size_type *__restrict__ slice_sets,
        const ValueType *__restrict__ alpha, const ValueType *__restrict__ a,
        const IndexType *__restrict__ col, const ValueType *__restrict__ b,
        const ValueType *__restrict__ beta, ValueType *__restrict__ c)
{
    const auto slice_id = blockIdx.x;
    const auto slice_size = blockDim.x;
    const auto row_in_slice = threadIdx.x;
    const auto global_row =
        static_cast<size_type>(slice_size) * slice_id + row_in_slice;
    const auto column_id = blockIdx.y;
    ValueType val = 0;
    IndexType ind = 0;
    if (global_row < num_rows && column_id < num_right_hand_sides) {
        for (size_type i = 0; i < slice_lengths[slice_id]; i++) {
            ind = row_in_slice + (slice_sets[slice_id] + i) * slice_size;
            val += alpha[0] * a[ind] * b[col[ind] * b_stride + column_id];
        }
        c[global_row * c_stride + column_id] =
            beta[0] * c[global_row * c_stride + column_id] + val;
    }
}


}  // namespace


namespace kernel {


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void initialize_zero_dense(
    size_type num_rows, size_type num_cols, size_type stride,
    ValueType *__restrict__ result)
{
    const auto tidx_x = threadIdx.x + blockDim.x * blockIdx.x;
    const auto tidx_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (tidx_x < num_cols && tidx_y < num_rows) {
        result[tidx_y * stride + tidx_x] = zero<ValueType>();
    }
}


template <unsigned int threads_per_row, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_dense(
    size_type num_rows, size_type num_cols, size_type stride,
    size_type slice_size, const size_type *__restrict__ slice_lengths,
    const size_type *__restrict__ slice_sets,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values, ValueType *__restrict__ result)
{
    const auto global_row =
        (blockDim.x * blockIdx.x + threadIdx.x) / threads_per_row;
    const auto row = global_row % slice_size;
    const auto slice = global_row / slice_size;
    const auto start_index = threadIdx.x % threads_per_row;

    if (global_row < num_rows) {
        for (auto i = start_index; i < slice_lengths[slice];
             i += threads_per_row) {
            if (values[(slice_sets[slice] + i) * slice_size + row] !=
                zero<ValueType>()) {
                result[global_row * stride +
                       col_idxs[(slice_sets[slice] + i) * slice_size + row]] =
                    values[(slice_sets[slice] + i) * slice_size + row];
            }
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void count_nnz_per_row(
    size_type num_rows, size_type slice_size,
    const size_type *__restrict__ slice_sets,
    const ValueType *__restrict__ values, IndexType *__restrict__ result)
{
    constexpr auto warp_size = config::warp_size;
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto row_idx = tidx / warp_size;
    const auto slice_id = row_idx / slice_size;
    const auto tid_in_warp = tidx % warp_size;
    const auto row_in_slice = row_idx % slice_size;

    if (row_idx < num_rows) {
        IndexType part_result{};
        for (size_type sellp_ind =
                 (slice_sets[slice_id] + tid_in_warp) * slice_size +
                 row_in_slice;
             sellp_ind < slice_sets[slice_id + 1] * slice_size;
             sellp_ind += warp_size * slice_size) {
            if (values[sellp_ind] != zero<ValueType>()) {
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
    size_type num_rows, size_type slice_size,
    const size_type *__restrict__ source_slice_sets,
    const IndexType *__restrict__ source_col_idxs,
    const ValueType *__restrict__ source_values,
    IndexType *__restrict__ result_row_ptrs,
    IndexType *__restrict__ result_col_idxs,
    ValueType *__restrict__ result_values)
{
    const auto row = threadIdx.x + blockIdx.x * blockDim.x;
    const auto slice_id = row / slice_size;
    const auto row_in_slice = row % slice_size;

    if (row < num_rows) {
        size_type csr_ind = result_row_ptrs[row];
        for (size_type sellp_ind =
                 source_slice_sets[slice_id] * slice_size + row_in_slice;
             sellp_ind < source_slice_sets[slice_id + 1] * slice_size;
             sellp_ind += slice_size) {
            if (source_values[sellp_ind] != zero<ValueType>()) {
                result_values[csr_ind] = source_values[sellp_ind];
                result_col_idxs[csr_ind] = source_col_idxs[sellp_ind];
                csr_ind++;
            }
        }
    }
}


}  // namespace kernel