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

#include "common_gmres_kernels.hpp.inc"


// Must be called with at least `num_rows * num_rhs` threads in total.
template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void initialize_2_2_kernel(
    size_type num_rows, size_type num_rhs,
    const ValueType *__restrict__ residual, size_type stride_residual,
    const remove_complex<ValueType> *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norm_collection,
    ValueType *__restrict__ krylov_bases, size_type stride_krylov,
    size_type *__restrict__ final_iter_nums)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_idx = global_id / num_rhs;
    const auto col_idx = global_id % num_rhs;

    if (global_id < num_rhs) {
        residual_norm_collection[global_id] = residual_norm[global_id];
        final_iter_nums[global_id] = 0;
    }

    if (row_idx < num_rows && col_idx < num_rhs) {
        auto value = residual[row_idx * stride_residual + col_idx] /
                     residual_norm[col_idx];
        krylov_bases[row_idx * stride_krylov + col_idx] = value;
    }
}


__global__
    __launch_bounds__(default_block_size) void increase_final_iteration_numbers_kernel(
        size_type *__restrict__ final_iter_nums,
        const stopping_status *__restrict__ stop_status, size_type total_number)
{
    const auto global_id = thread::get_thread_id_flat();
    if (global_id < total_number) {
        final_iter_nums[global_id] += !stop_status[global_id].has_stopped();
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_dot_size) void multidot_kernel(
    size_type k, size_type num_rows, size_type num_cols,
    const ValueType *__restrict__ krylov_bases,
    const ValueType *__restrict__ next_krylov_basis, size_type stride_krylov,
    ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto tidx = threadIdx.x;
    const auto tidy = threadIdx.y;
    const auto col_idx = blockIdx.x * default_dot_dim + tidx;
    const auto num = ceildiv(num_rows, gridDim.y);
    const auto start_row = blockIdx.y * num;
    const auto end_row =
        ((blockIdx.y + 1) * num > num_rows) ? num_rows : (blockIdx.y + 1) * num;
    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`
    __shared__
        UninitializedArray<ValueType, default_dot_dim *(default_dot_dim + 1)>
            reduction_helper_array;
    ValueType *__restrict__ reduction_helper = reduction_helper_array;

    ValueType local_res = zero<ValueType>();
    if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
        for (size_type i = start_row + tidy; i < end_row;
             i += default_dot_dim) {
            const auto krylov_idx = i * stride_krylov + col_idx;
            local_res +=
                conj(krylov_bases[krylov_idx]) * next_krylov_basis[krylov_idx];
        }
    }
    reduction_helper[tidx * (default_dot_dim + 1) + tidy] = local_res;
    __syncthreads();
    local_res = reduction_helper[tidy * (default_dot_dim + 1) + tidx];
    const auto tile_block =
        group::tiled_partition<default_dot_dim>(group::this_thread_block());
    const auto sum =
        reduce(tile_block, local_res,
               [](const ValueType &a, const ValueType &b) { return a + b; });
    const auto new_col_idx = blockIdx.x * default_dot_dim + tidy;
    if (tidx == 0 && new_col_idx < num_cols &&
        !stop_status[new_col_idx].has_stopped()) {
        const auto hessenberg_idx = k * stride_hessenberg + new_col_idx;
        atomic_add(hessenberg_iter + hessenberg_idx, sum);
    }
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void update_next_krylov_kernel(
    size_type k, size_type num_rows, size_type num_cols,
    const ValueType *__restrict__ krylov_bases,
    ValueType *__restrict__ next_krylov_basis, size_type stride_krylov,
    const ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_idx = global_id / stride_krylov;
    const auto col_idx = global_id % stride_krylov;

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_krylov + col_idx;
        const auto krylov_idx = row_idx * stride_krylov + col_idx;
        const auto hessenberg_idx = k * stride_hessenberg + col_idx;

        next_krylov_basis[next_krylov_idx] -=
            hessenberg_iter[hessenberg_idx] * krylov_bases[krylov_idx];
    }
}


// Must be called with at least `num_cols` blocks, each with `block_size`
// threads. `block_size` must be a power of 2.
template <int block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void update_hessenberg_2_kernel(
    size_type iter, size_type num_rows, size_type num_cols,
    const ValueType *__restrict__ next_krylov_basis,
    size_type stride_next_krylov, ValueType *__restrict__ hessenberg_iter,
    size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto tidx = threadIdx.x;
    const auto col_idx = blockIdx.x;

    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`
    __shared__ UninitializedArray<ValueType, block_size> reduction_helper_array;
    ValueType *__restrict__ reduction_helper = reduction_helper_array;

    if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
        ValueType local_res{};
        for (size_type i = tidx; i < num_rows; i += block_size) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            const auto next_krylov_value = next_krylov_basis[next_krylov_idx];

            local_res += next_krylov_value * next_krylov_value;
        }

        reduction_helper[tidx] = local_res;

        // Perform thread block reduction. Result is in reduction_helper[0]
        reduce(group::this_thread_block(), reduction_helper,
               [](const ValueType &a, const ValueType &b) { return a + b; });

        if (tidx == 0) {
            hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx] =
                sqrt(reduction_helper[0]);
        }
    }
}


// Must be called with at least `num_rows * stride_krylov` threads in
// total.
template <int block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void update_krylov_kernel(
    size_type iter, size_type num_rows, size_type num_cols,
    ValueType *__restrict__ krylov_bases, size_type stride_krylov,
    const ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_idx = global_id / stride_krylov;
    const auto col_idx = global_id % stride_krylov;
    const auto hessenberg =
        hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx];

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto krylov_idx = row_idx * stride_krylov + col_idx;

        krylov_bases[krylov_idx] /= hessenberg;
    }
}


// Must be called with at least `stride_preconditioner * num_rows` threads in
// total.
template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void calculate_Qy_kernel(
    size_type num_rows, size_type num_cols, size_type num_rhs,
    const ValueType *__restrict__ krylov_bases, size_type stride_krylov,
    const ValueType *__restrict__ y, size_type stride_y,
    ValueType *__restrict__ before_preconditioner,
    size_type stride_preconditioner,
    const size_type *__restrict__ final_iter_nums)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_id = global_id / stride_preconditioner;
    const auto col_id = global_id % stride_preconditioner;

    if (row_id < num_rows && col_id < num_cols) {
        ValueType temp = zero<ValueType>();

        for (size_type j = 0; j < final_iter_nums[col_id]; ++j) {
            temp +=
                krylov_bases[(row_id + j * num_rows) * stride_krylov + col_id] *
                y[j * stride_y + col_id];
        }
        before_preconditioner[global_id] = temp;
    }
}
