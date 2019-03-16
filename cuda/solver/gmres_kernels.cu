/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/solver/gmres_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "cuda/base/cublas_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace gmres {


constexpr int default_block_size = 512;


// Must be called with at least `max(stride_b * num_rows, krylov_dim *
// num_cols)` threads in total.
template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void initialize_1_kernel(
    size_type num_rows, size_type num_cols, size_type krylov_dim,
    const ValueType *__restrict__ b, size_type stride_b,
    ValueType *__restrict__ residual, size_type stride_residual,
    ValueType *__restrict__ givens_sin, size_type stride_sin,
    ValueType *__restrict__ givens_cos, size_type stride_cos,
    stopping_status *__restrict__ stop_status)
{
    const auto global_id = blockIdx.x * blockDim.x + threadIdx.x;

    const auto row_idx = global_id / stride_b;
    const auto col_idx = global_id % stride_b;

    if (global_id < num_cols) {
        stop_status[global_id].reset();
    }

    if (row_idx < num_rows && col_idx < num_cols) {
        residual[row_idx * stride_residual + col_idx] =
            b[row_idx * stride_b + col_idx];
    }

    if (global_id < krylov_dim * num_cols) {
        const auto row_givens = global_id / num_cols;
        const auto col_givens = global_id % num_cols;

        givens_sin[row_givens * stride_sin + col_givens] = zero<ValueType>();
        givens_cos[row_givens * stride_cos + col_givens] = zero<ValueType>();
    }
}


template <typename ValueType>
void initialize_1(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *b_norm,
                  matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *givens_sin,
                  matrix::Dense<ValueType> *givens_cos,
                  Array<stopping_status> *stop_status, size_type krylov_dim)
{
    const auto num_threads = std::max(b->get_size()[0] * b->get_stride(),
                                      krylov_dim * b->get_size()[1]);
    const dim3 grid_dim(ceildiv(num_threads, default_block_size), 1, 1);
    const dim3 block_dim(default_block_size, 1, 1);
    constexpr auto block_size = default_block_size;

    b->compute_norm2(b_norm);
    initialize_1_kernel<block_size><<<grid_dim, block_dim>>>(
        b->get_size()[0], b->get_size()[1], krylov_dim,
        as_cuda_type(b->get_const_values()), b->get_stride(),
        as_cuda_type(residual->get_values()), residual->get_stride(),
        as_cuda_type(givens_sin->get_values()), givens_sin->get_stride(),
        as_cuda_type(givens_cos->get_values()), givens_cos->get_stride(),
        as_cuda_type(stop_status->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_1_KERNEL);


// Must be called with at least `num_rows * stride_krylov` threads in total.
template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void initialize_2_1_kernel(
    size_type num_rows, size_type num_rhs, size_type krylov_dim,
    ValueType *__restrict__ krylov_bases, size_type stride_krylov,
    ValueType *__restrict__ residual_norm_collection,
    size_type stride_residual_nc)
{
    const auto global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row_idx = global_id / stride_krylov;
    const auto col_idx = global_id % stride_krylov;

    if (row_idx < num_rows && col_idx < (krylov_dim + 1) * num_rhs) {
        krylov_bases[row_idx * stride_krylov + col_idx] = zero<ValueType>();
    }

    if (row_idx < krylov_dim + 1 && col_idx < num_rhs) {
        residual_norm_collection[row_idx * stride_residual_nc + col_idx] =
            zero<ValueType>();
    }
}


// Must be called with at least `num_rows * num_rhs` threads in total.
template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void initialize_2_2_kernel(
    size_type num_rows, size_type num_rhs,
    const ValueType *__restrict__ residual, size_type stride_residual,
    const ValueType *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norm_collection,
    ValueType *__restrict__ krylov_bases, size_type stride_krylov,
    size_type *__restrict__ final_iter_nums)
{
    const auto global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row_idx = global_id / num_rhs;
    const auto col_idx = global_id % num_rhs;

    if (global_id < num_rhs) {
        residual_norm_collection[global_id] = residual_norm[global_id];
        final_iter_nums[global_id] = 0;
    }

    if (row_idx < num_rows && col_idx < num_rhs) {
        krylov_bases[row_idx * stride_krylov + col_idx] =
            residual[row_idx * stride_residual + col_idx] /
            residual_norm[col_idx];
    }
}


template <typename ValueType>
void initialize_2(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *residual_norm,
                  matrix::Dense<ValueType> *residual_norm_collection,
                  matrix::Dense<ValueType> *krylov_bases,
                  Array<size_type> *final_iter_nums, size_type krylov_dim)
{
    const auto num_rows = residual->get_size()[0];
    const auto num_rhs = residual->get_size()[1];
    const dim3 grid_dim_1(
        ceildiv(num_rows * krylov_bases->get_stride(), default_block_size), 1,
        1);
    const dim3 block_dim(default_block_size, 1, 1);
    constexpr auto block_size = default_block_size;

    initialize_2_1_kernel<block_size><<<grid_dim_1, block_dim>>>(
        residual->get_size()[0], residual->get_size()[1], krylov_dim,
        as_cuda_type(krylov_bases->get_values()), krylov_bases->get_stride(),
        as_cuda_type(residual_norm_collection->get_values()),
        residual_norm_collection->get_stride());
    residual->compute_norm2(residual_norm);

    const dim3 grid_dim_2(ceildiv(num_rows * num_rhs, default_block_size), 1,
                          1);
    initialize_2_2_kernel<block_size><<<grid_dim_2, block_dim>>>(
        residual->get_size()[0], residual->get_size()[1],
        as_cuda_type(residual->get_const_values()), residual->get_stride(),
        as_cuda_type(residual_norm->get_const_values()),
        as_cuda_type(residual_norm_collection->get_values()),
        as_cuda_type(krylov_bases->get_values()), krylov_bases->get_stride(),
        as_cuda_type(final_iter_nums->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_2_KERNEL);


__global__
    __launch_bounds__(default_block_size) void increase_final_iteration_numbers_kernel(
        size_type *__restrict__ final_iter_nums,
        const stopping_status *__restrict__ stop_status, size_type total_number)
{
    const auto global_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_id < total_number) {
        final_iter_nums[global_id] +=
            (1 - stop_status[global_id].has_stopped());
    }
}


// Must be called with at least `num_cols` blocks, each with `block_size`
// threads. `block_size` must be a power of 2.
template <int block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void update_hessenberg_kernel(
    size_type k, size_type num_rows, size_type num_cols,
    const ValueType *__restrict__ next_krylov_basis,
    size_type stride_next_krylov, const ValueType *__restrict__ krylov_bases,
    size_type stride_krylov, ValueType *__restrict__ hessenberg_iter,
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
        const auto krylov_col = k * num_cols + col_idx;
        for (size_type i = tidx; i < num_rows; i += block_size) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            const auto krylov_idx = i * stride_krylov + krylov_col;

            local_res +=
                next_krylov_basis[next_krylov_idx] * krylov_bases[krylov_idx];
        }

        reduction_helper[tidx] = local_res;

        // Perform thread block reduction. Result is in reduction_helper[0]
        reduce(group::this_thread_block(), reduction_helper,
               [](const ValueType &a, const ValueType &b) { return a + b; });

        if (tidx == 0) {
            const auto hessenberg_idx = k * stride_hessenberg + col_idx;
            hessenberg_iter[hessenberg_idx] = reduction_helper[0];
        }
    }
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void update_next_krylov_kernel(
    size_type k, size_type num_rows, size_type num_cols,
    ValueType *__restrict__ next_krylov_basis, size_type stride_next_krylov,
    const ValueType *__restrict__ krylov_bases, size_type stride_krylov,
    const ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row_idx = global_id / stride_next_krylov;
    const auto col_idx = global_id % stride_next_krylov;

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
        const auto krylov_idx =
            row_idx * stride_krylov + k * num_cols + col_idx;
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


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void update_krylov_next_krylov_kernel(
    size_type iter, size_type num_rows, size_type num_cols,
    ValueType *__restrict__ next_krylov_basis, size_type stride_next_krylov,
    ValueType *__restrict__ krylov_bases, size_type stride_krylov,
    const ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto global_id = threadIdx.x + blockIdx.x * blockDim.x;
    const auto row_idx = global_id / stride_next_krylov;
    const auto col_idx = global_id % stride_next_krylov;
    const auto hessenberg =
        hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx];

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
        const auto krylov_idx =
            row_idx * stride_krylov + num_cols * (iter + 1) + col_idx;

        const auto next_krylov_value =
            next_krylov_basis[next_krylov_idx] / hessenberg;

        next_krylov_basis[next_krylov_idx] = next_krylov_value;
        krylov_bases[krylov_idx] = next_krylov_value;
    }
}


template <typename ValueType>
void finish_arnoldi(std::shared_ptr<const CudaExecutor> exec,
                    matrix::Dense<ValueType> *next_krylov_basis,
                    matrix::Dense<ValueType> *krylov_bases,
                    matrix::Dense<ValueType> *hessenberg_iter, size_type iter,
                    const stopping_status *stop_status)
{
    const auto stride_next_krylov = next_krylov_basis->get_stride();
    const auto stride_krylov = krylov_bases->get_stride();
    const auto stride_hessenberg = hessenberg_iter->get_stride();
    const auto dim_size = next_krylov_basis->get_size();

    for (size_type k = 0; k < iter + 1; ++k) {
        update_hessenberg_kernel<default_block_size>
            <<<dim_size[1], default_block_size>>>(
                k, dim_size[0], dim_size[1],
                as_cuda_type(next_krylov_basis->get_const_values()),
                stride_next_krylov,
                as_cuda_type(krylov_bases->get_const_values()), stride_krylov,
                as_cuda_type(hessenberg_iter->get_values()), stride_hessenberg,
                as_cuda_type(stop_status));

        update_next_krylov_kernel<default_block_size>
            <<<ceildiv(dim_size[0] * stride_next_krylov, default_block_size),
               default_block_size>>>(
                k, dim_size[0], dim_size[1],
                as_cuda_type(next_krylov_basis->get_values()),
                stride_next_krylov,
                as_cuda_type(krylov_bases->get_const_values()), stride_krylov,
                as_cuda_type(hessenberg_iter->get_const_values()),
                stride_hessenberg, as_cuda_type(stop_status));
    }
    // for i in 1:iter
    //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
    //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
    // end


    update_hessenberg_2_kernel<default_block_size>
        <<<dim_size[1], default_block_size>>>(
            iter, dim_size[0], dim_size[1],
            as_cuda_type(next_krylov_basis->get_const_values()),
            stride_next_krylov, as_cuda_type(hessenberg_iter->get_values()),
            stride_hessenberg, as_cuda_type(stop_status));

    update_krylov_next_krylov_kernel<default_block_size>
        <<<ceildiv(dim_size[0] * stride_next_krylov, default_block_size),
           default_block_size>>>(
            iter, dim_size[0], dim_size[1],
            as_cuda_type(next_krylov_basis->get_values()), stride_next_krylov,
            as_cuda_type(krylov_bases->get_values()), stride_krylov,
            as_cuda_type(hessenberg_iter->get_const_values()),
            stride_hessenberg, as_cuda_type(stop_status));
    // next_krylov_basis /= hessenberg(iter, iter + 1)
    // krylov_bases(:, iter + 1) = next_krylov_basis
    // End of arnoldi
}


template <typename ValueType>
__device__ void calculate_sin_and_cos_kernel(
    size_type col_idx, size_type num_cols, size_type iter,
    const ValueType *hessenberg_iter, size_type stride_hessenberg,
    ValueType *givens_sin, size_type stride_sin, ValueType *givens_cos,
    size_type stride_cos)
{
    if (hessenberg_iter[iter * stride_hessenberg + col_idx] ==
        zero<ValueType>()) {
        givens_cos[iter * stride_cos + col_idx] = zero<ValueType>();
        givens_sin[iter * stride_sin + col_idx] = one<ValueType>();
    } else {
        auto hypotenuse =
            sqrt(hessenberg_iter[iter * stride_hessenberg + col_idx] *
                     hessenberg_iter[iter * stride_hessenberg + col_idx] +
                 hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx] *
                     hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx]);
        givens_cos[iter * stride_cos + col_idx] =
            abs(hessenberg_iter[iter * stride_hessenberg + col_idx]) /
            hypotenuse;
        givens_sin[iter * stride_sin + col_idx] =
            givens_cos[iter * stride_cos + col_idx] *
            hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx] /
            hessenberg_iter[iter * stride_hessenberg + col_idx];
    }
}


template <typename ValueType>
__device__ void calculate_residual_norm_kernel(
    size_type col_idx, size_type num_cols, size_type iter,
    const ValueType *givens_sin, size_type stride_sin,
    const ValueType *givens_cos, size_type stride_cos, ValueType *residual_norm,
    ValueType *residual_norm_collection,
    size_type stride_residual_norm_collection, const ValueType *b_norm)
{
    residual_norm_collection[(iter + 1) * stride_residual_norm_collection +
                             col_idx] =
        -givens_sin[iter * stride_sin + col_idx] *
        residual_norm_collection[iter * stride_residual_norm_collection +
                                 col_idx];
    residual_norm_collection[iter * stride_residual_norm_collection + col_idx] =
        givens_cos[iter * stride_cos + col_idx] *
        residual_norm_collection[iter * stride_residual_norm_collection +
                                 col_idx];
    residual_norm[col_idx] =
        abs(residual_norm_collection[(iter + 1) *
                                         stride_residual_norm_collection +
                                     col_idx]) /
        b_norm[col_idx];
}


// Must be called with at least `num_cols` threads in total.
template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void givens_rotation_kernel(
    size_type num_rows, size_type num_cols, size_type iter,
    ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    ValueType *__restrict__ givens_sin, size_type stride_sin,
    ValueType *__restrict__ givens_cos, size_type stride_cos,
    ValueType *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norm_collection,
    size_type stride_residual_norm_collection,
    const ValueType *__restrict__ b_norm,
    const stopping_status *__restrict__ stop_status)
{
    const auto col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (col_idx >= num_cols || stop_status[col_idx].has_stopped()) {
        return;
    }

    const auto current_thread_block = group::this_thread_block();

    for (size_type i = 0; i < iter; ++i) {
        const auto tmp =
            givens_cos[i * stride_cos + col_idx] *
                hessenberg_iter[i * stride_hessenberg + col_idx] +
            givens_sin[i * stride_sin + col_idx] *
                hessenberg_iter[(i + 1) * stride_hessenberg + col_idx];
        current_thread_block.sync();
        hessenberg_iter[(i + 1) * stride_hessenberg + col_idx] =
            givens_cos[i * stride_cos + col_idx] *
                hessenberg_iter[(i + 1) * stride_hessenberg + col_idx] -
            givens_sin[i * stride_sin + col_idx] *
                hessenberg_iter[i * stride_hessenberg + col_idx];
        hessenberg_iter[i * stride_hessenberg + col_idx] = tmp;
        current_thread_block.sync();
    }
    // for j in 1:iter - 1
    //     temp             =  cos(j)*hessenberg(j) +
    //                         sin(j)*hessenberg(j+1)
    //     hessenberg(j+1)  = -sin(j)*hessenberg(j) +
    //                         cos(j)*hessenberg(j+1)
    //     hessenberg(j)    =  temp;
    // end

    calculate_sin_and_cos_kernel(col_idx, num_cols, iter, hessenberg_iter,
                                 stride_hessenberg, givens_sin, stride_sin,
                                 givens_cos, stride_cos);
    // Calculate sin and cos

    hessenberg_iter[iter * stride_hessenberg + col_idx] =
        givens_cos[iter * stride_cos + col_idx] *
            hessenberg_iter[iter * stride_hessenberg + col_idx] +
        givens_sin[iter * stride_sin + col_idx] *
            hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx];
    hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx] =
        zero<ValueType>();
    // hessenberg(iter)   = cos(iter)*hessenberg(iter) +
    //                      sin(iter)*hessenberg(iter)
    // hessenberg(iter+1) = 0

    calculate_residual_norm_kernel(col_idx, num_cols, iter, givens_sin,
                                   stride_sin, givens_cos, stride_cos,
                                   residual_norm, residual_norm_collection,
                                   stride_residual_norm_collection, b_norm);
    // Calculate residual norm
}


template <typename ValueType>
void givens_rotation(std::shared_ptr<const CudaExecutor> exec,
                     matrix::Dense<ValueType> *givens_sin,
                     matrix::Dense<ValueType> *givens_cos,
                     matrix::Dense<ValueType> *hessenberg_iter,
                     matrix::Dense<ValueType> *residual_norm,
                     matrix::Dense<ValueType> *residual_norm_collection,
                     const matrix::Dense<ValueType> *b_norm, size_type iter,
                     const Array<stopping_status> *stop_status)
{
    // TODO: tune block_size for optimal performance
    constexpr auto block_size = default_block_size;
    const auto num_cols = hessenberg_iter->get_size()[1];
    const dim3 block_dim{block_size, 1, 1};
    const dim3 grid_dim{
        static_cast<unsigned int>(ceildiv(num_cols, block_size)), 1, 1};

    givens_rotation_kernel<block_size><<<grid_dim, block_dim>>>(
        hessenberg_iter->get_size()[0], hessenberg_iter->get_size()[1], iter,
        as_cuda_type(hessenberg_iter->get_values()),
        hessenberg_iter->get_stride(), as_cuda_type(givens_sin->get_values()),
        givens_sin->get_stride(), as_cuda_type(givens_cos->get_values()),
        givens_cos->get_stride(), as_cuda_type(residual_norm->get_values()),
        as_cuda_type(residual_norm_collection->get_values()),
        residual_norm_collection->get_stride(),
        as_cuda_type(b_norm->get_const_values()),
        as_cuda_type(stop_status->get_const_data()));
}


template <typename ValueType>
void step_1(std::shared_ptr<const CudaExecutor> exec,
            matrix::Dense<ValueType> *next_krylov_basis,
            matrix::Dense<ValueType> *givens_sin,
            matrix::Dense<ValueType> *givens_cos,
            matrix::Dense<ValueType> *residual_norm,
            matrix::Dense<ValueType> *residual_norm_collection,
            matrix::Dense<ValueType> *krylov_bases,
            matrix::Dense<ValueType> *hessenberg_iter,
            const matrix::Dense<ValueType> *b_norm, size_type iter,
            Array<size_type> *final_iter_nums,
            const Array<stopping_status> *stop_status)
{
    increase_final_iteration_numbers_kernel<<<
        static_cast<unsigned int>(
            ceildiv(final_iter_nums->get_num_elems(), default_block_size)),
        default_block_size>>>(as_cuda_type(final_iter_nums->get_data()),
                              as_cuda_type(stop_status->get_const_data()),
                              final_iter_nums->get_num_elems());
    finish_arnoldi(exec, next_krylov_basis, krylov_bases, hessenberg_iter, iter,
                   stop_status->get_const_data());
    givens_rotation(exec, givens_sin, givens_cos, hessenberg_iter,
                    residual_norm, residual_norm_collection, b_norm, iter,
                    stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_1_KERNEL);


// Must be called with at least `num_rhs` threads in total.
template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void solve_upper_triangular_kernel(
    size_type num_cols, size_type num_rhs,
    const ValueType *__restrict__ residual_norm_collection,
    size_type stride_residual_norm_collection,
    const ValueType *__restrict__ hessenberg, size_type stride_hessenberg,
    ValueType *__restrict__ y, size_type stride_y,
    const size_type *__restrict__ final_iter_nums)
{
    const auto col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (col_idx >= num_rhs) {
        return;
    }

    for (int i = final_iter_nums[col_idx] - 1; i >= 0; --i) {
        auto temp =
            residual_norm_collection[i * stride_residual_norm_collection +
                                     col_idx];
        for (size_type j = i + 1; j < final_iter_nums[col_idx]; ++j) {
            temp -= hessenberg[i * stride_hessenberg + j * num_rhs + col_idx] *
                    y[j * stride_y + col_idx];
        }

        y[i * stride_y + col_idx] =
            temp / hessenberg[i * stride_hessenberg + i * num_rhs + col_idx];
    }
    // Solve upper triangular.
    // y = hessenberg \ residual_norm_collection
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
    const auto global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row_id = global_id / stride_preconditioner;
    const auto col_id = global_id % stride_preconditioner;

    if (row_id < num_rows && col_id < num_cols) {
        before_preconditioner[global_id] = zero<ValueType>();
        for (size_type j = 0; j < final_iter_nums[col_id]; ++j) {
            before_preconditioner[global_id] +=
                krylov_bases[row_id * stride_krylov + j * num_rhs + col_id] *
                y[j * stride_y + col_id];
        }
    }
}


template <typename ValueType>
void solve_upper_triangular(
    const matrix::Dense<ValueType> *residual_norm_collection,
    const matrix::Dense<ValueType> *hessenberg, matrix::Dense<ValueType> *y,
    const Array<size_type> *final_iter_nums)
{
    // TODO: tune block_size for optimal performance
    constexpr auto block_size = default_block_size;
    const auto num_rhs = residual_norm_collection->get_size()[1];
    const dim3 block_dim{block_size, 1, 1};
    const dim3 grid_dim{static_cast<unsigned int>(ceildiv(num_rhs, block_size)),
                        1, 1};

    solve_upper_triangular_kernel<block_size><<<grid_dim, block_dim>>>(
        hessenberg->get_size()[1], num_rhs,
        as_cuda_type(residual_norm_collection->get_const_values()),
        residual_norm_collection->get_stride(),
        as_cuda_type(hessenberg->get_const_values()), hessenberg->get_stride(),
        as_cuda_type(y->get_values()), y->get_stride(),
        as_cuda_type(final_iter_nums->get_const_data()));
}


template <typename ValueType>
void solve_x(std::shared_ptr<const CudaExecutor> exec,
             const matrix::Dense<ValueType> *krylov_bases,
             const matrix::Dense<ValueType> *y, matrix::Dense<ValueType> *x,
             const Array<size_type> *final_iter_nums,
             const LinOp *preconditioner)
{
    // The CUDA 10.1 compiler does not compile
    // `matrix::Dense<ValueType>::create_with_config_of(x)`,
    // therefore, the `create` method is directly called.
    auto before_preconditioner = matrix::Dense<ValueType>::create(
        x->get_executor(), x->get_size(), x->get_stride());
    auto after_preconditioner = matrix::Dense<ValueType>::create(
        x->get_executor(), x->get_size(), x->get_stride());

    const auto num_rows = before_preconditioner->get_size()[0];
    const auto num_cols = krylov_bases->get_size()[1];
    const auto num_rhs = before_preconditioner->get_size()[1];
    const auto stride_before_preconditioner =
        before_preconditioner->get_stride();

    constexpr auto block_size = default_block_size;
    const dim3 grid_dim{
        static_cast<unsigned int>(
            ceildiv(num_rows * stride_before_preconditioner, block_size)),
        1, 1};
    const dim3 block_dim{block_size, 1, 1};


    calculate_Qy_kernel<block_size><<<grid_dim, block_dim>>>(
        num_rows, num_cols, num_rhs,
        as_cuda_type(krylov_bases->get_const_values()),
        krylov_bases->get_stride(), as_cuda_type(y->get_const_values()),
        y->get_stride(), as_cuda_type(before_preconditioner->get_values()),
        stride_before_preconditioner,
        as_cuda_type(final_iter_nums->get_const_data()));

    preconditioner->apply(before_preconditioner.get(),
                          after_preconditioner.get());

    auto one_op =
        initialize<matrix::Dense<ValueType>>({one<ValueType>()}, exec);
    x->add_scaled(one_op.get(), after_preconditioner.get());
    // Solve x
    // x = x + preconditioner_ * krylov_bases * y
}


template <typename ValueType>
void step_2(std::shared_ptr<const CudaExecutor> exec,
            const matrix::Dense<ValueType> *residual_norm_collection,
            const matrix::Dense<ValueType> *krylov_bases,
            const matrix::Dense<ValueType> *hessenberg,
            matrix::Dense<ValueType> *y, matrix::Dense<ValueType> *x,
            const Array<size_type> *final_iter_nums,
            const LinOp *preconditioner)
{
    solve_upper_triangular(residual_norm_collection, hessenberg, y,
                           final_iter_nums);
    solve_x(exec, krylov_bases, y, x, final_iter_nums, preconditioner);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_2_KERNEL);


}  // namespace gmres
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
