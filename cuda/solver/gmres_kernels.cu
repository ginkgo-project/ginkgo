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


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "cuda/base/cublas_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/reduction.cuh"
#include "cuda/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace gmres {


constexpr int default_block_size = 512;


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void initialize_1_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    size_type krylov_dim, const ValueType *__restrict__ b,
    ValueType *__restrict__ b_norm, ValueType *__restrict__ residual,
    ValueType *__restrict__ givens_sin, ValueType *__restrict__ givens_cos,
    stopping_status *__restrict__ stop_status)
{
    constexpr auto warps_per_block = block_size / cuda_config::warp_size;
    const auto global_id =
        thread::get_thread_id<cuda_config::warp_size, warps_per_block>();

    if (global_id < num_cols) {
        stop_status[global_id].reset();
    }

    if (global_id < num_rows * stride) {
        residual[global_id] = b[global_id];
    }

    if (global_id < krylov_dim * num_cols) {
        givens_sin[global_id] = zero<ValueType>();
        givens_cos[global_id] = zero<ValueType>();
    }
}


template <typename ValueType>
void initialize_1(
    std::shared_ptr<const CudaExecutor> exec, const matrix::Dense<ValueType> *b,
    matrix::Dense<ValueType> *b_norm, matrix::Dense<ValueType> *residual,
    matrix::Dense<ValueType> *givens_sin, matrix::Dense<ValueType> *givens_cos,
    Array<stopping_status> *stop_status, const size_type krylov_dim)
{
    const dim3 grid_dim(
        ceildiv(b->get_size()[0], default_block_size) * b->get_stride(), 1, 1);
    const dim3 block_dim(default_block_size, 1, 1);
    constexpr auto block_size = default_block_size;

    b->compute_norm2(b_norm);
    initialize_1_kernel<block_size><<<grid_dim, block_dim>>>(
        b->get_size()[0], b->get_size()[1], b->get_stride(), krylov_dim,
        as_cuda_type(b->get_const_values()), as_cuda_type(b_norm->get_values()),
        as_cuda_type(residual->get_values()),
        as_cuda_type(givens_sin->get_values()),
        as_cuda_type(givens_cos->get_values()),
        as_cuda_type(stop_status->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_1_KERNEL);


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void initialize_2_1_kernel(
    size_type num_rows, size_type num_cols, size_type krylov_dim,
    ValueType *__restrict__ residual_norm_collection,
    ValueType *__restrict__ krylov_bases)
{
    constexpr auto warps_per_block = block_size / cuda_config::warp_size;
    const auto global_id =
        thread::get_thread_id<cuda_config::warp_size, warps_per_block>();

    if (global_id < num_rows * (krylov_dim + 1) * num_cols) {
        krylov_bases[global_id] = 0;
    }

    if (global_id < (krylov_dim + 1) * num_cols) {
        residual_norm_collection[global_id] = 0;
    }
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void initialize_2_2_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ residual,
    const ValueType *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norm_collection,
    ValueType *__restrict__ krylov_bases,
    size_type *__restrict__ final_iter_nums)
{
    constexpr auto warps_per_block = block_size / cuda_config::warp_size;
    const auto global_id =
        thread::get_thread_id<cuda_config::warp_size, warps_per_block>();
    const auto row = ceildiv(global_id + 1, num_cols) - 1;
    const auto column = global_id - row * num_cols;

    if (global_id < num_cols) {
        residual_norm_collection[global_id] = residual_norm[global_id];
        final_iter_nums[global_id] = 0;
    }

    if (global_id < num_rows * num_cols) {
        krylov_bases[row * stride + column] =
            residual[global_id] / residual_norm[column];
    }
}


template <typename ValueType>
void initialize_2(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *residual_norm,
                  matrix::Dense<ValueType> *residual_norm_collection,
                  matrix::Dense<ValueType> *krylov_bases,
                  Array<size_type> *final_iter_nums, const size_type krylov_dim)
{
    const dim3 grid_dim(
        ceildiv(krylov_bases->get_size()[0], default_block_size) *
            krylov_bases->get_stride(),
        1, 1);
    const dim3 block_dim(default_block_size, 1, 1);
    constexpr auto block_size = default_block_size;

    initialize_2_1_kernel<block_size><<<grid_dim, block_dim>>>(
        residual->get_size()[0], residual->get_size()[1], krylov_dim,
        as_cuda_type(residual_norm_collection->get_values()),
        as_cuda_type(krylov_bases->get_values()));
    residual->compute_norm2(residual_norm);
    initialize_2_2_kernel<block_size><<<grid_dim, block_dim>>>(
        residual->get_size()[0], residual->get_size()[1],
        krylov_bases->get_stride(), as_cuda_type(residual->get_const_values()),
        as_cuda_type(residual_norm->get_const_values()),
        as_cuda_type(residual_norm_collection->get_values()),
        as_cuda_type(krylov_bases->get_values()),
        as_cuda_type(final_iter_nums->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_2_KERNEL);


__global__
    __launch_bounds__(default_block_size) void increase_final_iteration_numbers_kernel(
        size_type *__restrict__ final_iter_nums,
        const stopping_status *__restrict__ stop_status, size_type total_number)
{
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx < total_number) {
        final_iter_nums[tidx] += (1 - stop_status[tidx].has_stopped());
    }
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void divide_kernel(
    size_type num_rows, size_type num_cols, size_type num_alpha_cols,
    const ValueType *__restrict__ alpha, ValueType *__restrict__ x,
    size_type stride_x)
{
    constexpr auto warps_per_block = block_size / cuda_config::warp_size;
    const auto global_id =
        thread::get_thread_id<cuda_config::warp_size, warps_per_block>();
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    const auto alpha_id = num_alpha_cols == 1 ? 0 : col_id;
    if (row_id < num_rows) {
        x[row_id * stride_x + col_id] =
            alpha[alpha_id] == zero<ValueType>()
                ? zero<ValueType>()
                : x[row_id * stride_x + col_id] / alpha[alpha_id];
    }
}


template <typename ValueType>
void divide(std::shared_ptr<const CudaExecutor> exec,
            const matrix::Dense<ValueType> *alpha, matrix::Dense<ValueType> *x)
{
    if (cublas::is_supported<ValueType>::value && x->get_size()[1] == 1) {
        cublas::scal(exec->get_cublas_handle(), x->get_size()[0],
                     alpha->get_const_values(), x->get_values(),
                     x->get_stride());
    } else {
        // TODO: tune this parameter
        constexpr auto block_size = default_block_size;
        const dim3 grid_dim =
            ceildiv(x->get_size()[0] * x->get_size()[1], block_size);
        const dim3 block_dim{cuda_config::warp_size, 1,
                             block_size / cuda_config::warp_size};
        divide_kernel<block_size><<<grid_dim, block_dim>>>(
            x->get_size()[0], x->get_size()[1], alpha->get_size()[1],
            as_cuda_type(alpha->get_const_values()),
            as_cuda_type(x->get_values()), x->get_stride());
    }
}


template <typename ValueType>
void __global__
update_krylov_bases_kernels(size_type iter, size_type m, size_type n,
                            size_type stride_bases, size_type stride_next_basis,
                            const ValueType *__restrict__ next_krylov_basis,
                            ValueType *__restrict__ krylov_bases)
{
    const auto global_id = threadIdx.x + blockIdx.x * blockDim.x;
    const auto row = global_id / stride_bases;
    const auto column = global_id % stride_bases;
    if (global_id < stride_bases * m) {
        krylov_bases[row * stride_bases + n * (iter + 1) + column] =
            next_krylov_basis[row * stride_next_basis + column];
    }
}

template <typename ValueType>
void finish_arnoldi(std::shared_ptr<const CudaExecutor> exec,
                    matrix::Dense<ValueType> *next_krylov_basis,
                    matrix::Dense<ValueType> *krylov_bases,
                    matrix::Dense<ValueType> *hessenberg_iter,
                    const size_type iter, const stopping_status *stop_status)
{
    auto neg_one_op =
        initialize<matrix::Dense<ValueType>>({-one<ValueType>()}, exec);

    for (size_type i = 0; i < iter + 1; ++i) {
        auto krylov_basis = krylov_bases->create_submatrix(
            span{0, next_krylov_basis->get_size()[0]},
            span{i * next_krylov_basis->get_size()[1],
                 (i + 1) * next_krylov_basis->get_size()[1]});
        auto hessenberg_iter_column = hessenberg_iter->create_submatrix(
            span{i, i + 1}, span{0, next_krylov_basis->get_size()[1]});
        next_krylov_basis->compute_dot(krylov_basis.get(),
                                       hessenberg_iter_column.get());
        krylov_basis->scale(neg_one_op.get());
        next_krylov_basis->add_scaled(hessenberg_iter_column.get(),
                                      krylov_basis.get());
        // krylov_basis->scale(neg_one_op.get());
    }
    // for i in 1:iter
    //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
    //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
    // end

    auto next_krylov_basis_norm = hessenberg_iter->create_submatrix(
        span{iter + 1, iter + 2}, span{0, next_krylov_basis->get_size()[1]});
    next_krylov_basis->compute_norm2(next_krylov_basis_norm.get());
    // hessenberg(iter, iter + 1) = norm(next_krylov_basis)

    divide(exec, next_krylov_basis_norm.get(), next_krylov_basis);
    // next_krylov_basis /= hessenberg(iter, iter + 1)
    // End of arnoldi

    auto dims = next_krylov_basis->get_size();
    auto stride_bases = krylov_bases->get_stride();
    auto stride_next_basis = next_krylov_basis->get_stride();
    auto num_elems = dims[0] * std::max(stride_bases, stride_next_basis);
    dim3 block_dim(default_block_size);
    dim3 grid_dim(ceildiv(num_elems, block_dim.x));

    update_krylov_bases_kernels<<<grid_dim, block_dim>>>(
        iter, dims[0], dims[1], stride_bases, stride_next_basis,
        as_cuda_type(next_krylov_basis->get_values()),
        as_cuda_type(krylov_bases->get_values()));
}


template <typename ValueType>
__device__ void calculate_sin_and_cos(const size_type num_cols,
                                      const ValueType *hessenberg_iter,
                                      ValueType *givens_sin,
                                      ValueType *givens_cos,
                                      const size_type iter)
{
    const auto local_id = thread::get_local_thread_id<cuda_config::warp_size>();

    if (hessenberg_iter[iter * num_cols + local_id] == zero<ValueType>()) {
        givens_cos[iter * num_cols + local_id] = zero<ValueType>();
        givens_sin[iter * num_cols + local_id] = one<ValueType>();
    } else {
        auto hypotenuse =
            sqrt(hessenberg_iter[iter * num_cols + local_id] *
                     hessenberg_iter[iter * num_cols + local_id] +
                 hessenberg_iter[(iter + 1) * num_cols + local_id] *
                     hessenberg_iter[(iter + 1) * num_cols + local_id]);
        givens_cos[iter * num_cols + local_id] =
            abs(hessenberg_iter[iter * num_cols + local_id]) / hypotenuse;
        givens_sin[iter * num_cols + local_id] =
            givens_cos[iter * num_cols + local_id] *
            hessenberg_iter[(iter + 1) * num_cols + local_id] /
            hessenberg_iter[iter * num_cols + local_id];
    }
}


template <typename ValueType>
__device__ void calculate_residual_norm(const size_type num_cols,
                                        const ValueType *givens_sin,
                                        const ValueType *givens_cos,
                                        ValueType *residual_norm,
                                        ValueType *residual_norm_collection,
                                        const ValueType *b_norm,
                                        const size_type iter)
{
    const auto local_id = thread::get_local_thread_id<cuda_config::warp_size>();

    residual_norm_collection[(iter + 1) * num_cols + local_id] =
        -givens_sin[iter * num_cols + local_id] *
        residual_norm_collection[iter * num_cols + local_id];
    residual_norm_collection[iter * num_cols + local_id] =
        givens_cos[iter * num_cols + local_id] *
        residual_norm_collection[iter * num_cols + local_id];
    residual_norm[local_id] =
        abs(residual_norm_collection[(iter + 1) * num_cols + local_id]) /
        b_norm[local_id];
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void givens_rotation_kernel(
    const size_type num_rows, const size_type num_cols,
    ValueType *__restrict__ hessenberg_iter, ValueType *__restrict__ givens_sin,
    ValueType *__restrict__ givens_cos, ValueType *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norm_collection,
    const ValueType *__restrict__ b_norm, const size_type iter,
    const stopping_status *__restrict__ stop_status)
{
    const auto local_id = thread::get_local_thread_id<cuda_config::warp_size>();
    __shared__ UninitializedArray<ValueType, block_size> tmp;

    if (local_id >= num_cols || stop_status[local_id].has_stopped()) return;

    for (size_type i = 0; i < iter; ++i) {
        tmp[local_id] = givens_cos[i * num_cols + local_id] *
                            hessenberg_iter[i * num_cols + local_id] +
                        givens_sin[i * num_cols + local_id] *
                            hessenberg_iter[(i + 1) * num_cols + local_id];
        __syncthreads();
        hessenberg_iter[(i + 1) * num_cols + local_id] =
            givens_cos[i * num_cols + local_id] *
                hessenberg_iter[(i + 1) * num_cols + local_id] -
            givens_sin[i * num_cols + local_id] *
                hessenberg_iter[i * num_cols + local_id];
        hessenberg_iter[i * num_cols + local_id] = tmp[local_id];
        __syncthreads();
    }
    // for j in 1:iter - 1
    //     temp             =  cos(j)*hessenberg(j) +
    //                         sin(j)*hessenberg(j+1)
    //     hessenberg(j+1)  = -sin(j)*hessenberg(j) +
    //                         cos(j)*hessenberg(j+1)
    //     hessenberg(j)    =  temp;
    // end

    calculate_sin_and_cos(num_cols, hessenberg_iter, givens_sin, givens_cos,
                          iter);
    // Calculate sin and cos

    hessenberg_iter[iter * num_cols + local_id] =
        givens_cos[iter * num_cols + local_id] *
            hessenberg_iter[iter * num_cols + local_id] +
        givens_sin[iter * num_cols + local_id] *
            hessenberg_iter[(iter + 1) * num_cols + local_id];
    hessenberg_iter[(iter + 1) * num_cols + local_id] = zero<ValueType>();
    // hessenberg(iter)   = cos(iter)*hessenberg(iter) +
    //                      sin(iter)*hessenberg(iter)
    // hessenberg(iter+1) = 0

    calculate_residual_norm(num_cols, givens_sin, givens_cos, residual_norm,
                            residual_norm_collection, b_norm, iter);
    // Calculate residual norm
}


template <typename ValueType>
void givens_rotation(std::shared_ptr<const CudaExecutor> exec,
                     matrix::Dense<ValueType> *givens_sin,
                     matrix::Dense<ValueType> *givens_cos,
                     matrix::Dense<ValueType> *hessenberg_iter,
                     matrix::Dense<ValueType> *residual_norm,
                     matrix::Dense<ValueType> *residual_norm_collection,
                     const matrix::Dense<ValueType> *b_norm,
                     const size_type iter,
                     const Array<stopping_status> *stop_status)
{
    // TODO: tune this parameter
    // TODO: when number of right hand side is larger than block_size
    constexpr auto block_size = default_block_size;
    const dim3 block_dim{cuda_config::warp_size, 1,
                         block_size / cuda_config::warp_size};

    givens_rotation_kernel<block_size><<<1, block_dim>>>(
        hessenberg_iter->get_size()[0], hessenberg_iter->get_size()[1],
        as_cuda_type(hessenberg_iter->get_values()),
        as_cuda_type(givens_sin->get_values()),
        as_cuda_type(givens_cos->get_values()),
        as_cuda_type(residual_norm->get_values()),
        as_cuda_type(residual_norm_collection->get_values()),
        as_cuda_type(b_norm->get_const_values()), iter,
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
            const matrix::Dense<ValueType> *b_norm, const size_type iter,
            Array<size_type> *final_iter_nums,
            const Array<stopping_status> *stop_status)
{
    increase_final_iteration_numbers_kernel<<<
        ceildiv(final_iter_nums->get_num_elems(), default_block_size),
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


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void solve_upper_triangular_kernel(
    size_type num_cols, size_type num_rhs,
    const ValueType *__restrict__ residual_norm_collection,
    const ValueType *__restrict__ hessenberg, ValueType *__restrict__ y,
    const size_type *__restrict__ final_iter_nums)
{
    const auto local_id = thread::get_local_thread_id<cuda_config::warp_size>();

    if (local_id >= num_rhs) return;

    for (int i = final_iter_nums[local_id] - 1; i >= 0; --i) {
        auto temp = residual_norm_collection[i * num_rhs + local_id];
        for (size_type j = i + 1; j < final_iter_nums[local_id]; ++j) {
            temp -= hessenberg[i * num_cols + j * num_rhs + local_id] *
                    y[j * num_rhs + local_id];
        }
        y[i * num_rhs + local_id] =
            temp / hessenberg[i * num_cols + i * num_rhs + local_id];
        __syncthreads();
    }
    // Solve upper triangular.
    // y = hessenberg \ residual_norm_collection
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void calculate_Qy_kernel(
    const size_type num_rows, const size_type num_cols, const size_type num_rhs,
    const ValueType *__restrict__ krylov_bases, const ValueType *__restrict__ y,
    ValueType *__restrict__ before_preconditioner,
    const size_type *__restrict__ final_iter_nums)
{
    constexpr auto warps_per_block = block_size / cuda_config::warp_size;
    const auto global_id =
        thread::get_thread_id<cuda_config::warp_size, warps_per_block>();
    const auto row_id = global_id / num_rhs;
    const auto col_id = global_id % num_rhs;

    before_preconditioner[global_id] = zero<ValueType>();
    for (size_type j = 0; j < final_iter_nums[col_id]; ++j) {
        before_preconditioner[global_id] +=
            krylov_bases[row_id * num_cols + j * num_rhs + col_id] *
            y[j * num_rhs + col_id];
    }
}


template <typename ValueType>
void solve_upper_triangular(
    const matrix::Dense<ValueType> *residual_norm_collection,
    const matrix::Dense<ValueType> *hessenberg, matrix::Dense<ValueType> *y,
    const Array<size_type> *final_iter_nums)
{
    // TODO: tune this parameter
    // TODO: when number of right hand side is larger than block_size
    constexpr auto block_size = default_block_size;
    const dim3 block_dim{cuda_config::warp_size, 1,
                         block_size / cuda_config::warp_size};

    solve_upper_triangular_kernel<block_size><<<1, block_dim>>>(
        hessenberg->get_size()[1], residual_norm_collection->get_size()[1],
        as_cuda_type(residual_norm_collection->get_const_values()),
        as_cuda_type(hessenberg->get_const_values()),
        as_cuda_type(y->get_values()),
        as_cuda_type(final_iter_nums->get_const_data()));
}


template <typename ValueType>
void solve_x(std::shared_ptr<const CudaExecutor> exec,
             const matrix::Dense<ValueType> *krylov_bases,
             matrix::Dense<ValueType> *y, matrix::Dense<ValueType> *x,
             const Array<size_type> *final_iter_nums,
             const LinOp *preconditioner)
{
    auto before_preconditioner =
        matrix::Dense<ValueType>::create_with_config_of(x);
    auto after_preconditioner =
        matrix::Dense<ValueType>::create_with_config_of(x);

    constexpr auto block_size = default_block_size;
    const dim3 grid_dim =
        ceildiv(x->get_size()[0] * x->get_size()[1], block_size);
    const dim3 block_dim{cuda_config::warp_size, 1,
                         block_size / cuda_config::warp_size};

    calculate_Qy_kernel<block_size><<<grid_dim, block_dim>>>(
        before_preconditioner->get_size()[0], krylov_bases->get_size()[1],
        before_preconditioner->get_size()[1],
        as_cuda_type(krylov_bases->get_const_values()),
        as_cuda_type(y->get_const_values()),
        as_cuda_type(before_preconditioner.get()->get_values()),
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
