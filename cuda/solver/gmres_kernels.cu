/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

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


#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/reduction.cuh"
#include "cuda/components/uninitialized_array.hpp"
#include "cuda/matrix/dense_kernels.cu"


#include <iostream>


namespace gko {
namespace kernels {
namespace cuda {
namespace gmres {


constexpr int default_block_size = 512;


namespace kernel {


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


}  // namespace kernel


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

    dense::compute_norm2(exec, b, b_norm);
    kernel::initialize_1_kernel<block_size><<<grid_dim, block_dim>>>(
        b->get_size()[0], b->get_size()[1], b->get_stride(), krylov_dim,
        as_cuda_type(b->get_const_values()), as_cuda_type(b_norm->get_values()),
        as_cuda_type(residual->get_values()),
        as_cuda_type(givens_sin->get_values()),
        as_cuda_type(givens_cos->get_values()),
        as_cuda_type(stop_status->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_1_KERNEL);


namespace kernel {


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void initialize_2_kernel_1(
    size_type num_rows, size_type num_cols, size_type stride,
    size_type krylov_dim, const ValueType *__restrict__ residual,
    ValueType *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norms,
    ValueType *__restrict__ krylov_bases,
    size_type *__restrict__ final_iter_nums)
{
    constexpr auto warps_per_block = block_size / cuda_config::warp_size;
    const auto global_id =
        thread::get_thread_id<cuda_config::warp_size, warps_per_block>();

    if (global_id < num_rows * (krylov_dim + 1) * num_cols) {
        krylov_bases[global_id] = 0;
    }

    if (global_id < (krylov_dim + 1) * num_cols) {
        residual_norms[global_id] = 0;
    }
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void initialize_2_kernel_2(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ residual,
    const ValueType *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norms,
    ValueType *__restrict__ krylov_bases)
{
    constexpr auto warps_per_block = block_size / cuda_config::warp_size;
    const auto global_id =
        thread::get_thread_id<cuda_config::warp_size, warps_per_block>();
    const auto row = ceildiv(global_id + 1, num_cols) - 1;
    const auto column = global_id - row * num_cols;

    if (global_id < num_cols) {
        residual_norms[global_id] = residual_norm[global_id];
    }

    if (global_id < num_rows * num_cols) {
        krylov_bases[row * stride + column] =
            residual[global_id] / residual_norm[column];
    }
}


}  // namespace kernel


template <typename ValueType>
void initialize_2(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *residual_norm,
                  matrix::Dense<ValueType> *residual_norms,
                  matrix::Dense<ValueType> *krylov_bases,
                  Array<size_type> *final_iter_nums, const size_type krylov_dim)
{
    const dim3 grid_dim(
        ceildiv(krylov_bases->get_size()[0], default_block_size) *
            krylov_bases->get_stride(),
        1, 1);
    const dim3 block_dim(default_block_size, 1, 1);
    constexpr auto block_size = default_block_size;

    kernel::initialize_2_kernel_1<block_size><<<grid_dim, block_dim>>>(
        residual->get_size()[0], residual->get_size()[1],
        residual->get_stride(), krylov_dim,
        as_cuda_type(residual->get_const_values()),
        as_cuda_type(residual_norm->get_values()),
        as_cuda_type(residual_norms->get_values()),
        as_cuda_type(krylov_bases->get_values()),
        as_cuda_type(final_iter_nums->get_data()));
    dense::compute_norm2(exec, residual, residual_norm);
    kernel::initialize_2_kernel_2<block_size><<<grid_dim, block_dim>>>(
        residual->get_size()[0], residual->get_size()[1],
        krylov_bases->get_stride(), as_cuda_type(residual->get_const_values()),
        as_cuda_type(residual_norm->get_const_values()),
        as_cuda_type(residual_norms->get_values()),
        as_cuda_type(krylov_bases->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_2_KERNEL);


namespace kernel {


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void minus_scaled(
    size_type num_rows, size_type num_cols, size_type num_alpha_cols,
    const ValueType *__restrict__ alpha, const ValueType *__restrict__ x,
    size_type stride_x, ValueType *__restrict__ y, size_type stride_y)
{
    constexpr auto warps_per_block = block_size / cuda_config::warp_size;
    const auto global_id =
        thread::get_thread_id<cuda_config::warp_size, warps_per_block>();
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    const auto alpha_id = num_alpha_cols == 1 ? 0 : col_id;
    if (row_id < num_rows && alpha[alpha_id] != zero<ValueType>()) {
        y[row_id * stride_y + col_id] -=
            x[row_id * stride_x + col_id] * alpha[alpha_id];
    }
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void divide(
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


}  // namespace kernel


template <typename ValueType>
void minus_scaled(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Dense<ValueType> *alpha,
                  const matrix::Dense<ValueType> *x,
                  matrix::Dense<ValueType> *y)
{
    if (cublas::is_supported<ValueType>::value && x->get_size()[1] == 1) {
        cublas::axpy(exec->get_cublas_handle(), x->get_size()[0],
                     alpha->get_const_values(), x->get_const_values(),
                     x->get_stride(), y->get_values(), y->get_stride());
    } else {
        // TODO: tune this parameter
        constexpr auto block_size = default_block_size;
        const dim3 grid_dim =
            ceildiv(x->get_size()[0] * x->get_size()[1], block_size);
        const dim3 block_dim{cuda_config::warp_size, 1,
                             block_size / cuda_config::warp_size};
        kernel::minus_scaled<block_size><<<grid_dim, block_dim>>>(
            x->get_size()[0], x->get_size()[1], alpha->get_size()[1],
            as_cuda_type(alpha->get_const_values()),
            as_cuda_type(x->get_const_values()), x->get_stride(),
            as_cuda_type(y->get_values()), y->get_stride());
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
        kernel::divide<block_size><<<grid_dim, block_dim>>>(
            x->get_size()[0], x->get_size()[1], alpha->get_size()[1],
            as_cuda_type(alpha->get_const_values()),
            as_cuda_type(x->get_values()), x->get_stride());
    }
}


template <typename ValueType>
void finish_arnoldi(std::shared_ptr<const CudaExecutor> exec,
                    matrix::Dense<ValueType> *next_krylov_basis,
                    matrix::Dense<ValueType> *krylov_bases,
                    matrix::Dense<ValueType> *hessenberg_iter,
                    const size_type iter, const stopping_status *stop_status)
{
    for (size_type i = 0; i < iter + 1; ++i) {
        auto krylov_basis = krylov_bases->create_submatrix(
            span{0, next_krylov_basis->get_size()[0]},
            span{i * next_krylov_basis->get_size()[1],
                 (i + 1) * next_krylov_basis->get_size()[1]});
        auto hessenberg_iter_column = hessenberg_iter->create_submatrix(
            span{i, i + 1}, span{0, next_krylov_basis->get_size()[1]});
        dense::compute_dot(exec, next_krylov_basis, krylov_basis.get(),
                           hessenberg_iter_column.get());
        minus_scaled(exec, hessenberg_iter_column.get(), krylov_basis.get(),
                     next_krylov_basis);
    }
    // for i in 1:iter
    //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
    //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
    // end

    auto next_krylov_basis_norm = hessenberg_iter->create_submatrix(
        span{iter + 1, iter + 2}, span{0, next_krylov_basis->get_size()[1]});
    dense::compute_norm2(exec, next_krylov_basis, next_krylov_basis_norm.get());
    // hessenberg(iter, iter + 1) = norm(next_krylov_basis)

    divide(exec, next_krylov_basis_norm.get(), next_krylov_basis);
    // next_krylov_basis /= hessenberg(iter, iter + 1)
    // End of arnoldi
}


namespace kernel {


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
__device__ void claculate_residual_norm(
    const size_type num_cols, const ValueType *givens_sin,
    const ValueType *givens_cos, ValueType *residual_norm,
    ValueType *residual_norms, const ValueType *b_norm, const size_type iter)
{
    const auto local_id = thread::get_local_thread_id<cuda_config::warp_size>();

    residual_norms[(iter + 1) * num_cols + local_id] =
        -givens_sin[iter * num_cols + local_id] *
        residual_norms[iter * num_cols + local_id];
    residual_norms[iter * num_cols + local_id] =
        givens_cos[iter * num_cols + local_id] *
        residual_norms[iter * num_cols + local_id];
    residual_norm[local_id] =
        abs(residual_norms[(iter + 1) * num_cols + local_id]) /
        b_norm[local_id];
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void givens_rotation(
    const size_type num_rows, const size_type num_cols,
    ValueType *__restrict__ hessenberg_iter, ValueType *__restrict__ givens_sin,
    ValueType *__restrict__ givens_cos, ValueType *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norms,
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

    claculate_residual_norm(num_cols, givens_sin, givens_cos, residual_norm,
                            residual_norms, b_norm, iter);
    // Calculate residual norm
}


}  // namespace kernel


template <typename ValueType>
void givens_rotation(std::shared_ptr<const CudaExecutor> exec,
                     matrix::Dense<ValueType> *givens_sin,
                     matrix::Dense<ValueType> *givens_cos,
                     matrix::Dense<ValueType> *hessenberg_iter,
                     matrix::Dense<ValueType> *residual_norm,
                     matrix::Dense<ValueType> *residual_norms,
                     const matrix::Dense<ValueType> *b_norm,
                     const size_type iter,
                     const Array<stopping_status> *stop_status)
{
    // TODO: tune this parameter
    // TODO: when number of right hand side is larger than block_size
    constexpr auto block_size = default_block_size;
    const dim3 block_dim{cuda_config::warp_size, 1,
                         block_size / cuda_config::warp_size};

    kernel::givens_rotation<block_size><<<1, block_dim>>>(
        hessenberg_iter->get_size()[0], hessenberg_iter->get_size()[1],
        as_cuda_type(hessenberg_iter->get_values()),
        as_cuda_type(givens_sin->get_values()),
        as_cuda_type(givens_cos->get_values()),
        as_cuda_type(residual_norm->get_values()),
        as_cuda_type(residual_norms->get_values()),
        as_cuda_type(b_norm->get_const_values()), iter,
        as_cuda_type(stop_status->get_const_data()));
}


template <typename ValueType>
void step_1(std::shared_ptr<const CudaExecutor> exec,
            matrix::Dense<ValueType> *next_krylov_basis,
            matrix::Dense<ValueType> *givens_sin,
            matrix::Dense<ValueType> *givens_cos,
            matrix::Dense<ValueType> *residual_norm,
            matrix::Dense<ValueType> *residual_norms,
            matrix::Dense<ValueType> *krylov_bases,
            matrix::Dense<ValueType> *hessenberg_iter,
            const matrix::Dense<ValueType> *b_norm, const size_type iter,
            const Array<stopping_status> *stop_status)
{
    finish_arnoldi(exec, next_krylov_basis, krylov_bases, hessenberg_iter, iter,
                   stop_status->get_const_data());
    givens_rotation(exec, givens_sin, givens_cos, hessenberg_iter,
                    residual_norm, residual_norms, b_norm, iter, stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_1_KERNEL);


namespace kernel {


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void solve_upper_triangular(
    size_type num_cols, size_type num_rhs,
    const ValueType *__restrict__ residual_norms,
    ValueType *__restrict__ hessenberg, ValueType *__restrict__ y,
    const size_type *__restrict__ final_iter_nums)
{
    const auto local_id = thread::get_local_thread_id<cuda_config::warp_size>();

    if (local_id >= num_rhs) return;

    for (int i = final_iter_nums[local_id] - 1; i >= 0; --i) {
        auto temp = residual_norms[i * num_rhs + local_id];
        for (size_type j = i + 1; j < final_iter_nums[local_id]; ++j) {
            temp -= hessenberg[i * num_cols + j * num_rhs + local_id] *
                    y[j * num_rhs + local_id];
        }
        y[i * num_rhs + local_id] =
            temp / hessenberg[i * num_cols + i * num_rhs + local_id];
        __syncthreads();
    }
    // Solve upper triangular.
    // y = hessenberg \ residual_norms
}


}  // namespace kernel


template <typename ValueType>
void solve_upper_triangular(const matrix::Dense<ValueType> *residual_norms,
                            matrix::Dense<ValueType> *hessenberg,
                            matrix::Dense<ValueType> *y,
                            const Array<size_type> *final_iter_nums)
{
    // TODO: tune this parameter
    // TODO: when number of right hand side is larger than block_size
    constexpr auto block_size = default_block_size;
    const dim3 block_dim{cuda_config::warp_size, 1,
                         block_size / cuda_config::warp_size};

    kernel::solve_upper_triangular<block_size><<<1, block_dim>>>(
        hessenberg->get_size()[1], residual_norms->get_size()[1],
        as_cuda_type(residual_norms->get_const_values()),
        as_cuda_type(hessenberg->get_values()), as_cuda_type(y->get_values()),
        as_cuda_type(final_iter_nums->get_const_data()));
}


template <typename ValueType>
void solve_x(std::shared_ptr<const CudaExecutor> exec,
             matrix::Dense<ValueType> *krylov_bases,
             matrix::Dense<ValueType> *y, matrix::Dense<ValueType> *x,
             const Array<size_type> *final_iter_nums,
             const LinOp *preconditioner)
{
    // Solve x
    // x = x + preconditioner_ * krylov_bases * y
}


template <typename ValueType>
void step_2(std::shared_ptr<const CudaExecutor> exec,
            const matrix::Dense<ValueType> *residual_norms,
            matrix::Dense<ValueType> *krylov_bases,
            matrix::Dense<ValueType> *hessenberg, matrix::Dense<ValueType> *y,
            matrix::Dense<ValueType> *x,
            const Array<size_type> *final_iter_nums,
            const LinOp *preconditioner)
{
    solve_upper_triangular(residual_norms, hessenberg, y, final_iter_nums);
    solve_x(exec, krylov_bases, y, x, final_iter_nums, preconditioner);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_2_KERNEL);


}  // namespace gmres
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
