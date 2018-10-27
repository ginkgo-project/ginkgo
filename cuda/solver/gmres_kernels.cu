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


namespace gko {
namespace kernels {
namespace cuda {
namespace gmres {


constexpr int default_block_size = 512;


namespace {


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void initialize_1_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    size_type krylov_dim, const ValueType *__restrict__ b,
    ValueType *__restrict__ b_norm, ValueType *__restrict__ residual,
    ValueType *__restrict__ givens_sin, ValueType *__restrict__ givens_cos,
    stopping_status *__restrict__ stop_status)
{
    const auto idx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (idx < num_cols) {
        b_norm[idx] = zero<ValueType>();
        // TODO: implement reduction for b_norm
        for (size_type i = 0; i < num_rows; ++i) {
            b_norm[idx] += b[i * stride + idx] * b[i * stride + idx];
        }
        b_norm[idx] = sqrt(b_norm[idx]);
        stop_status[idx].reset();
    }

    if (idx < num_rows * stride) {
        residual[idx] = b[idx];
    }

    if (idx < krylov_dim * num_cols) {
        givens_sin[idx] = zero<ValueType>();
        givens_cos[idx] = zero<ValueType>();
    }
}


}  // namespace


template <typename ValueType>
void initialize_1(
    std::shared_ptr<const CudaExecutor> exec, const matrix::Dense<ValueType> *b,
    matrix::Dense<ValueType> *b_norm, matrix::Dense<ValueType> *residual,
    matrix::Dense<ValueType> *givens_sin, matrix::Dense<ValueType> *givens_cos,
    Array<stopping_status> *stop_status, const size_type krylov_dim)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(b->get_size()[0], default_block_size) * b->get_size()[1], 1, 1);
    initialize_1_kernel<<<grid_size, block_size, 0, 0>>>(
        b->get_size()[0], b->get_size()[1], b->get_stride(), krylov_dim,
        as_cuda_type(b->get_const_values()), as_cuda_type(b_norm->get_values()),
        as_cuda_type(residual->get_values()),
        as_cuda_type(givens_sin->get_values()),
        as_cuda_type(givens_cos->get_values()),
        as_cuda_type(stop_status->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_1_KERNEL);


namespace {


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void initialize_2_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    size_type krylov_dim, const ValueType *__restrict__ residual,
    ValueType *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norms,
    ValueType *__restrict__ krylov_bases,
    size_type *__restrict__ final_iter_nums)
{
    const auto idx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (idx < (krylov_dim + 1) * num_cols) {
        residual_norms[idx] = 0;
    }

    if (idx < num_cols) {
        residual_norm[idx] = zero<ValueType>();
        // TODO: implement reduction for residual_norm
        for (size_type i = 0; i < num_rows; ++i) {
            residual_norm[idx] +=
                residual[i * stride + idx] * residual[i * stride + idx];
        }
        residual_norm[idx] = sqrt(residual_norm[idx]);
        final_iter_nums[idx] = 0;
        residual_norms[idx] = residual_norm[idx];
    }

    if (idx < num_rows * stride) {
        krylov_bases[idx] =
            residual[idx] / residual_norm[ceildiv(idx, stride) - 1];
    }
}


}  // namespace


template <typename ValueType>
void initialize_2(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *residual_norm,
                  matrix::Dense<ValueType> *residual_norms,
                  matrix::Dense<ValueType> *krylov_bases,
                  Array<size_type> *final_iter_nums, const size_type krylov_dim)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(krylov_bases->get_size()[0], default_block_size) *
            krylov_bases->get_size()[1],
        1, 1);
    initialize_2_kernel<<<grid_size, block_size, 0, 0>>>(
        krylov_bases->get_size()[0], krylov_bases->get_size()[1],
        krylov_bases->get_stride(), krylov_dim,
        as_cuda_type(residual->get_const_values()),
        as_cuda_type(residual_norm->get_values()),
        as_cuda_type(residual_norms->get_values()),
        as_cuda_type(krylov_bases->get_values()),
        as_cuda_type(final_iter_nums->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_2_KERNEL);


namespace {


template <typename ValueType>
__device__ void finish_arnoldi_kernel(ValueType *next_krylov_basis,
                                      ValueType *krylov_bases,
                                      ValueType *hessenberg_iter,
                                      const size_type iter,
                                      const stopping_status *stop_status)
{}


template <typename ValueType>
__device__ void givens_rotation_kernel(ValueType *next_krylov_basis,
                                       ValueType *givens_sin,
                                       ValueType *givens_cos,
                                       ValueType *hessenberg_iter,
                                       const size_type iter,
                                       const stopping_status *stop_status)
{}


template <typename ValueType>
__device__ void calculate_next_residual_norm_kernel(
    ValueType *givens_sin, ValueType *givens_cos, ValueType *residual_norm,
    ValueType *residual_norms, const ValueType *b_norm, const size_type iter,
    const stopping_status *stop_status)
{}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_1_kernel(
    size_type num_rows, size_type num_cols, size_type stride, size_type iter,
    ValueType *__restrict__ next_krylov_basis,
    ValueType *__restrict__ givens_sin, ValueType *__restrict__ givens_cos,
    ValueType *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norms,
    ValueType *__restrict__ krylov_bases,
    ValueType *__restrict__ hessenberg_iter,
    const ValueType *__restrict__ b_norm,
    const stopping_status *__restrict__ stop_status)
{
    finish_arnoldi_kernel(next_krylov_basis, krylov_bases, hessenberg_iter,
                          iter, stop_status);
    givens_rotation_kernel(next_krylov_basis, givens_sin, givens_cos,
                           hessenberg_iter, iter, stop_status);
    calculate_next_residual_norm_kernel(givens_sin, givens_cos, residual_norm,
                                        residual_norms, b_norm, iter,
                                        stop_status);
}


}  // namespace


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
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(krylov_bases->get_size()[0], default_block_size) *
            krylov_bases->get_size()[1],
        1, 1);
    step_1_kernel<<<grid_size, block_size, 0, 0>>>(
        krylov_bases->get_size()[0], krylov_bases->get_size()[1],
        krylov_bases->get_stride(), iter,
        as_cuda_type(next_krylov_basis->get_values()),
        as_cuda_type(givens_sin->get_values()),
        as_cuda_type(givens_cos->get_values()),
        as_cuda_type(residual_norm->get_values()),
        as_cuda_type(residual_norms->get_values()),
        as_cuda_type(krylov_bases->get_values()),
        as_cuda_type(hessenberg_iter->get_values()),
        as_cuda_type(b_norm->get_const_values()),
        as_cuda_type(stop_status->get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_1_KERNEL);


namespace {


template <typename ValueType>
__device__ void solve_upper_triangular_kernel(const ValueType *residual_norms,
                                              ValueType *hessenberg,
                                              ValueType *y,
                                              size_type *final_iter_nums)
{}


template <typename ValueType>
__device__ void solve_x_kernel(ValueType *krylov_bases, ValueType *y,
                               ValueType *x, size_type *final_iter_nums,
                               LinOp *preconditioner)
{}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_2_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ residual_norms,
    ValueType *__restrict__ krylov_bases, ValueType *__restrict__ hessenberg,
    ValueType *__restrict__ y, ValueType *__restrict__ x,
    const size_type *__restrict__ final_iter_nums, const LinOp *preconditioner)
{
    solve_upper_triangular_kernel(residual_norms, hessenberg, y,
                                  final_iter_nums);
    solve_x_kernel(krylov_bases, y, x, final_iter_nums, preconditioner);
}


}  // namespace


template <typename ValueType>
void step_2(std::shared_ptr<const CudaExecutor> exec,
            const matrix::Dense<ValueType> *residual_norms,
            matrix::Dense<ValueType> *krylov_bases,
            matrix::Dense<ValueType> *hessenberg, matrix::Dense<ValueType> *y,
            matrix::Dense<ValueType> *x,
            const Array<size_type> *final_iter_nums,
            const LinOp *preconditioner)
{
    NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_2_KERNEL);


}  // namespace gmres
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
