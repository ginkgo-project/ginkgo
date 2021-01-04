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

#include "core/solver/gmres_kernels.hpp"


#include <algorithm>


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/fill_array.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/hipblas_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The GMRES solver namespace.
 *
 * @ingroup gmres
 */
namespace gmres {


constexpr int default_block_size = 512;
// default_dot_dim can not be 64 in hip because 64 * 64 exceeds their max block
// size limit.
constexpr int default_dot_dim = 32;
constexpr int default_dot_size = default_dot_dim * default_dot_dim;


#include "common/solver/gmres_kernels.hpp.inc"


template <typename ValueType>
void initialize_1(std::shared_ptr<const HipExecutor> exec,
                  const matrix::Dense<ValueType> *b,
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

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(initialize_1_kernel<block_size>), dim3(grid_dim),
        dim3(block_dim), 0, 0, b->get_size()[0], b->get_size()[1], krylov_dim,
        as_hip_type(b->get_const_values()), b->get_stride(),
        as_hip_type(residual->get_values()), residual->get_stride(),
        as_hip_type(givens_sin->get_values()), givens_sin->get_stride(),
        as_hip_type(givens_cos->get_values()), givens_cos->get_stride(),
        as_hip_type(stop_status->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_1_KERNEL);


template <typename ValueType>
void initialize_2(std::shared_ptr<const HipExecutor> exec,
                  const matrix::Dense<ValueType> *residual,
                  matrix::Dense<remove_complex<ValueType>> *residual_norm,
                  matrix::Dense<ValueType> *residual_norm_collection,
                  matrix::Dense<ValueType> *krylov_bases,
                  Array<size_type> *final_iter_nums, size_type krylov_dim)
{
    const auto num_rows = residual->get_size()[0];
    const auto num_rhs = residual->get_size()[1];
    const dim3 grid_dim_1(
        ceildiv(krylov_bases->get_size()[0] * krylov_bases->get_stride(),
                default_block_size),
        1, 1);
    const dim3 block_dim(default_block_size, 1, 1);
    constexpr auto block_size = default_block_size;

    residual->compute_norm2(residual_norm);

    const dim3 grid_dim_2(ceildiv(num_rows * num_rhs, default_block_size), 1,
                          1);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(initialize_2_2_kernel<block_size>), dim3(grid_dim_2),
        dim3(block_dim), 0, 0, residual->get_size()[0], residual->get_size()[1],
        as_hip_type(residual->get_const_values()), residual->get_stride(),
        as_hip_type(residual_norm->get_const_values()),
        as_hip_type(residual_norm_collection->get_values()),
        as_hip_type(krylov_bases->get_values()), krylov_bases->get_stride(),
        as_hip_type(final_iter_nums->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_2_KERNEL);


template <typename ValueType>
void finish_arnoldi(std::shared_ptr<const HipExecutor> exec, size_type num_rows,
                    matrix::Dense<ValueType> *krylov_bases,
                    matrix::Dense<ValueType> *hessenberg_iter, size_type iter,
                    const stopping_status *stop_status)
{
    const auto stride_krylov = krylov_bases->get_stride();
    const auto stride_hessenberg = hessenberg_iter->get_stride();
    auto hipblas_handle = exec->get_hipblas_handle();
    const dim3 grid_size(
        ceildiv(hessenberg_iter->get_size()[1], default_dot_dim),
        exec->get_num_multiprocessor() * 2);
    const dim3 block_size(default_dot_dim, default_dot_dim);
    auto next_krylov_basis =
        krylov_bases->get_values() +
        (iter + 1) * num_rows * hessenberg_iter->get_size()[1];
    for (size_type k = 0; k < iter + 1; ++k) {
        const auto k_krylov_bases =
            krylov_bases->get_const_values() +
            k * num_rows * hessenberg_iter->get_size()[1];
        if (hessenberg_iter->get_size()[1] > 1) {
            // TODO: this condition should be tuned
            // single rhs will use vendor's dot, otherwise, use our own
            // multidot_kernel which parallelize multiple rhs.
            components::fill_array(
                exec, hessenberg_iter->get_values() + k * stride_hessenberg,
                hessenberg_iter->get_size()[1], zero<ValueType>());
            hipLaunchKernelGGL(
                multidot_kernel, dim3(grid_size), dim3(block_size), 0, 0, k,
                num_rows, hessenberg_iter->get_size()[1],
                as_hip_type(k_krylov_bases), as_hip_type(next_krylov_basis),
                stride_krylov, as_hip_type(hessenberg_iter->get_values()),
                stride_hessenberg, as_hip_type(stop_status));
        } else {
            hipblas::dot(exec->get_hipblas_handle(), num_rows, k_krylov_bases,
                         stride_krylov, next_krylov_basis, stride_krylov,
                         hessenberg_iter->get_values() + k * stride_hessenberg);
        }
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(update_next_krylov_kernel<default_block_size>),
            dim3(ceildiv(num_rows * stride_krylov, default_block_size)),
            dim3(default_block_size), 0, 0, k, num_rows,
            hessenberg_iter->get_size()[1], as_hip_type(k_krylov_bases),
            as_hip_type(next_krylov_basis), stride_krylov,
            as_hip_type(hessenberg_iter->get_const_values()), stride_hessenberg,
            as_hip_type(stop_status));
    }
    // for i in 1:iter
    //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
    //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
    // end


    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(update_hessenberg_2_kernel<default_block_size>),
        dim3(hessenberg_iter->get_size()[1]), dim3(default_block_size), 0, 0,
        iter, num_rows, hessenberg_iter->get_size()[1],
        as_hip_type(next_krylov_basis), stride_krylov,
        as_hip_type(hessenberg_iter->get_values()), stride_hessenberg,
        as_hip_type(stop_status));

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(update_krylov_kernel<default_block_size>),
        dim3(ceildiv(num_rows * stride_krylov, default_block_size)),
        dim3(default_block_size), 0, 0, iter, num_rows,
        hessenberg_iter->get_size()[1], as_hip_type(next_krylov_basis),
        stride_krylov, as_hip_type(hessenberg_iter->get_const_values()),
        stride_hessenberg, as_hip_type(stop_status));
    // next_krylov_basis /= hessenberg(iter, iter + 1)
    // End of arnoldi
}


template <typename ValueType>
void givens_rotation(std::shared_ptr<const HipExecutor> exec,
                     matrix::Dense<ValueType> *givens_sin,
                     matrix::Dense<ValueType> *givens_cos,
                     matrix::Dense<ValueType> *hessenberg_iter,
                     matrix::Dense<remove_complex<ValueType>> *residual_norm,
                     matrix::Dense<ValueType> *residual_norm_collection,
                     size_type iter, const Array<stopping_status> *stop_status)
{
    // TODO: tune block_size for optimal performance
    constexpr auto block_size = default_block_size;
    const auto num_cols = hessenberg_iter->get_size()[1];
    const dim3 block_dim{block_size, 1, 1};
    const dim3 grid_dim{
        static_cast<unsigned int>(ceildiv(num_cols, block_size)), 1, 1};

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(givens_rotation_kernel<block_size>), dim3(grid_dim),
        dim3(block_dim), 0, 0, hessenberg_iter->get_size()[0],
        hessenberg_iter->get_size()[1], iter,
        as_hip_type(hessenberg_iter->get_values()),
        hessenberg_iter->get_stride(), as_hip_type(givens_sin->get_values()),
        givens_sin->get_stride(), as_hip_type(givens_cos->get_values()),
        givens_cos->get_stride(), as_hip_type(residual_norm->get_values()),
        as_hip_type(residual_norm_collection->get_values()),
        residual_norm_collection->get_stride(),
        as_hip_type(stop_status->get_const_data()));
}


template <typename ValueType>
void step_1(std::shared_ptr<const HipExecutor> exec, size_type num_rows,
            matrix::Dense<ValueType> *givens_sin,
            matrix::Dense<ValueType> *givens_cos,
            matrix::Dense<remove_complex<ValueType>> *residual_norm,
            matrix::Dense<ValueType> *residual_norm_collection,
            matrix::Dense<ValueType> *krylov_bases,
            matrix::Dense<ValueType> *hessenberg_iter, size_type iter,
            Array<size_type> *final_iter_nums,
            const Array<stopping_status> *stop_status)
{
    hipLaunchKernelGGL(
        increase_final_iteration_numbers_kernel,
        dim3(static_cast<unsigned int>(
            ceildiv(final_iter_nums->get_num_elems(), default_block_size))),
        dim3(default_block_size), 0, 0,
        as_hip_type(final_iter_nums->get_data()),
        as_hip_type(stop_status->get_const_data()),
        final_iter_nums->get_num_elems());
    finish_arnoldi(exec, num_rows, krylov_bases, hessenberg_iter, iter,
                   stop_status->get_const_data());
    givens_rotation(exec, givens_sin, givens_cos, hessenberg_iter,
                    residual_norm, residual_norm_collection, iter, stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_1_KERNEL);


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

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(solve_upper_triangular_kernel<block_size>),
        dim3(grid_dim), dim3(block_dim), 0, 0, hessenberg->get_size()[1],
        num_rhs, as_hip_type(residual_norm_collection->get_const_values()),
        residual_norm_collection->get_stride(),
        as_hip_type(hessenberg->get_const_values()), hessenberg->get_stride(),
        as_hip_type(y->get_values()), y->get_stride(),
        as_hip_type(final_iter_nums->get_const_data()));
}


template <typename ValueType>
void calculate_qy(const matrix::Dense<ValueType> *krylov_bases,
                  const matrix::Dense<ValueType> *y,
                  matrix::Dense<ValueType> *before_preconditioner,
                  const Array<size_type> *final_iter_nums)
{
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


    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(calculate_Qy_kernel<block_size>), dim3(grid_dim),
        dim3(block_dim), 0, 0, num_rows, num_cols, num_rhs,
        as_hip_type(krylov_bases->get_const_values()),
        krylov_bases->get_stride(), as_hip_type(y->get_const_values()),
        y->get_stride(), as_hip_type(before_preconditioner->get_values()),
        stride_before_preconditioner,
        as_hip_type(final_iter_nums->get_const_data()));
    // Calculate qy
    // before_preconditioner = krylov_bases * y
}


template <typename ValueType>
void step_2(std::shared_ptr<const HipExecutor> exec,
            const matrix::Dense<ValueType> *residual_norm_collection,
            const matrix::Dense<ValueType> *krylov_bases,
            const matrix::Dense<ValueType> *hessenberg,
            matrix::Dense<ValueType> *y,
            matrix::Dense<ValueType> *before_preconditioner,
            const Array<size_type> *final_iter_nums)
{
    solve_upper_triangular(residual_norm_collection, hessenberg, y,
                           final_iter_nums);
    calculate_qy(krylov_bases, y, before_preconditioner, final_iter_nums);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_2_KERNEL);


}  // namespace gmres
}  // namespace hip
}  // namespace kernels
}  // namespace gko
