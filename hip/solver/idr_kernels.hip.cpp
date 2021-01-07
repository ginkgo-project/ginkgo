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

#include "core/solver/idr_kernels.hpp"


#include <ctime>
#include <random>


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/components/fill_array.hpp"
#include "hip/base/hipblas_bindings.hip.hpp"
#include "hip/base/hiprand_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The IDR solver namespace.
 *
 * @ingroup idr
 */
namespace idr {


constexpr int default_block_size = 512;
constexpr int default_dot_dim = 32;
constexpr int default_dot_size = default_dot_dim * default_dot_dim;


#include "common/solver/idr_kernels.hpp.inc"


namespace {


template <typename ValueType>
void initialize_m(const size_type nrhs, matrix::Dense<ValueType> *m,
                  Array<stopping_status> *stop_status)
{
    const auto subspace_dim = m->get_size()[0];
    const auto m_stride = m->get_stride();

    const auto grid_dim = ceildiv(m_stride * subspace_dim, default_block_size);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(initialize_m_kernel), grid_dim,
                       default_block_size, 0, 0, subspace_dim, nrhs,
                       as_hip_type(m->get_values()), m_stride,
                       as_hip_type(stop_status->get_data()));
}


template <typename ValueType>
void initialize_subspace_vectors(matrix::Dense<ValueType> *subspace_vectors,
                                 bool deterministic)
{
    if (deterministic) {
        auto subspace_vectors_data = matrix_data<ValueType>(
            subspace_vectors->get_size(), std::normal_distribution<>(0.0, 1.0),
            std::ranlux48(15));
        subspace_vectors->read(subspace_vectors_data);
    } else {
        auto gen =
            hiprand::rand_generator(time(NULL), HIPRAND_RNG_PSEUDO_DEFAULT);
        hiprand::rand_vector(
            gen,
            subspace_vectors->get_size()[0] * subspace_vectors->get_stride(),
            0.0, 1.0, subspace_vectors->get_values());
    }
}


template <typename ValueType>
void orthonormalize_subspace_vectors(matrix::Dense<ValueType> *subspace_vectors)
{
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            orthonormalize_subspace_vectors_kernel<default_block_size>),
        1, default_block_size, 0, 0, subspace_vectors->get_size()[0],
        subspace_vectors->get_size()[1],
        as_hip_type(subspace_vectors->get_values()),
        subspace_vectors->get_stride());
}


template <typename ValueType>
void solve_lower_triangular(const size_type nrhs,
                            const matrix::Dense<ValueType> *m,
                            const matrix::Dense<ValueType> *f,
                            matrix::Dense<ValueType> *c,
                            const Array<stopping_status> *stop_status)
{
    const auto subspace_dim = m->get_size()[0];

    const auto grid_dim = ceildiv(nrhs, default_block_size);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(solve_lower_triangular_kernel), grid_dim,
                       default_block_size, 0, 0, subspace_dim, nrhs,
                       as_hip_type(m->get_const_values()), m->get_stride(),
                       as_hip_type(f->get_const_values()), f->get_stride(),
                       as_hip_type(c->get_values()), c->get_stride(),
                       as_hip_type(stop_status->get_const_data()));
}


template <typename ValueType>
void update_g_and_u(std::shared_ptr<const HipExecutor> exec,
                    const size_type nrhs, const size_type k,
                    const matrix::Dense<ValueType> *p,
                    const matrix::Dense<ValueType> *m,
                    matrix::Dense<ValueType> *alpha,
                    matrix::Dense<ValueType> *g, matrix::Dense<ValueType> *g_k,
                    matrix::Dense<ValueType> *u,
                    const Array<stopping_status> *stop_status)
{
    const auto size = g->get_size()[0];
    const auto p_stride = p->get_stride();

    const dim3 grid_dim(ceildiv(nrhs, default_dot_dim),
                        exec->get_num_multiprocessor() * 2);
    const dim3 block_dim(default_dot_dim, default_dot_dim);

    for (size_type i = 0; i < k; i++) {
        const auto p_i = p->get_const_values() + i * p_stride;
        if (nrhs > 1 || is_complex<ValueType>()) {
            components::fill_array(exec, alpha->get_values(), nrhs,
                                   zero<ValueType>());
            hipLaunchKernelGGL(
                multidot_kernel, grid_dim, block_dim, 0, 0, size, nrhs,
                as_hip_type(p_i), as_hip_type(g_k->get_values()),
                g_k->get_stride(), as_hip_type(alpha->get_values()),
                as_hip_type(stop_status->get_const_data()));
        } else {
            hipblas::dot(exec->get_hipblas_handle(), size, p_i, 1,
                         g_k->get_values(), g_k->get_stride(),
                         alpha->get_values());
        }
        hipLaunchKernelGGL(
            update_g_k_and_u_kernel<default_block_size>,
            ceildiv(size * g_k->get_stride(), default_block_size),
            default_block_size, 0, 0, k, i, size, nrhs,
            as_hip_type(alpha->get_const_values()),
            as_hip_type(m->get_const_values()), m->get_stride(),
            as_hip_type(g->get_const_values()), g->get_stride(),
            as_hip_type(g_k->get_values()), g_k->get_stride(),
            as_hip_type(u->get_values()), u->get_stride(),
            as_hip_type(stop_status->get_const_data()));
    }
    hipLaunchKernelGGL(update_g_kernel<default_block_size>,
                       ceildiv(size * g_k->get_stride(), default_block_size),
                       default_block_size, 0, 0, k, size, nrhs,
                       as_hip_type(g_k->get_const_values()), g_k->get_stride(),
                       as_hip_type(g->get_values()), g->get_stride(),
                       as_hip_type(stop_status->get_const_data()));
}


template <typename ValueType>
void update_m(std::shared_ptr<const HipExecutor> exec, const size_type nrhs,
              const size_type k, const matrix::Dense<ValueType> *p,
              const matrix::Dense<ValueType> *g_k, matrix::Dense<ValueType> *m,
              const Array<stopping_status> *stop_status)
{
    const auto size = g_k->get_size()[0];
    const auto subspace_dim = m->get_size()[0];
    const auto p_stride = p->get_stride();
    const auto m_stride = m->get_stride();

    const dim3 grid_dim(ceildiv(nrhs, default_dot_dim),
                        exec->get_num_multiprocessor() * 2);
    const dim3 block_dim(default_dot_dim, default_dot_dim);

    for (size_type i = k; i < subspace_dim; i++) {
        const auto p_i = p->get_const_values() + i * p_stride;
        auto m_i = m->get_values() + i * m_stride + k * nrhs;
        if (nrhs > 1 || is_complex<ValueType>()) {
            components::fill_array(exec, m_i, nrhs, zero<ValueType>());
            hipLaunchKernelGGL(multidot_kernel, grid_dim, block_dim, 0, 0, size,
                               nrhs, as_hip_type(p_i),
                               as_hip_type(g_k->get_const_values()),
                               g_k->get_stride(), as_hip_type(m_i),
                               as_hip_type(stop_status->get_const_data()));
        } else {
            hipblas::dot(exec->get_hipblas_handle(), size, p_i, 1,
                         g_k->get_const_values(), g_k->get_stride(), m_i);
        }
    }
}


template <typename ValueType>
void update_x_r_and_f(std::shared_ptr<const HipExecutor> exec,
                      const size_type nrhs, const size_type k,
                      const matrix::Dense<ValueType> *m,
                      const matrix::Dense<ValueType> *g,
                      const matrix::Dense<ValueType> *u,
                      matrix::Dense<ValueType> *f, matrix::Dense<ValueType> *r,
                      matrix::Dense<ValueType> *x,
                      const Array<stopping_status> *stop_status)
{
    const auto size = x->get_size()[0];
    const auto subspace_dim = m->get_size()[0];

    const auto grid_dim = ceildiv(size * x->get_stride(), default_block_size);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(update_x_r_and_f_kernel), grid_dim,
                       default_block_size, 0, 0, k, size, subspace_dim, nrhs,
                       as_hip_type(m->get_const_values()), m->get_stride(),
                       as_hip_type(g->get_const_values()), g->get_stride(),
                       as_hip_type(u->get_const_values()), u->get_stride(),
                       as_hip_type(f->get_values()), f->get_stride(),
                       as_hip_type(r->get_values()), r->get_stride(),
                       as_hip_type(x->get_values()), x->get_stride(),
                       as_hip_type(stop_status->get_const_data()));
    components::fill_array(exec, f->get_values() + k * f->get_stride(), nrhs,
                           zero<ValueType>());
}


}  // namespace


template <typename ValueType>
void initialize(std::shared_ptr<const HipExecutor> exec, const size_type nrhs,
                matrix::Dense<ValueType> *m,
                matrix::Dense<ValueType> *subspace_vectors, bool deterministic,
                Array<stopping_status> *stop_status)
{
    initialize_m(nrhs, m, stop_status);
    initialize_subspace_vectors(subspace_vectors, deterministic);
    orthonormalize_subspace_vectors(subspace_vectors);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const HipExecutor> exec, const size_type nrhs,
            const size_type k, const matrix::Dense<ValueType> *m,
            const matrix::Dense<ValueType> *f,
            const matrix::Dense<ValueType> *residual,
            const matrix::Dense<ValueType> *g, matrix::Dense<ValueType> *c,
            matrix::Dense<ValueType> *v,
            const Array<stopping_status> *stop_status)
{
    solve_lower_triangular(nrhs, m, f, c, stop_status);

    const auto num_rows = v->get_size()[0];
    const auto subspace_dim = m->get_size()[0];

    const auto grid_dim =
        ceildiv(v->get_stride() * num_rows, default_block_size);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(step_1_kernel), grid_dim, default_block_size, 0, 0, k,
        num_rows, subspace_dim, nrhs, as_hip_type(residual->get_const_values()),
        residual->get_stride(), as_hip_type(c->get_const_values()),
        c->get_stride(), as_hip_type(g->get_const_values()), g->get_stride(),
        as_hip_type(v->get_values()), v->get_stride(),
        as_hip_type(stop_status->get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const HipExecutor> exec, const size_type nrhs,
            const size_type k, const matrix::Dense<ValueType> *omega,
            const matrix::Dense<ValueType> *preconditioned_vector,
            const matrix::Dense<ValueType> *c, matrix::Dense<ValueType> *u,
            const Array<stopping_status> *stop_status)
{
    const auto num_rows = preconditioned_vector->get_size()[0];
    const auto subspace_dim = u->get_size()[1] / nrhs;

    const auto grid_dim =
        ceildiv(u->get_stride() * num_rows, default_block_size);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(step_2_kernel), grid_dim, default_block_size, 0, 0, k,
        num_rows, subspace_dim, nrhs, as_hip_type(omega->get_const_values()),
        as_hip_type(preconditioned_vector->get_const_values()),
        preconditioned_vector->get_stride(), as_hip_type(c->get_const_values()),
        c->get_stride(), as_hip_type(u->get_values()), u->get_stride(),
        as_hip_type(stop_status->get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_2_KERNEL);


template <typename ValueType>
void step_3(std::shared_ptr<const HipExecutor> exec, const size_type nrhs,
            const size_type k, const matrix::Dense<ValueType> *p,
            matrix::Dense<ValueType> *g, matrix::Dense<ValueType> *g_k,
            matrix::Dense<ValueType> *u, matrix::Dense<ValueType> *m,
            matrix::Dense<ValueType> *f, matrix::Dense<ValueType> *alpha,
            matrix::Dense<ValueType> *residual, matrix::Dense<ValueType> *x,
            const Array<stopping_status> *stop_status)
{
    update_g_and_u(exec, nrhs, k, p, m, alpha, g, g_k, u, stop_status);
    update_m(exec, nrhs, k, p, g_k, m, stop_status);
    update_x_r_and_f(exec, nrhs, k, m, g, u, f, residual, x, stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_3_KERNEL);


template <typename ValueType>
void compute_omega(
    std::shared_ptr<const HipExecutor> exec, const size_type nrhs,
    const remove_complex<ValueType> kappa, const matrix::Dense<ValueType> *tht,
    const matrix::Dense<remove_complex<ValueType>> *residual_norm,
    matrix::Dense<ValueType> *omega, const Array<stopping_status> *stop_status)
{
    const auto grid_dim = ceildiv(nrhs, config::warp_size);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_omega_kernel), grid_dim,
                       config::warp_size, 0, 0, nrhs, kappa,
                       as_hip_type(tht->get_const_values()),
                       as_hip_type(residual_norm->get_const_values()),
                       as_hip_type(omega->get_values()),
                       as_hip_type(stop_status->get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_COMPUTE_OMEGA_KERNEL);


}  // namespace idr
}  // namespace hip
}  // namespace kernels
}  // namespace gko
