// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/idr_kernels.hpp"


#include <ctime>
#include <random>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/cublas_bindings.hpp"
#include "cuda/base/curand_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The IDR solver namespace.
 *
 * @ingroup idr
 */
namespace idr {


constexpr int default_block_size = 512;
constexpr int default_dot_dim = 32;
constexpr int default_dot_size = default_dot_dim * default_dot_dim;


#include "common/cuda_hip/solver/idr_kernels.hpp.inc"


namespace {


template <typename ValueType>
void initialize_m(std::shared_ptr<const DefaultExecutor> exec,
                  const size_type nrhs, matrix::Dense<ValueType>* m,
                  array<stopping_status>* stop_status)
{
    const auto subspace_dim = m->get_size()[0];
    const auto m_stride = m->get_stride();

    const auto grid_dim = ceildiv(m_stride * subspace_dim, default_block_size);
    initialize_m_kernel<<<grid_dim, default_block_size, 0,
                          exec->get_stream()>>>(
        subspace_dim, nrhs, as_device_type(m->get_values()), m_stride,
        as_device_type(stop_status->get_data()));
}


template <typename ValueType>
void initialize_subspace_vectors(std::shared_ptr<const DefaultExecutor> exec,
                                 matrix::Dense<ValueType>* subspace_vectors,
                                 bool deterministic)
{
    if (!deterministic) {
        auto gen = curand::rand_generator(std::random_device{}(),
                                          CURAND_RNG_PSEUDO_DEFAULT,
                                          exec->get_stream());
        curand::rand_vector(
            gen,
            subspace_vectors->get_size()[0] * subspace_vectors->get_stride(),
            0.0, 1.0, subspace_vectors->get_values());
        curand::destroy(gen);
    }
}


template <typename ValueType>
void orthonormalize_subspace_vectors(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::Dense<ValueType>* subspace_vectors)
{
    orthonormalize_subspace_vectors_kernel<default_block_size>
        <<<1, default_block_size, 0, exec->get_stream()>>>(
            subspace_vectors->get_size()[0], subspace_vectors->get_size()[1],
            as_device_type(subspace_vectors->get_values()),
            subspace_vectors->get_stride());
}


template <typename ValueType>
void solve_lower_triangular(std::shared_ptr<const DefaultExecutor> exec,
                            const size_type nrhs,
                            const matrix::Dense<ValueType>* m,
                            const matrix::Dense<ValueType>* f,
                            matrix::Dense<ValueType>* c,
                            const array<stopping_status>* stop_status)
{
    const auto subspace_dim = m->get_size()[0];

    const auto grid_dim = ceildiv(nrhs, default_block_size);
    solve_lower_triangular_kernel<<<grid_dim, default_block_size, 0,
                                    exec->get_stream()>>>(
        subspace_dim, nrhs, as_device_type(m->get_const_values()),
        m->get_stride(), as_device_type(f->get_const_values()), f->get_stride(),
        as_device_type(c->get_values()), c->get_stride(),
        stop_status->get_const_data());
}


template <typename ValueType>
void update_g_and_u(std::shared_ptr<const DefaultExecutor> exec,
                    const size_type nrhs, const size_type k,
                    const matrix::Dense<ValueType>* p,
                    const matrix::Dense<ValueType>* m,
                    matrix::Dense<ValueType>* alpha,
                    matrix::Dense<ValueType>* g, matrix::Dense<ValueType>* g_k,
                    matrix::Dense<ValueType>* u,
                    const array<stopping_status>* stop_status)
{
    if (nrhs == 0) {
        return;
    }
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
            multidot_kernel<<<grid_dim, block_dim, 0, exec->get_stream()>>>(
                size, nrhs, as_device_type(p_i),
                as_device_type(g_k->get_values()), g_k->get_stride(),
                as_device_type(alpha->get_values()),
                stop_status->get_const_data());
        } else {
            cublas::dot(exec->get_cublas_handle(), size, p_i, 1,
                        g_k->get_values(), g_k->get_stride(),
                        alpha->get_values());
        }
        update_g_k_and_u_kernel<default_block_size>
            <<<ceildiv(size * g_k->get_stride(), default_block_size),
               default_block_size, 0, exec->get_stream()>>>(
                k, i, size, nrhs, as_device_type(alpha->get_const_values()),
                as_device_type(m->get_const_values()), m->get_stride(),
                as_device_type(g->get_const_values()), g->get_stride(),
                as_device_type(g_k->get_values()), g_k->get_stride(),
                as_device_type(u->get_values()), u->get_stride(),
                stop_status->get_const_data());
    }
    update_g_kernel<default_block_size>
        <<<ceildiv(size * g_k->get_stride(), default_block_size),
           default_block_size, 0, exec->get_stream()>>>(
            k, size, nrhs, as_device_type(g_k->get_const_values()),
            g_k->get_stride(), as_device_type(g->get_values()), g->get_stride(),
            stop_status->get_const_data());
}


template <typename ValueType>
void update_m(std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,
              const size_type k, const matrix::Dense<ValueType>* p,
              const matrix::Dense<ValueType>* g_k, matrix::Dense<ValueType>* m,
              const array<stopping_status>* stop_status)
{
    if (nrhs == 0) {
        return;
    }
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
            multidot_kernel<<<grid_dim, block_dim, 0, exec->get_stream()>>>(
                size, nrhs, as_device_type(p_i),
                as_device_type(g_k->get_const_values()), g_k->get_stride(),
                as_device_type(m_i), stop_status->get_const_data());
        } else {
            cublas::dot(exec->get_cublas_handle(), size, p_i, 1,
                        g_k->get_const_values(), g_k->get_stride(), m_i);
        }
    }
}


template <typename ValueType>
void update_x_r_and_f(std::shared_ptr<const DefaultExecutor> exec,
                      const size_type nrhs, const size_type k,
                      const matrix::Dense<ValueType>* m,
                      const matrix::Dense<ValueType>* g,
                      const matrix::Dense<ValueType>* u,
                      matrix::Dense<ValueType>* f, matrix::Dense<ValueType>* r,
                      matrix::Dense<ValueType>* x,
                      const array<stopping_status>* stop_status)
{
    const auto size = x->get_size()[0];
    const auto subspace_dim = m->get_size()[0];

    const auto grid_dim = ceildiv(size * x->get_stride(), default_block_size);
    update_x_r_and_f_kernel<<<grid_dim, default_block_size, 0,
                              exec->get_stream()>>>(
        k, size, subspace_dim, nrhs, as_device_type(m->get_const_values()),
        m->get_stride(), as_device_type(g->get_const_values()), g->get_stride(),
        as_device_type(u->get_const_values()), u->get_stride(),
        as_device_type(f->get_values()), f->get_stride(),
        as_device_type(r->get_values()), r->get_stride(),
        as_device_type(x->get_values()), x->get_stride(),
        stop_status->get_const_data());
    components::fill_array(exec, f->get_values() + k * f->get_stride(), nrhs,
                           zero<ValueType>());
}


}  // namespace


template <typename ValueType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const size_type nrhs, matrix::Dense<ValueType>* m,
                matrix::Dense<ValueType>* subspace_vectors, bool deterministic,
                array<stopping_status>* stop_status)
{
    initialize_m(exec, nrhs, m, stop_status);
    initialize_subspace_vectors(exec, subspace_vectors, deterministic);
    orthonormalize_subspace_vectors(exec, subspace_vectors);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,
            const size_type k, const matrix::Dense<ValueType>* m,
            const matrix::Dense<ValueType>* f,
            const matrix::Dense<ValueType>* residual,
            const matrix::Dense<ValueType>* g, matrix::Dense<ValueType>* c,
            matrix::Dense<ValueType>* v,
            const array<stopping_status>* stop_status)
{
    solve_lower_triangular(exec, nrhs, m, f, c, stop_status);

    const auto num_rows = v->get_size()[0];
    const auto subspace_dim = m->get_size()[0];

    const auto grid_dim = ceildiv(nrhs * num_rows, default_block_size);
    step_1_kernel<<<grid_dim, default_block_size, 0, exec->get_stream()>>>(
        k, num_rows, subspace_dim, nrhs,
        as_device_type(residual->get_const_values()), residual->get_stride(),
        as_device_type(c->get_const_values()), c->get_stride(),
        as_device_type(g->get_const_values()), g->get_stride(),
        as_device_type(v->get_values()), v->get_stride(),
        stop_status->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,
            const size_type k, const matrix::Dense<ValueType>* omega,
            const matrix::Dense<ValueType>* preconditioned_vector,
            const matrix::Dense<ValueType>* c, matrix::Dense<ValueType>* u,
            const array<stopping_status>* stop_status)
{
    if (nrhs == 0) {
        return;
    }
    const auto num_rows = preconditioned_vector->get_size()[0];
    const auto subspace_dim = u->get_size()[1] / nrhs;

    const auto grid_dim = ceildiv(nrhs * num_rows, default_block_size);
    step_2_kernel<<<grid_dim, default_block_size, 0, exec->get_stream()>>>(
        k, num_rows, subspace_dim, nrhs,
        as_device_type(omega->get_const_values()),
        as_device_type(preconditioned_vector->get_const_values()),
        preconditioned_vector->get_stride(),
        as_device_type(c->get_const_values()), c->get_stride(),
        as_device_type(u->get_values()), u->get_stride(),
        stop_status->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_2_KERNEL);


template <typename ValueType>
void step_3(std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,
            const size_type k, const matrix::Dense<ValueType>* p,
            matrix::Dense<ValueType>* g, matrix::Dense<ValueType>* g_k,
            matrix::Dense<ValueType>* u, matrix::Dense<ValueType>* m,
            matrix::Dense<ValueType>* f, matrix::Dense<ValueType>* alpha,
            matrix::Dense<ValueType>* residual, matrix::Dense<ValueType>* x,
            const array<stopping_status>* stop_status)
{
    update_g_and_u(exec, nrhs, k, p, m, alpha, g, g_k, u, stop_status);
    update_m(exec, nrhs, k, p, g_k, m, stop_status);
    update_x_r_and_f(exec, nrhs, k, m, g, u, f, residual, x, stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_3_KERNEL);


template <typename ValueType>
void compute_omega(
    std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,
    const remove_complex<ValueType> kappa, const matrix::Dense<ValueType>* tht,
    const matrix::Dense<remove_complex<ValueType>>* residual_norm,
    matrix::Dense<ValueType>* omega, const array<stopping_status>* stop_status)
{
    const auto grid_dim = ceildiv(nrhs, config::warp_size);
    compute_omega_kernel<<<grid_dim, config::warp_size, 0,
                           exec->get_stream()>>>(
        nrhs, as_device_type(kappa), as_device_type(tht->get_const_values()),
        as_device_type(residual_norm->get_const_values()),
        as_device_type(omega->get_values()), stop_status->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_COMPUTE_OMEGA_KERNEL);


}  // namespace idr
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
