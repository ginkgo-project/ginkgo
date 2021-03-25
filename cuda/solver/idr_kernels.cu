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


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "accessor/range.hpp"
#include "accessor/reduced_row_major.hpp"
#include "core/components/fill_array.hpp"
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


#include "common/solver/idr_kernels.hpp.inc"


// Specialization, so the Accessor can use the same function as regular pointers
template <int dim, typename Type1, typename Type2>
GKO_INLINE auto as_cuda_accessor(
    const acc::range<acc::reduced_row_major<dim, Type1, Type2>> &acc)
{
    return acc::range<
        acc::reduced_row_major<dim, cuda_type<Type1>, cuda_type<Type2>>>(
        acc.get_accessor().get_size(),
        as_cuda_type(acc.get_accessor().get_stored_data()),
        acc.get_accessor().get_stride());
}

template <int dim, typename Type1, typename Type2, size_type mask>
GKO_INLINE auto as_cuda_accessor(
    const acc::range<acc::scaled_reduced_row_major<dim, Type1, Type2, mask>>
        &acc)
{
    return acc::range<acc::scaled_reduced_row_major<dim, cuda_type<Type1>,
                                                    cuda_type<Type2>, mask>>(
        acc.get_accessor().get_size(),
        as_cuda_type(acc.get_accessor().get_stored_data()),
        acc.get_accessor().get_storage_stride(),
        as_cuda_type(acc.get_accessor().get_scalar()),
        acc.get_accessor().get_scalar_stride());
}


namespace {


template <typename ValueType>
void initialize_m(const size_type nrhs, matrix::Dense<ValueType> *m,
                  Array<stopping_status> *stop_status)
{
    const auto subspace_dim = m->get_size()[0];
    const auto m_stride = m->get_stride();

    const auto grid_dim = ceildiv(m_stride * subspace_dim, default_block_size);
    initialize_m_kernel<<<grid_dim, default_block_size>>>(
        subspace_dim, nrhs, as_cuda_type(m->get_values()), m_stride,
        as_cuda_type(stop_status->get_data()));
}


template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<!is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(Distribution &&dist, Generator &&gen)
{
    return ValueType{dist(gen)};
}


template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(Distribution &&dist, Generator &&gen)
{
    return ValueType(dist(gen), dist(gen));
}


template <typename ValueType>
void initialize_subspace_vectors(ValueType *subspace_vectors, gko::dim<2> size,
                                 bool deterministic)
{
    /*if (!deterministic || deterministic) {
        auto dist = std::normal_distribution<>(0.0, 1.0);
        auto gen = std::ranlux48(15);
        for (size_type i = 0; i < size[0]; i++) {
            for (size_type j = 0; j < size[1]; j++) {
                subspace_vectors[i * size[1] + j] =
                    get_rand_value<ValueType>(dist, gen);
            }
        }
    } else {*/
    auto gen = curand::rand_generator(time(NULL), CURAND_RNG_PSEUDO_DEFAULT);
    curand::rand_vector(gen, size[0] * size[1], 0.0, 1.0, subspace_vectors);
    //}
}


template <typename Acc>
void orthonormalize_subspace_vectors(Acc subspace_vectors)
{
    orthonormalize_subspace_vectors_kernel<default_block_size>
        <<<1, default_block_size>>>(subspace_vectors->get_size()[0],
                                    subspace_vectors->get_size()[1],
                                    as_cuda_accessor(subspace_vectors),
                                    subspace_vectors->get_size()[1]);
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
    solve_lower_triangular_kernel<<<grid_dim, default_block_size>>>(
        subspace_dim, nrhs, as_cuda_type(m->get_const_values()),
        m->get_stride(), as_cuda_type(f->get_const_values()), f->get_stride(),
        as_cuda_type(c->get_values()), c->get_stride(),
        as_cuda_type(stop_status->get_const_data()));
}


template <typename ValueType, typename Acc>
void update_g_and_u(std::shared_ptr<const CudaExecutor> exec,
                    const size_type nrhs, const size_type k, Acc p,
                    const matrix::Dense<ValueType> *m,
                    matrix::Dense<ValueType> *alpha,
                    matrix::Dense<ValueType> *g, matrix::Dense<ValueType> *g_k,
                    matrix::Dense<ValueType> *u,
                    const Array<stopping_status> *stop_status)
{
    const auto size = g->get_size()[0];
    const auto p_stride = p.get_accessor().get_stride();

    const dim3 grid_dim(ceildiv(nrhs, default_dot_dim),
                        exec->get_num_multiprocessor() * 2);
    const dim3 block_dim(default_dot_dim, default_dot_dim);

    for (size_type i = 0; i < k; i++) {
        components::fill_array(exec, alpha->get_values(), nrhs,
                               zero<ValueType>());
        multidot_kernel<<<grid_dim, block_dim>>>(
            size, nrhs, as_cuda_accessor(p), i, as_cuda_type(g_k->get_values()),
            g_k->get_stride(), as_cuda_type(alpha->get_values()),
            as_cuda_type(stop_status->get_const_data()));
        update_g_k_and_u_kernel<default_block_size>
            <<<ceildiv(size * g_k->get_stride(), default_block_size),
               default_block_size>>>(
                k, i, size, nrhs, as_cuda_type(alpha->get_const_values()),
                as_cuda_type(m->get_const_values()), m->get_stride(),
                as_cuda_type(g->get_const_values()), g->get_stride(),
                as_cuda_type(g_k->get_values()), g_k->get_stride(),
                as_cuda_type(u->get_values()), u->get_stride(),
                as_cuda_type(stop_status->get_const_data()));
    }
    update_g_kernel<default_block_size>
        <<<ceildiv(size * g_k->get_stride(), default_block_size),
           default_block_size>>>(
            k, size, nrhs, as_cuda_type(g_k->get_const_values()),
            g_k->get_stride(), as_cuda_type(g->get_values()), g->get_stride(),
            as_cuda_type(stop_status->get_const_data()));
}


template <typename ValueType, typename Acc>
void update_m(std::shared_ptr<const CudaExecutor> exec, const size_type nrhs,
              const size_type k, Acc p, const matrix::Dense<ValueType> *g_k,
              matrix::Dense<ValueType> *m,
              const Array<stopping_status> *stop_status)
{
    const auto size = g_k->get_size()[0];
    const auto subspace_dim = m->get_size()[0];
    const auto p_stride = p.get_accessor().get_stride();
    const auto m_stride = m->get_stride();

    const dim3 grid_dim(ceildiv(nrhs, default_dot_dim),
                        exec->get_num_multiprocessor() * 2);
    const dim3 block_dim(default_dot_dim, default_dot_dim);

    for (size_type i = k; i < subspace_dim; i++) {
        auto m_i = m->get_values() + i * m_stride + k * nrhs;
        components::fill_array(exec, m_i, nrhs, zero<ValueType>());
        multidot_kernel<<<grid_dim, block_dim>>>(
            size, nrhs, as_cuda_accessor(p), i,
            as_cuda_type(g_k->get_const_values()), g_k->get_stride(),
            as_cuda_type(m_i), as_cuda_type(stop_status->get_const_data()));
    }
}


template <typename ValueType>
void update_x_r_and_f(std::shared_ptr<const CudaExecutor> exec,
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
    update_x_r_and_f_kernel<<<grid_dim, default_block_size>>>(
        k, size, subspace_dim, nrhs, as_cuda_type(m->get_const_values()),
        m->get_stride(), as_cuda_type(g->get_const_values()), g->get_stride(),
        as_cuda_type(u->get_const_values()), u->get_stride(),
        as_cuda_type(f->get_values()), f->get_stride(),
        as_cuda_type(r->get_values()), r->get_stride(),
        as_cuda_type(x->get_values()), x->get_stride(),
        as_cuda_type(stop_status->get_const_data()));
    components::fill_array(exec, f->get_values() + k * f->get_stride(), nrhs,
                           zero<ValueType>());
}


}  // namespace


template <typename ValueType, typename Acc>
void initialize(std::shared_ptr<const CudaExecutor> exec, const size_type nrhs,
                matrix::Dense<ValueType> *m, Acc subspace_vectors,
                bool deterministic, Array<stopping_status> *stop_status)
{
    initialize_m(nrhs, m, stop_status);
    initialize_subspace_vectors(
        subspace_vectors.get_accessor().get_stored_data(),
        gko::dim<2>{subspace_vectors->get_size()[0],
                    subspace_vectors->get_size()[1]},
        deterministic);
    orthonormalize_subspace_vectors(subspace_vectors);
}

GKO_INSTANTIATE_FOR_EACH_IDR_TYPE(GKO_DECLARE_IDR_INITIALIZE_KERNEL);


template <typename ValueType, typename Acc>
void apply_subspace(std::shared_ptr<const CudaExecutor> exec,
                    Acc subspace_vectors,
                    const matrix::Dense<ValueType> *residual,
                    matrix::Dense<ValueType> *f)
{
    const auto grid_dim = ceildiv(
        residual->get_size()[0] * residual->get_size()[1] * config::warp_size,
        default_block_size);

    apply_subspace_kernel<config::warp_size><<<grid_dim, default_block_size>>>(
        f->get_size()[0], residual->get_size()[0], residual->get_size()[1],
        as_cuda_accessor(subspace_vectors),
        as_cuda_type(residual->get_const_values()),
        as_cuda_type(f->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_IDR_TYPE(GKO_DECLARE_IDR_APPLY_SUBSPACE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const CudaExecutor> exec, const size_type nrhs,
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

    const auto grid_dim = ceildiv(nrhs * num_rows, default_block_size);
    step_1_kernel<<<grid_dim, default_block_size>>>(
        k, num_rows, subspace_dim, nrhs,
        as_cuda_type(residual->get_const_values()), residual->get_stride(),
        as_cuda_type(c->get_const_values()), c->get_stride(),
        as_cuda_type(g->get_const_values()), g->get_stride(),
        as_cuda_type(v->get_values()), v->get_stride(),
        as_cuda_type(stop_status->get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const CudaExecutor> exec, const size_type nrhs,
            const size_type k, const matrix::Dense<ValueType> *omega,
            const matrix::Dense<ValueType> *preconditioned_vector,
            const matrix::Dense<ValueType> *c, matrix::Dense<ValueType> *u,
            const Array<stopping_status> *stop_status)
{
    const auto num_rows = preconditioned_vector->get_size()[0];
    const auto subspace_dim = u->get_size()[1] / nrhs;

    const auto grid_dim = ceildiv(nrhs * num_rows, default_block_size);
    step_2_kernel<<<grid_dim, default_block_size>>>(
        k, num_rows, subspace_dim, nrhs,
        as_cuda_type(omega->get_const_values()),
        as_cuda_type(preconditioned_vector->get_const_values()),
        preconditioned_vector->get_stride(),
        as_cuda_type(c->get_const_values()), c->get_stride(),
        as_cuda_type(u->get_values()), u->get_stride(),
        as_cuda_type(stop_status->get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_2_KERNEL);


template <typename ValueType, typename Acc>
void step_3(std::shared_ptr<const CudaExecutor> exec, const size_type nrhs,
            const size_type k, Acc p, matrix::Dense<ValueType> *g,
            matrix::Dense<ValueType> *g_k, matrix::Dense<ValueType> *u,
            matrix::Dense<ValueType> *m, matrix::Dense<ValueType> *f,
            matrix::Dense<ValueType> *alpha, matrix::Dense<ValueType> *residual,
            matrix::Dense<ValueType> *x,
            const Array<stopping_status> *stop_status)
{
    update_g_and_u(exec, nrhs, k, p, m, alpha, g, g_k, u, stop_status);
    update_m(exec, nrhs, k, p, g_k, m, stop_status);
    update_x_r_and_f(exec, nrhs, k, m, g, u, f, residual, x, stop_status);
}

GKO_INSTANTIATE_FOR_EACH_IDR_TYPE(GKO_DECLARE_IDR_STEP_3_KERNEL);


template <typename ValueType>
void compute_omega(
    std::shared_ptr<const CudaExecutor> exec, const size_type nrhs,
    const remove_complex<ValueType> kappa, const matrix::Dense<ValueType> *tht,
    const matrix::Dense<remove_complex<ValueType>> *residual_norm,
    matrix::Dense<ValueType> *omega, const Array<stopping_status> *stop_status)
{
    const auto grid_dim = ceildiv(nrhs, config::warp_size);
    compute_omega_kernel<<<grid_dim, config::warp_size>>>(
        nrhs, kappa, as_cuda_type(tht->get_const_values()),
        as_cuda_type(residual_norm->get_const_values()),
        as_cuda_type(omega->get_values()),
        as_cuda_type(stop_status->get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_COMPUTE_OMEGA_KERNEL);


template <typename ValueType>
void compute_gamma(std::shared_ptr<const CudaExecutor> exec,
                   const size_type nrhs, const matrix::Dense<ValueType> *tht,
                   matrix::Dense<ValueType> *gamma,
                   matrix::Dense<ValueType> *one_minus_gamma,
                   const Array<stopping_status> *stop_status)
{
    const auto grid_dim = ceildiv(nrhs, config::warp_size);
    compute_gamma_kernel<<<grid_dim, config::warp_size>>>(
        nrhs, as_cuda_type(tht->get_const_values()),
        as_cuda_type(gamma->get_values()),
        as_cuda_type(one_minus_gamma->get_values()),
        as_cuda_type(stop_status->get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_COMPUTE_GAMMA_KERNEL);


}  // namespace idr
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
