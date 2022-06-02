/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/solver/idr.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/distributed/helpers.hpp"
#include "core/solver/idr_kernels.hpp"


namespace gko {
namespace solver {
namespace idr {
namespace {


GKO_REGISTER_OPERATION(initialize, idr::initialize);
GKO_REGISTER_OPERATION(step_1, idr::step_1);
GKO_REGISTER_OPERATION(step_2, idr::step_2);
GKO_REGISTER_OPERATION(step_3, idr::step_3);
GKO_REGISTER_OPERATION(compute_omega, idr::compute_omega);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);


}  // anonymous namespace
}  // namespace idr


template <typename ValueType>
std::unique_ptr<LinOp> Idr<ValueType>::transpose() const
{
    return build()
        .with_generated_preconditioner(
            share(as<Transposable>(this->get_preconditioner())->transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .on(this->get_executor())
        ->generate(
            share(as<Transposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<LinOp> Idr<ValueType>::conj_transpose() const
{
    return build()
        .with_generated_preconditioner(share(
            as<Transposable>(this->get_preconditioner())->conj_transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType>
template <typename VectorType>
void Idr<ValueType>::iterate(const VectorType* dense_b,
                             VectorType* dense_x) const
{
    using std::swap;
    using SubspaceType = typename VectorType::value_type;
    using Vector = VectorType;
    using LocalVector = matrix::Dense<SubspaceType>;
    using NormVector = matrix::Dense<remove_complex<ValueType>>;

    auto exec = this->get_executor();

    array<char> reduction_tmp{exec};

    auto one_op =
        initialize<matrix::Dense<ValueType>>({one<ValueType>()}, exec);
    auto neg_one_op =
        initialize<matrix::Dense<ValueType>>({-one<ValueType>()}, exec);
    auto subspace_neg_one_op = initialize<Vector>({-one<SubspaceType>()}, exec);

    constexpr uint8 RelativeStoppingId{1};

    const auto problem_size = this->get_size()[0];
    const auto nrhs = dense_b->get_size()[1];
    const auto subspace_dim = this->get_subspace_dim();
    const auto is_deterministic = this->get_deterministic();
    const auto kappa = this->get_kappa();

    auto residual = detail::create_with_config_of(dense_b);
    auto v = detail::create_with_config_of(dense_b);
    auto t = detail::create_with_config_of(dense_b);
    auto helper = detail::create_with_config_of(dense_b);

    auto m =
        Vector ::create(exec, gko::dim<2>{subspace_dim, subspace_dim * nrhs});

    auto g =
        Vector ::create(exec, gko::dim<2>{problem_size, subspace_dim * nrhs});
    auto u =
        Vector ::create(exec, gko::dim<2>{problem_size, subspace_dim * nrhs});

    auto f = Vector::create(exec, gko::dim<2>{subspace_dim, nrhs});
    auto c = Vector::create(exec, gko::dim<2>{subspace_dim, nrhs});

    auto omega = LocalVector ::create(exec, gko::dim<2>{1, nrhs});
    auto residual_norm = NormVector::create(exec, dim<2>{1, nrhs});
    auto tht = LocalVector ::create(exec, dim<2>{1, nrhs});
    auto alpha = LocalVector ::create(exec, gko::dim<2>{1, nrhs});

    bool one_changed{};
    array<stopping_status> stop_status(exec, nrhs);

    // The dense matrix containing the randomly generated subspace vectors.
    // Stored in column major order and complex conjugated. So, if the
    // matrix containing the subspace vectors in row major order is called P,
    // subspace_vectors actually contains P^H.
    auto subspace_vectors =
        Vector::create(exec, gko::dim<2>(subspace_dim, problem_size));

    // Initialization
    // m = identity
    exec->run(idr::make_initialize(nrhs, detail::get_local(m.get()),
                                   detail::get_local(subspace_vectors.get()),
                                   is_deterministic, &stop_status));

    // omega = 1
    exec->run(
        idr::make_fill_array(omega->get_values(), nrhs, one<SubspaceType>()));

    // residual = b - Ax
    residual->copy_from(dense_b);
    this->get_system_matrix()->apply(neg_one_op.get(), dense_x, one_op.get(),
                                     residual.get());
    residual->compute_norm2(residual_norm.get(), reduction_tmp);

    // g = u = 0
    exec->run(idr::make_fill_array(
        g->get_values(), problem_size * g->get_stride(), zero<SubspaceType>()));
    exec->run(idr::make_fill_array(
        u->get_values(), problem_size * u->get_stride(), zero<SubspaceType>()));


    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
        residual.get());

    int total_iter = -1;

    /* Memory movement summary for iteration with subspace dimension s
     * Per iteration:
     * (11/2s^2+31/2s+18)n * values + (s+1) * matrix/preconditioner storage
     * (s+1)x SpMV:                2(s+1)n * values + (s+1) * storage
     * (s+1)x Preconditioner:      2(s+1)n * values + (s+1) * storage
     * 1x multidot (gemv)           (s+1)n
     * sx step 1 (fused axpys) s(s/2+5/2)n = sum k=[0,s) of (s-k+2)n
     * sx step 2 (fused axpys) s(s/2+5/2)n = sum k=[0,s) of (s-k+2)n
     * sx step 3:            s(9/2s+11/2)n = sum k=[0,s) of (8k+2+s-k+1+6)n
     *       1x orthogonalize g+u      (8k+2)n in iteration k (0-based)
     *       1x multidot (gemv)       (s-k+1)n in iteration k (0-based)
     *       2x axpy                        6n
     * 1x dot                           2n
     * 2x norm2                         2n
     * 1x scale                         2n
     * 2x axpy                          6n
     * 1x norm2 residual                 n
     */
    while (true) {
        ++total_iter;
        this->template log<log::Logger::iteration_complete>(
            this, total_iter, residual.get(), dense_x);

        if (stop_criterion->update()
                .num_iterations(total_iter)
                .residual(residual.get())
                .residual_norm(residual_norm.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        // f = P^H * residual
        subspace_vectors->apply(residual.get(), f.get());

        for (size_type k = 0; k < subspace_dim; k++) {
            // c = M \ f = (c_1, ..., c_s)^T
            // v = residual - sum i=[k,s) of (c_i * g_i)
            exec->run(idr::make_step_1(
                nrhs, k, detail::get_local(m.get()), detail::get_local(f.get()),
                detail::get_local(residual.get()), detail::get_local(g.get()),
                detail::get_local(c.get()), detail::get_local(v.get()),
                &stop_status));

            this->get_preconditioner()->apply(v.get(), helper.get());

            // u_k = omega * precond_vector + sum i=[k,s) of (c_i * u_i)
            exec->run(idr::make_step_2(
                nrhs, k, detail::get_local(omega.get()),
                detail::get_local(helper.get()), detail::get_local(c.get()),
                detail::get_local(u.get()), &stop_status));

            auto u_k = u->create_submatrix(span{0, problem_size},
                                           span{k * nrhs, (k + 1) * nrhs});

            // g_k = Au_k
            this->get_system_matrix()->apply(u_k.get(), helper.get());

            // for i = [0,k)
            //     alpha = p^H_i * g_k / m_i,i
            //     g_k -= alpha * g_i
            //     u_k -= alpha * u_i
            // end for
            // store g_k to g
            // for i = [k,s)
            //     m_i,k = p^H_i * g_k
            // end for
            // beta = f_k / m_k,k
            // residual -= beta * g_k
            // dense_x += beta * u_k
            // f = (0,...,0,f_k+1 - beta * m_k+1,k,...,f_s-1 - beta * m_s-1,k)
            exec->run(idr::make_step_3(
                nrhs, k, detail::get_local(subspace_vectors.get()),
                detail::get_local(g.get()), detail::get_local(helper.get()),
                detail::get_local(u.get()), detail::get_local(m.get()),
                detail::get_local(f.get()), detail::get_local(alpha.get()),
                detail::get_local(residual.get()), detail::get_local(dense_x),
                &stop_status));
        }

        this->get_preconditioner()->apply(residual.get(), helper.get());
        this->get_system_matrix()->apply(helper.get(), t.get());

        t->compute_conj_dot(residual.get(), omega.get(), reduction_tmp);
        t->compute_conj_dot(t.get(), tht.get(), reduction_tmp);
        residual->compute_norm2(residual_norm.get(), reduction_tmp);

        // omega = (t^H * residual) / (t^H * t)
        // rho = (t^H * residual) / (norm(t) * norm(residual))
        // if abs(rho) < kappa then
        //     omega *= kappa / abs(rho)
        // end if
        // residual -= omega * t
        // dense_x += omega * v
        exec->run(idr::make_compute_omega(
            nrhs, kappa, detail::get_local(tht.get()),
            detail::get_local(residual_norm.get()),
            detail::get_local(omega.get()), &stop_status));

        t->scale(subspace_neg_one_op.get());
        residual->add_scaled(omega.get(), t.get());
        dense_x->add_scaled(omega.get(), helper.get());
    }
}


template <typename ValueType>
void Idr<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            // If ValueType is complex, the subspace matrix P will be complex
            // anyway.
            if (!is_complex<ValueType>() && this->get_complex_subspace()) {
                auto complex_b = dense_b->make_complex();
                auto complex_x = dense_x->make_complex();
                this->iterate(complex_b.get(), complex_x.get());
                complex_x->get_real(
                    dynamic_cast<matrix::Dense<remove_complex<ValueType>>*>(
                        dense_x));
            } else {
                this->iterate(dense_b, dense_x);
            }
        },
        b, x);
}


template <typename ValueType>
void Idr<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                const LinOp* beta, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_IDR(_type) class Idr<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR);


}  // namespace solver
}  // namespace gko
