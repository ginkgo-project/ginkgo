/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/solver/cgs.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>


#include "core/solver/cgs_kernels.hpp"


namespace gko {
namespace solver {


namespace cgs {


GKO_REGISTER_OPERATION(initialize, cgs::initialize);
GKO_REGISTER_OPERATION(step_1, cgs::step_1);
GKO_REGISTER_OPERATION(step_2, cgs::step_2);
GKO_REGISTER_OPERATION(step_3, cgs::step_3);


}  // namespace cgs


template <typename ValueType>
std::unique_ptr<LinOp> Cgs<ValueType>::transpose() const
{
    return build()
        .with_generated_preconditioner(
            share(as<Transposable>(this->get_preconditioner())->transpose()))
        .with_criteria(this->stop_criterion_factory_)
        .on(this->get_executor())
        ->generate(
            share(as<Transposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<LinOp> Cgs<ValueType>::conj_transpose() const
{
    return build()
        .with_generated_preconditioner(share(
            as<Transposable>(this->get_preconditioner())->conj_transpose()))
        .with_criteria(this->stop_criterion_factory_)
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType>
void Cgs<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using std::swap;
    using Vector = matrix::Dense<ValueType>;
    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();
    size_type num_vectors = dense_b->get_size()[1];

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto r = Vector::create_with_config_of(dense_b);
    auto r_tld = Vector::create_with_config_of(dense_b);
    auto p = Vector::create_with_config_of(dense_b);
    auto q = Vector::create_with_config_of(dense_b);
    auto u = Vector::create_with_config_of(dense_b);
    auto u_hat = Vector::create_with_config_of(dense_b);
    auto v_hat = Vector::create_with_config_of(dense_b);
    auto t = Vector::create_with_config_of(dense_b);

    auto alpha = Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    auto beta = Vector::create_with_config_of(alpha.get());
    auto gamma = Vector::create_with_config_of(alpha.get());
    auto rho_prev = Vector::create_with_config_of(alpha.get());
    auto rho = Vector::create_with_config_of(alpha.get());

    bool one_changed{};
    Array<stopping_status> stop_status(alpha->get_executor(),
                                       dense_b->get_size()[1]);

    // TODO: replace this with automatic merged kernel generator
    exec->run(cgs::make_initialize(
        dense_b, r.get(), r_tld.get(), p.get(), q.get(), u.get(), u_hat.get(),
        v_hat.get(), t.get(), alpha.get(), beta.get(), gamma.get(),
        rho_prev.get(), rho.get(), &stop_status));
    // r = dense_b
    // r_tld = r
    // rho = 0.0
    // rho_prev = 1.0
    // p = q = u = u_hat = v_hat = t = 0

    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(), r.get());
    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *) {}),
        x, r.get());
    r_tld->copy_from(r.get());

    int iter = 0;
    while (true) {
        r->compute_dot(r_tld.get(), rho.get());
        exec->run(cgs::make_step_1(r.get(), u.get(), p.get(), q.get(),
                                   beta.get(), rho.get(), rho_prev.get(),
                                   &stop_status));
        // beta = rho / rho_prev
        // u = r + beta * q;
        // p = u + beta * ( q + beta * p );
        get_preconditioner()->apply(p.get(), t.get());
        system_matrix_->apply(t.get(), v_hat.get());
        r_tld->compute_dot(v_hat.get(), gamma.get());
        exec->run(cgs::make_step_2(u.get(), v_hat.get(), q.get(), t.get(),
                                   alpha.get(), rho.get(), gamma.get(),
                                   &stop_status));

        ++iter;
        this->template log<log::Logger::iteration_complete>(this, iter, r.get(),
                                                            dense_x);

        // alpha = rho / gamma
        // q = u - alpha * v_hat
        // t = u + q
        get_preconditioner()->apply(t.get(), u_hat.get());
        system_matrix_->apply(u_hat.get(), t.get());
        exec->run(cgs::make_step_3(t.get(), u_hat.get(), r.get(), dense_x,
                                   alpha.get(), &stop_status));
        // r = r -alpha * t
        // x = x + alpha * u_hat

        ++iter;
        this->template log<log::Logger::iteration_complete>(this, iter, r.get(),
                                                            dense_x);
        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        swap(rho_prev, rho);
    }
}


template <typename ValueType>
void Cgs<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_CGS(_type) class Cgs<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS);


}  // namespace solver
}  // namespace gko
