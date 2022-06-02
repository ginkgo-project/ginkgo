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

#include <ginkgo/core/solver/cgs.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>


#include "core/distributed/helpers.hpp"
#include "core/solver/cgs_kernels.hpp"


namespace gko {
namespace solver {
namespace cgs {
namespace {


GKO_REGISTER_OPERATION(initialize, cgs::initialize);
GKO_REGISTER_OPERATION(step_1, cgs::step_1);
GKO_REGISTER_OPERATION(step_2, cgs::step_2);
GKO_REGISTER_OPERATION(step_3, cgs::step_3);


}  // anonymous namespace
}  // namespace cgs


template <typename ValueType>
std::unique_ptr<LinOp> Cgs<ValueType>::transpose() const
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
std::unique_ptr<LinOp> Cgs<ValueType>::conj_transpose() const
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
void Cgs<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType>
template <typename VectorType>
void Cgs<ValueType>::apply_dense_impl(const VectorType* dense_b,
                                      VectorType* dense_x) const
{
    using std::swap;
    using LocalVector = matrix::Dense<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();
    size_type num_vectors = dense_b->get_size()[1];

    array<char> reduction_tmp{exec};

    auto one_op = initialize<LocalVector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<LocalVector>({-one<ValueType>()}, exec);

    auto r = detail::create_with_same_size(dense_b);
    auto r_tld = detail::create_with_same_size(dense_b);
    auto p = detail::create_with_same_size(dense_b);
    auto q = detail::create_with_same_size(dense_b);
    auto u = detail::create_with_same_size(dense_b);
    auto u_hat = detail::create_with_same_size(dense_b);
    auto v_hat = detail::create_with_same_size(dense_b);
    auto t = detail::create_with_same_size(dense_b);

    auto alpha = LocalVector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    auto beta = LocalVector::create_with_config_of(alpha.get());
    auto gamma = LocalVector::create_with_config_of(alpha.get());
    auto rho_prev = LocalVector::create_with_config_of(alpha.get());
    auto rho = LocalVector::create_with_config_of(alpha.get());

    bool one_changed{};
    array<stopping_status> stop_status(alpha->get_executor(),
                                       dense_b->get_size()[1]);

    // TODO: replace this with automatic merged kernel generator
    exec->run(cgs::make_initialize(
        detail::get_local(dense_b), detail::get_local(r.get()),
        detail::get_local(r_tld.get()), detail::get_local(p.get()),
        detail::get_local(q.get()), detail::get_local(u.get()),
        detail::get_local(u_hat.get()), detail::get_local(v_hat.get()),
        detail::get_local(t.get()), alpha.get(), beta.get(), gamma.get(),
        rho_prev.get(), rho.get(), &stop_status));
    // r = dense_b
    // r_tld = r
    // rho = 0.0
    // rho_prev = alpha = beta = gamma = 1.0
    // p = q = u = u_hat = v_hat = t = 0

    this->get_system_matrix()->apply(neg_one_op.get(), dense_x, one_op.get(),
                                     r.get());
    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
        r.get());
    r_tld->copy_from(r.get());

    int iter = -1;
    /* Memory movement summary:
     * 28n * values + 2 * matrix/preconditioner storage
     * 2x SpMV:                4n * values + 2 * storage
     * 2x Preconditioner:      4n * values + 2 * storage
     * 2x dot                  4n
     * 1x step 1 (fused axpys) 5n
     * 1x step 2 (fused axpys) 4n
     * 1x step 3 (axpys)       6n
     * 1x norm2 residual        n
     */
    while (true) {
        r->compute_conj_dot(r_tld.get(), rho.get(), reduction_tmp);

        ++iter;
        this->template log<log::Logger::iteration_complete>(
            this, iter, r.get(), dense_x, nullptr, rho.get());
        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r.get())
                .implicit_sq_residual_norm(rho.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        // beta = rho / rho_prev
        // u = r + beta * q
        // p = u + beta * ( q + beta * p )
        exec->run(cgs::make_step_1(
            detail::get_local(r.get()), detail::get_local(u.get()),
            detail::get_local(p.get()), detail::get_local(q.get()), beta.get(),
            rho.get(), rho_prev.get(), &stop_status));
        this->get_preconditioner()->apply(p.get(), t.get());
        this->get_system_matrix()->apply(t.get(), v_hat.get());
        r_tld->compute_conj_dot(v_hat.get(), gamma.get(), reduction_tmp);
        // alpha = rho / gamma
        // q = u - alpha * v_hat
        // t = u + q
        exec->run(cgs::make_step_2(
            detail::get_local(u.get()), detail::get_local(v_hat.get()),
            detail::get_local(q.get()), detail::get_local(t.get()), alpha.get(),
            rho.get(), gamma.get(), &stop_status));

        this->get_preconditioner()->apply(t.get(), u_hat.get());
        this->get_system_matrix()->apply(u_hat.get(), t.get());
        // r = r - alpha * t
        // x = x + alpha * u_hat
        exec->run(cgs::make_step_3(
            detail::get_local(t.get()), detail::get_local(u_hat.get()),
            detail::get_local(r.get()), detail::get_local(dense_x), alpha.get(),
            &stop_status));

        swap(rho_prev, rho);
    }
}


template <typename ValueType>
void Cgs<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                const LinOp* beta, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_CGS(_type) class Cgs<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS);


}  // namespace solver
}  // namespace gko
