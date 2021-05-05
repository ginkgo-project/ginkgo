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

#include <ginkgo/core/solver/bicgstab.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>


#include "core/solver/bicgstab_kernels.hpp"
#include "core/solver/distributed_helpers.hpp"


namespace gko {
namespace solver {
namespace bicgstab {
namespace {


GKO_REGISTER_OPERATION(initialize, bicgstab::initialize);
GKO_REGISTER_OPERATION(step_1, bicgstab::step_1);
GKO_REGISTER_OPERATION(step_2, bicgstab::step_2);
GKO_REGISTER_OPERATION(step_3, bicgstab::step_3);
GKO_REGISTER_OPERATION(finalize, bicgstab::finalize);


}  // anonymous namespace
}  // namespace bicgstab


template <typename ValueType>
std::unique_ptr<LinOp> Bicgstab<ValueType>::transpose() const
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
std::unique_ptr<LinOp> Bicgstab<ValueType>::conj_transpose() const
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
void Bicgstab<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType>
template <typename VectorType>
void Bicgstab<ValueType>::apply_dense_impl(const VectorType *dense_b,
                                           VectorType *dense_x) const
{
    using std::swap;
    using LocalVector = matrix::Dense<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();

    auto one_op = initialize<LocalVector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<LocalVector>({-one<ValueType>()}, exec);

    auto r = detail::create_with_same_size(dense_b);
    auto z = detail::create_with_same_size(dense_b);
    auto y = detail::create_with_same_size(dense_b);
    auto v = detail::create_with_same_size(dense_b);
    auto s = detail::create_with_same_size(dense_b);
    auto t = detail::create_with_same_size(dense_b);
    auto p = detail::create_with_same_size(dense_b);
    auto rr = detail::create_with_same_size(dense_b);

    auto alpha = LocalVector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    auto beta = LocalVector::create_with_config_of(alpha.get());
    auto gamma = LocalVector::create_with_config_of(alpha.get());
    auto prev_rho = LocalVector::create_with_config_of(alpha.get());
    auto rho = LocalVector::create_with_config_of(alpha.get());
    auto omega = LocalVector::create_with_config_of(alpha.get());

    bool one_changed{};
    Array<stopping_status> stop_status(alpha->get_executor(),
                                       dense_b->get_size()[1]);

    // TODO: replace this with automatic merged kernel generator
    exec->run(bicgstab::make_initialize(
        detail::get_local(dense_b), detail::get_local(r.get()),
        detail::get_local(rr.get()), detail::get_local(y.get()),
        detail::get_local(s.get()), detail::get_local(t.get()),
        detail::get_local(z.get()), detail::get_local(v.get()),
        detail::get_local(p.get()), prev_rho.get(), rho.get(), alpha.get(),
        beta.get(), gamma.get(), omega.get(), &stop_status));
    // r = dense_b
    // prev_rho = rho = omega = alpha = beta = gamma = 1.0
    // rr = v = s = t = z = y = p = 0
    // stop_status = 0x00

    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(), r.get());
    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_,
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
        r.get());
    rr->copy_from(r.get());

    int iter = -1;

    /* Memory movement summary:
     * 31n * values + 2 * matrix/preconditioner storage
     * 2x SpMV:                4n * values + 2 * storage
     * 2x Preconditioner:      4n * values + 2 * storage
     * 3x dot                  6n
     * 1x norm2                 n
     * 1x step 1 (fused axpys) 4n
     * 1x step 2 (axpy)        3n
     * 1x step 3 (fused axpys) 7n
     * 2x norm2 residual       2n
     */
    while (true) {
        ++iter;
        this->template log<log::Logger::iteration_complete>(
            this, iter, r.get(), dense_x, nullptr, rho.get());
        rr->compute_conj_dot(r.get(), rho.get());

        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r.get())
                .implicit_sq_residual_norm(rho.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        // tmp = rho / prev_rho * alpha / omega
        // p = r + tmp * (p - omega * v)
        exec->run(bicgstab::make_step_1(
            detail::get_local(r.get()), detail::get_local(p.get()),
            detail::get_local(v.get()), rho.get(), prev_rho.get(), alpha.get(),
            omega.get(), &stop_status));

        get_preconditioner()->apply(p.get(), y.get());
        system_matrix_->apply(y.get(), v.get());
        rr->compute_conj_dot(v.get(), beta.get());
        // alpha = rho / beta
        // s = r - alpha * v
        exec->run(bicgstab::make_step_2(detail::get_local(r.get()),
                                        detail::get_local(s.get()),
                                        detail::get_local(v.get()), rho.get(),
                                        alpha.get(), beta.get(), &stop_status));

        auto all_converged =
            stop_criterion->update()
                .num_iterations(iter)
                .residual(s.get())
                .implicit_sq_residual_norm(rho.get())
                // .solution(dense_x) // outdated at this point
                .check(RelativeStoppingId, false, &stop_status, &one_changed);
        if (one_changed) {
            exec->run(bicgstab::make_finalize(detail::get_local(dense_x),
                                              detail::get_local(y.get()),
                                              alpha.get(), &stop_status));
        }
        if (all_converged) {
            break;
        }

        get_preconditioner()->apply(s.get(), z.get());
        system_matrix_->apply(z.get(), t.get());
        s->compute_conj_dot(t.get(), gamma.get());
        t->compute_conj_dot(t.get(), beta.get());
        // omega = gamma / beta
        // x = x + alpha * y + omega * z
        // r = s - omega * t
        exec->run(bicgstab::make_step_3(
            detail::get_local(dense_x), detail::get_local(r.get()),
            detail::get_local(s.get()), detail::get_local(t.get()),
            detail::get_local(y.get()), detail::get_local(z.get()), alpha.get(),
            beta.get(), gamma.get(), omega.get(), &stop_status));
        swap(prev_rho, rho);
    }
}


template <typename ValueType>
void Bicgstab<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                     const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_BICGSTAB(_type) class Bicgstab<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB);


}  // namespace solver
}  // namespace gko
