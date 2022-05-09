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

#include <ginkgo/core/solver/bicgstab.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>


#include "core/solver/bicgstab_kernels.hpp"


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
        .with_criteria(this->get_stop_criterion_factory())
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
        .with_criteria(this->get_stop_criterion_factory())
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType>
void Bicgstab<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType>
void Bicgstab<ValueType>::apply_dense_impl(
    const matrix::Dense<ValueType>* dense_b,
    matrix::Dense<ValueType>* dense_x) const
{
    using std::swap;
    using Vector = matrix::Dense<ValueType>;
    using AbsVector = matrix::Dense<remove_complex<ValueType>>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();

    auto r = this->create_workspace_with_config_of(0, dense_b);
    auto z = this->create_workspace_with_config_of(1, dense_b);
    auto y = this->create_workspace_with_config_of(2, dense_b);
    auto v = this->create_workspace_with_config_of(3, dense_b);
    auto s = this->create_workspace_with_config_of(4, dense_b);
    auto t = this->create_workspace_with_config_of(5, dense_b);
    auto p = this->create_workspace_with_config_of(6, dense_b);
    auto rr = this->create_workspace_with_config_of(7, dense_b);

    auto alpha = this->template create_workspace_scalar<ValueType>(
        8, dense_b->get_size()[1]);
    auto beta = this->template create_workspace_scalar<ValueType>(
        9, dense_b->get_size()[1]);
    auto gamma = this->template create_workspace_scalar<ValueType>(
        10, dense_b->get_size()[1]);
    auto prev_rho = this->template create_workspace_scalar<ValueType>(
        11, dense_b->get_size()[1]);
    auto rho = this->template create_workspace_scalar<ValueType>(
        12, dense_b->get_size()[1]);
    auto omega = this->template create_workspace_scalar<ValueType>(
        13, dense_b->get_size()[1]);

    auto one_op = this->template create_workspace_scalar<ValueType>(14, 1);
    auto neg_one_op = this->template create_workspace_scalar<ValueType>(15, 1);
    one_op->fill(one<ValueType>());
    neg_one_op->fill(-one<ValueType>());

    bool one_changed{};
    auto& stop_status = this->template create_workspace_array<stopping_status>(
        0, dense_b->get_size()[1]);
    auto& reduction_tmp = this->template create_workspace_array<char>(1, 0);

    // TODO: replace this with automatic merged kernel generator
    exec->run(bicgstab::make_initialize(dense_b, r, rr, y, s, t, z, v, p,
                                        prev_rho, rho, alpha, beta, gamma,
                                        omega, &stop_status));
    // r = dense_b
    // prev_rho = rho = omega = alpha = beta = gamma = 1.0
    // rr = v = s = t = z = y = p = 0
    // stop_status = 0x00

    this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, r);
    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x, r);
    rr->copy_from(r);

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
            this, iter, r, dense_x, nullptr, rho);
        rr->compute_conj_dot(r, rho, reduction_tmp);

        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r)
                .implicit_sq_residual_norm(rho)
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        // tmp = rho / prev_rho * alpha / omega
        // p = r + tmp * (p - omega * v)
        exec->run(bicgstab::make_step_1(r, p, v, rho, prev_rho, alpha, omega,
                                        &stop_status));

        this->get_preconditioner()->apply(p, y);
        this->get_system_matrix()->apply(y, v);
        rr->compute_conj_dot(v, beta, reduction_tmp);
        // alpha = rho / beta
        // s = r - alpha * v
        exec->run(
            bicgstab::make_step_2(r, s, v, rho, alpha, beta, &stop_status));

        auto all_converged =
            stop_criterion->update()
                .num_iterations(iter)
                .residual(s)
                .implicit_sq_residual_norm(rho)
                // .solution(dense_x) // outdated at this point
                .check(RelativeStoppingId, false, &stop_status, &one_changed);
        if (one_changed) {
            exec->run(bicgstab::make_finalize(dense_x, y, alpha, &stop_status));
        }
        if (all_converged) {
            break;
        }

        this->get_preconditioner()->apply(s, z);
        this->get_system_matrix()->apply(z, t);
        s->compute_conj_dot(t, gamma, reduction_tmp);
        t->compute_conj_dot(t, beta, reduction_tmp);
        // omega = gamma / beta
        // x = x + alpha * y + omega * z
        // r = s - omega * t
        exec->run(bicgstab::make_step_3(dense_x, r, s, t, y, z, alpha, beta,
                                        gamma, omega, &stop_status));
        swap(prev_rho, rho);
    }
}


template <typename ValueType>
void Bicgstab<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                     const LinOp* beta, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
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
