/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

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

#include <ginkgo/core/solver/bicgstab.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>


#include "core/solver/bicgstab_kernels.hpp"


namespace gko {
namespace solver {
namespace bicgstab {


GKO_REGISTER_OPERATION(initialize, bicgstab::initialize);
GKO_REGISTER_OPERATION(step_1, bicgstab::step_1);
GKO_REGISTER_OPERATION(step_2, bicgstab::step_2);
GKO_REGISTER_OPERATION(step_3, bicgstab::step_3);
GKO_REGISTER_OPERATION(finalize, bicgstab::finalize);


}  // namespace bicgstab


template <typename ValueType>
void Bicgstab<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using std::swap;
    using Vector = matrix::Dense<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto dense_b = as<Vector>(b);
    auto dense_x = as<Vector>(x);
    auto r = Vector::create_with_config_of(dense_b);
    auto z = Vector::create_with_config_of(dense_b);
    auto y = Vector::create_with_config_of(dense_b);
    auto v = Vector::create_with_config_of(dense_b);
    auto s = Vector::create_with_config_of(dense_b);
    auto t = Vector::create_with_config_of(dense_b);
    auto p = Vector::create_with_config_of(dense_b);
    auto rr = Vector::create_with_config_of(dense_b);

    auto alpha = Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    auto beta = Vector::create_with_config_of(alpha.get());
    auto gamma = Vector::create_with_config_of(alpha.get());
    auto prev_rho = Vector::create_with_config_of(alpha.get());
    auto rho = Vector::create_with_config_of(alpha.get());
    auto omega = Vector::create_with_config_of(alpha.get());

    bool one_changed{};
    Array<stopping_status> stop_status(alpha->get_executor(),
                                       dense_b->get_size()[1]);

    // TODO: replace this with automatic merged kernel generator
    exec->run(bicgstab::make_initialize(
        dense_b, r.get(), rr.get(), y.get(), s.get(), t.get(), z.get(), v.get(),
        p.get(), prev_rho.get(), rho.get(), alpha.get(), beta.get(),
        gamma.get(), omega.get(), &stop_status));
    // r = dense_b
    // prev_rho = rho = omega = alpha = beta = gamma = 1.0
    // rr = v = s = t = z = y = p = 0
    // stop_status = 0x00

    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(), r.get());
    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *) {}),
        x, r.get());
    rr->copy_from(r.get());
    system_matrix_->apply(r.get(), v.get());

    int iter = -1;
    while (true) {
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

        rr->compute_dot(r.get(), rho.get());

        exec->run(bicgstab::make_step_1(r.get(), p.get(), v.get(), rho.get(),
                                        prev_rho.get(), alpha.get(),
                                        omega.get(), &stop_status));
        // tmp = rho / prev_rho * alpha / omega
        // p = r + tmp * (p - omega * v)

        preconditioner_->apply(p.get(), y.get());
        system_matrix_->apply(y.get(), v.get());
        rr->compute_dot(v.get(), beta.get());
        exec->run(bicgstab::make_step_2(r.get(), s.get(), v.get(), rho.get(),
                                        alpha.get(), beta.get(), &stop_status));
        // alpha = rho / beta
        // s = r - alpha * v

        ++iter;
        auto all_converged =
            stop_criterion->update()
                .num_iterations(iter)
                .residual(s.get())
                // .solution(dense_x) // outdated at this point
                .check(RelativeStoppingId, false, &stop_status, &one_changed);
        if (one_changed) {
            exec->run(bicgstab::make_finalize(dense_x, y.get(), alpha.get(),
                                              &stop_status));
        }
        this->template log<log::Logger::iteration_complete>(this, iter,
                                                            r.get());
        if (all_converged) {
            break;
        }

        preconditioner_->apply(s.get(), z.get());
        system_matrix_->apply(z.get(), t.get());
        s->compute_dot(t.get(), gamma.get());
        t->compute_dot(t.get(), beta.get());
        exec->run(bicgstab::make_step_3(
            dense_x, r.get(), s.get(), t.get(), y.get(), z.get(), alpha.get(),
            beta.get(), gamma.get(), omega.get(), &stop_status));
        // omega = gamma / beta
        // x = x + alpha * y + omega * z
        // r = s - omega * t
        swap(prev_rho, rho);
    }
}  // namespace solver


template <typename ValueType>
void Bicgstab<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                     const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);
    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_BICGSTAB(_type) class Bicgstab<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB);


}  // namespace solver
}  // namespace gko
