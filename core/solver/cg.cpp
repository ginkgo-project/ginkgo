/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

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

#include <ginkgo/core/solver/cg.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/utils.hpp>
#include "core/base/exception_helpers.hpp"
#include "core/solver/cg_kernels.hpp"


namespace gko {
namespace solver {


namespace {


template <typename ValueType>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(initialize, cg::initialize<ValueType>);
    GKO_REGISTER_OPERATION(step_1, cg::step_1<ValueType>);
    GKO_REGISTER_OPERATION(step_2, cg::step_2<ValueType>);
};


}  // namespace


template <typename ValueType>
void Cg<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using std::swap;
    using Vector = matrix::Dense<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);
    auto r = Vector::create_with_config_of(dense_b);
    auto z = Vector::create_with_config_of(dense_b);
    auto p = Vector::create_with_config_of(dense_b);
    auto q = Vector::create_with_config_of(dense_b);

    auto alpha = Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    auto beta = Vector::create_with_config_of(alpha.get());
    auto prev_rho = Vector::create_with_config_of(alpha.get());
    auto rho = Vector::create_with_config_of(alpha.get());

    bool one_changed{};
    Array<stopping_status> stop_status(alpha->get_executor(),
                                       dense_b->get_size()[1]);

    // TODO: replace this with automatic merged kernel generator
    exec->run(TemplatedOperation<ValueType>::make_initialize_operation(
        dense_b, r.get(), z.get(), p.get(), q.get(), prev_rho.get(), rho.get(),
        &stop_status));
    // r = dense_b
    // rho = 0.0
    // prev_rho = 1.0
    // z = p = q = 0

    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(), r.get());
    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *) {}),
        x, r.get());

    int iter = 0;
    while (true) {
        preconditioner_->apply(r.get(), z.get());
        r->compute_dot(z.get(), rho.get());

        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        exec->run(TemplatedOperation<ValueType>::make_step_1_operation(
            p.get(), z.get(), rho.get(), prev_rho.get(), &stop_status));
        // tmp = rho / prev_rho
        // p = z + tmp * p
        system_matrix_->apply(p.get(), q.get());
        p->compute_dot(q.get(), beta.get());
        exec->run(TemplatedOperation<ValueType>::make_step_2_operation(
            dense_x, r.get(), p.get(), q.get(), beta.get(), rho.get(),
            &stop_status));
        // tmp = rho / beta
        // x = x + tmp * p
        // r = r - tmp * q
        swap(prev_rho, rho);
        this->template log<log::Logger::iteration_complete>(this, iter + 1,
                                                            r.get(), dense_x);
        iter++;
    }
}


template <typename ValueType>
void Cg<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                               const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_CG(_type) class Cg<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG);
#undef GKO_DECLARE_CG


}  // namespace solver
}  // namespace gko
