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

#include <ginkgo/core/solver/ir.hpp>


#include <ginkgo/core/matrix/dense.hpp>


#include "core/solver/ir_kernels.hpp"


namespace gko {
namespace solver {
namespace ir {


GKO_REGISTER_OPERATION(initialize, ir::initialize);


}  // namespace ir


template <typename ValueType>
std::unique_ptr<LinOp> Ir<ValueType>::transpose() const
{
    return build()
        .with_generated_solver(
            share(as<Transposable>(this->get_solver())->transpose()))
        .with_criteria(this->stop_criterion_factory_)
        .on(this->get_executor())
        ->generate(
            share(as<Transposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<LinOp> Ir<ValueType>::conj_transpose() const
{
    return build()
        .with_generated_solver(
            share(as<Transposable>(this->get_solver())->conj_transpose()))
        .with_criteria(this->stop_criterion_factory_)
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType>
void Ir<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using Vector = matrix::Dense<ValueType>;
    constexpr uint8 relative_stopping_id{1};

    auto exec = this->get_executor();
    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);
    auto residual = Vector::create_with_config_of(dense_b);
    auto inner_solution = Vector::create_with_config_of(dense_b);

    bool one_changed{};
    Array<stopping_status> stop_status(exec, dense_b->get_size()[1]);
    exec->run(ir::make_initialize(&stop_status));

    residual->copy_from(dense_b);
    system_matrix_->apply(lend(neg_one_op), dense_x, lend(one_op),
                          lend(residual));

    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *) {}),
        x, lend(residual));

    int iter = -1;
    while (true) {
        ++iter;
        this->template log<log::Logger::iteration_complete>(
            this, iter, lend(residual), dense_x);

        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(lend(residual))
                .solution(dense_x)
                .check(relative_stopping_id, true, &stop_status,
                       &one_changed)) {
            break;
        }

        if (solver_->apply_uses_initial_guess()) {
            // Use the inner solver to solve
            // A * inner_solution = residual
            // with residual as initial guess.
            inner_solution->copy_from(lend(residual));
            solver_->apply(lend(residual), lend(inner_solution));

            // x = x + inner_solution
            dense_x->add_scaled(lend(one_op), lend(inner_solution));

            // residual = b - A * x
            residual->copy_from(dense_b);
            system_matrix_->apply(lend(neg_one_op), dense_x, lend(one_op),
                                  lend(residual));
        } else {
            // x = x + A \ residual
            solver_->apply(lend(one_op), lend(residual), lend(one_op), dense_x);

            // residual = b - A * x
            residual->copy_from(dense_b);
            system_matrix_->apply(lend(neg_one_op), dense_x, lend(one_op),
                                  lend(residual));
        }
    }
}


template <typename ValueType>
void Ir<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                               const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_IR(_type) class Ir<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IR);


}  // namespace solver
}  // namespace gko
