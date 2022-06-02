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

#include <ginkgo/core/solver/ir.hpp>


#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/distributed/helpers.hpp"
#include "core/solver/ir_kernels.hpp"


namespace gko {
namespace solver {
namespace ir {
namespace {


GKO_REGISTER_OPERATION(initialize, ir::initialize);


}  // anonymous namespace
}  // namespace ir


template <typename ValueType>
void Ir<ValueType>::set_solver(std::shared_ptr<const LinOp> new_solver)
{
    auto exec = this->get_executor();
    if (new_solver) {
        GKO_ASSERT_EQUAL_DIMENSIONS(new_solver, this);
        GKO_ASSERT_IS_SQUARE_MATRIX(new_solver);
        if (new_solver->get_executor() != exec) {
            new_solver = gko::clone(exec, new_solver);
        }
    }
    solver_ = new_solver;
}


template <typename ValueType>
void Ir<ValueType>::set_relaxation_factor(
    std::shared_ptr<const matrix::Dense<ValueType>> new_factor)
{
    auto exec = this->get_executor();
    if (new_factor && new_factor->get_executor() != exec) {
        new_factor = gko::clone(exec, new_factor);
    }
    relaxation_factor_ = new_factor;
}


template <typename ValueType>
Ir<ValueType>& Ir<ValueType>::operator=(const Ir& other)
{
    if (&other != this) {
        EnableLinOp<Ir>::operator=(other);
        EnableSolverBase<Ir>::operator=(other);
        EnableIterativeBase<Ir>::operator=(other);
        this->set_solver(other.get_solver());
        this->set_relaxation_factor(other.relaxation_factor_);
        parameters_ = other.parameters_;
    }
    return *this;
}


template <typename ValueType>
Ir<ValueType>& Ir<ValueType>::operator=(Ir&& other)
{
    if (&other != this) {
        EnableLinOp<Ir>::operator=(std::move(other));
        EnableSolverBase<Ir>::operator=(std::move(other));
        EnableIterativeBase<Ir>::operator=(std::move(other));
        this->set_solver(other.get_solver());
        this->set_relaxation_factor(other.relaxation_factor_);
        other.set_solver(nullptr);
        other.set_relaxation_factor(nullptr);
        parameters_ = other.parameters_;
    }
    return *this;
}


template <typename ValueType>
Ir<ValueType>::Ir(const Ir& other) : Ir(other.get_executor())
{
    *this = other;
}


template <typename ValueType>
Ir<ValueType>::Ir(Ir&& other) : Ir(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType>
std::unique_ptr<LinOp> Ir<ValueType>::transpose() const
{
    return build()
        .with_generated_solver(
            share(as<Transposable>(this->get_solver())->transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .with_relaxation_factor(parameters_.relaxation_factor)
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
        .with_criteria(this->get_stop_criterion_factory())
        .with_relaxation_factor(conj(parameters_.relaxation_factor))
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType>
void Ir<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
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
void Ir<ValueType>::apply_dense_impl(const VectorType* dense_b,
                                     VectorType* dense_x) const
{
    using LocalVector = matrix::Dense<ValueType>;
    constexpr uint8 relative_stopping_id{1};

    auto exec = this->get_executor();
    auto one_op = initialize<LocalVector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<LocalVector>({-one<ValueType>()}, exec);

    auto residual = detail::create_with_config_of(dense_b);
    auto inner_solution = detail::create_with_config_of(dense_b);

    bool one_changed{};
    array<stopping_status> stop_status(exec, dense_b->get_size()[1]);
    exec->run(ir::make_initialize(&stop_status));

    residual->copy_from(dense_b);
    this->get_system_matrix()->apply(lend(neg_one_op), dense_x, lend(one_op),
                                     lend(residual));

    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
        lend(residual));

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

            // x = x + relaxation_factor * inner_solution
            dense_x->add_scaled(lend(relaxation_factor_), lend(inner_solution));

            // residual = b - A * x
            residual->copy_from(dense_b);
            this->get_system_matrix()->apply(lend(neg_one_op), dense_x,
                                             lend(one_op), lend(residual));
        } else {
            // x = x + relaxation_factor * A \ residual
            solver_->apply(lend(relaxation_factor_), lend(residual),
                           lend(one_op), dense_x);

            // residual = b - A * x
            residual->copy_from(dense_b);
            this->get_system_matrix()->apply(lend(neg_one_op), dense_x,
                                             lend(one_op), lend(residual));
        }
    }
}


template <typename ValueType>
void Ir<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
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


#define GKO_DECLARE_IR(_type) class Ir<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IR);


}  // namespace solver
}  // namespace gko
