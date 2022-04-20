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


#include "core/solver/ir_kernels.hpp"


namespace gko {
namespace solver {
namespace ir {
namespace {


GKO_REGISTER_OPERATION(initialize, ir::initialize);


}  // anonymous namespace
}  // namespace ir


template <typename ValueType>
std::unique_ptr<LinOp> Ir<ValueType>::transpose() const
{
    return build()
        .with_generated_solver(
            share(as<Transposable>(this->get_solver())->transpose()))
        .with_criteria(this->stop_criterion_factory_)
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
        .with_criteria(this->stop_criterion_factory_)
        .with_relaxation_factor(conj(parameters_.relaxation_factor))
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType>
void Ir<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType>
void Ir<ValueType>::apply_dense_impl(const matrix::Dense<ValueType>* dense_b,
                                     matrix::Dense<ValueType>* dense_x) const
{
    using Vector = matrix::Dense<ValueType>;
    constexpr uint8 relative_stopping_id{1};
    bool zero_input = this->get_input_zero();

    auto exec = this->get_executor();

    // TODO: add tempory output clone
    auto residual_cache =
        std::dynamic_pointer_cast<Vector>(this->get_residual_cache());
    Vector* residual = nullptr;
    if (residual_cache) {
        residual = residual_cache.get();
    }
    if (!residual) {
        if (!residual_op_.get() ||
            residual_op_->get_size() != dense_b->get_size() ||
            residual_op_->get_executor() != dense_b->get_executor()) {
            residual_op_ = Vector::create_with_config_of(dense_b);
        }
        residual = residual_op_.get();
    }
    if (!inner_solution_.get() ||
        inner_solution_->get_size() != dense_b->get_size()) {
        inner_solution_ = Vector::create_with_config_of(dense_b);
    }
    bool one_changed{};
    stop_status_.resize_and_reset(dense_b->get_size()[1]);
    exec->run(ir::make_initialize(&stop_status_));
    if (!zero_input) {
        residual->copy_from(dense_b);
        system_matrix_->apply(lend(neg_one_op_), dense_x, lend(one_op_),
                              lend(residual));
    }
    const Vector* residual_ptr = zero_input ? dense_b : residual;
    // zero input the residual is dense_b

    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_,
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
        residual_ptr);

    int iter = -1;
    while (true) {
        ++iter;
        this->template log<log::Logger::iteration_complete>(
            this, iter, residual_ptr, dense_x);

        if (iter == 0) {
            // It already prepared the residual for the first iteration
            if (stop_criterion->update()
                    .num_iterations(iter)
                    .residual(residual_ptr)
                    .solution(dense_x)
                    .check(relative_stopping_id, true, &stop_status_,
                           &one_changed)) {
                break;
            }
        } else {
            // We check the iteration criterion first.
            if (stop_criterion->update()
                    .num_iterations(iter)
                    .solution(dense_x)
                    .check(relative_stopping_id, false, &stop_status_,
                           &one_changed)) {
                break;
            }
            // If it is not terminated due to iteration, prepare the residual
            // for check or the further running.
            residual_ptr = residual;
            // residual = b - A * x
            residual->copy_from(dense_b);
            system_matrix_->apply(lend(neg_one_op_), dense_x, lend(one_op_),
                                  lend(residual));
            if (stop_criterion->update()
                    .num_iterations(iter)
                    .residual(residual_ptr)
                    .solution(dense_x)
                    .check(relative_stopping_id, true, &stop_status_,
                           &one_changed)) {
                break;
            }
        }

        if (solver_->apply_uses_initial_guess()) {
            // Use the inner solver to solve
            // A * inner_solution = residual
            // with residual as initial guess.
            inner_solution_->copy_from(residual_ptr);
            solver_->apply(residual_ptr, lend(inner_solution_));

            // x = x + relaxation_factor * inner_solution
            dense_x->add_scaled(lend(relaxation_factor_),
                                lend(inner_solution_));
        } else {
            // x = x + relaxation_factor * A \ residual
            solver_->apply(lend(relaxation_factor_), residual_ptr,
                           lend(one_op_), dense_x);
        }
    }
}


template <typename ValueType>
void Ir<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                               const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
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
