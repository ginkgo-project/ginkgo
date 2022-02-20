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
    precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType>
void Ir<ValueType>::apply_impl(const LinOp* b, LinOp* x,
                               const OverlapMask& wmask) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [this, wmask](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x, wmask);
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

    auto residual = detail::create_with_same_size(dense_b);
    auto inner_solution = detail::create_with_same_size(dense_b);

    bool one_changed{};
    Array<stopping_status> stop_status(exec, dense_b->get_size()[1]);
    exec->run(ir::make_initialize(&stop_status));

    residual->copy_from(dense_b);
    system_matrix_->apply(lend(neg_one_op), dense_x, lend(one_op),
                          lend(residual));

    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_,
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
            system_matrix_->apply(lend(neg_one_op), dense_x, lend(one_op),
                                  lend(residual));
        } else {
            // x = x + relaxation_factor * A \ residual
            solver_->apply(lend(relaxation_factor_), lend(residual),
                           lend(one_op), dense_x);

            // residual = b - A * x
            residual->copy_from(dense_b);
            system_matrix_->apply(lend(neg_one_op), dense_x, lend(one_op),
                                  lend(residual));
        }
    }
}


template <typename ValueType>
template <typename VectorType>
void Ir<ValueType>::apply_dense_impl(const VectorType* dense_b,
                                     VectorType* dense_x,
                                     const OverlapMask& wmask) const
{
    using LocalVector = matrix::Dense<ValueType>;
    constexpr uint8 relative_stopping_id{1};

    auto exec = this->get_executor();
    auto one_op = initialize<LocalVector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<LocalVector>({-one<ValueType>()}, exec);

    // FIXME - Performance
    auto x_clone = as<VectorType>(dense_x->clone());
    auto residual = detail::create_with_same_size(dense_b);
    auto inner_solution = detail::create_with_same_size(dense_b);

    bool one_changed{};
    Array<stopping_status> stop_status(exec, dense_b->get_size()[1]);
    exec->run(ir::make_initialize(&stop_status));

    residual->copy_from(dense_b);
    system_matrix_->apply(lend(neg_one_op), x_clone.get(), lend(one_op),
                          lend(residual));

    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_,
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}),
        x_clone.get(), lend(residual));

    int iter = -1;
    while (true) {
        ++iter;
        this->template log<log::Logger::iteration_complete>(
            this, iter, lend(residual), x_clone.get());

        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(lend(residual))
                .solution(x_clone.get())
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
            x_clone->add_scaled(lend(relaxation_factor_), lend(inner_solution));

            // residual = b - A * x
            residual->copy_from(dense_b);
            system_matrix_->apply(lend(neg_one_op), x_clone.get(), lend(one_op),
                                  lend(residual));
        } else {
            // x = x + relaxation_factor * A \ residual
            solver_->apply(lend(relaxation_factor_), lend(residual),
                           lend(one_op), x_clone.get());

            // residual = b - A * x
            residual->copy_from(dense_b);
            system_matrix_->apply(lend(neg_one_op), x_clone.get(), lend(one_op),
                                  lend(residual));
        }
    }
    // FIXME
    auto x_view = dense_x->create_submatrix(
        wmask.write_idxs, gko::span(0, dense_x->get_size()[1]));
    auto xclone_view = x_clone->create_submatrix(
        wmask.write_idxs, gko::span(0, dense_x->get_size()[1]));
    x_view->copy_from(xclone_view.get());
}


template <typename ValueType>
void Ir<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
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


template <typename ValueType>
void Ir<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                               const LinOp* beta, LinOp* x,
                               const OverlapMask& wmask) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [this, wmask](auto dense_alpha, auto dense_b, auto dense_beta,
                      auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get(), wmask);
            auto x_view = dense_x->create_submatrix(
                wmask.write_idxs, span(0, dense_x->get_size()[1]));
            auto xclone_view = dense_x->create_submatrix(
                wmask.write_idxs, span(0, x_clone->get_size()[1]));
            x_view->scale(dense_beta);
            x_view->add_scaled(dense_alpha, xclone_view.get());
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_IR(_type) class Ir<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IR);


}  // namespace solver
}  // namespace gko
