// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/ir.hpp>


#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


#include "core/distributed/helpers.hpp"
#include "core/solver/ir_kernels.hpp"
#include "core/solver/solver_base.hpp"
#include "core/solver/solver_boilerplate.hpp"


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
        this->parameters_ = other.parameters_;
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
        this->parameters_ = std::exchange(other.parameters_, parameters_type{});
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
    this->apply_with_initial_guess_impl(b, x,
                                        this->get_default_initial_guess());
}


template <typename ValueType>
void Ir<ValueType>::apply_with_initial_guess_impl(
    const LinOp* b, LinOp* x, initial_guess_mode guess) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this, guess](auto dense_b, auto dense_x) {
            prepare_initial_guess(dense_b, dense_x, guess);
            this->apply_dense_impl(dense_b, dense_x, guess);
        },
        b, x);
}


template <typename ValueType>
template <typename VectorType>
void Ir<ValueType>::apply_dense_impl(const VectorType* dense_b,
                                     VectorType* dense_x,
                                     initial_guess_mode guess) const
{
    using Vector = matrix::Dense<ValueType>;
    using ws = workspace_traits<Ir>;
    constexpr uint8 relative_stopping_id{1};

    auto exec = this->get_executor();
    this->setup_workspace();

    GKO_SOLVER_VECTOR(residual, dense_b);
    GKO_SOLVER_VECTOR(inner_solution, dense_b);

    GKO_SOLVER_ONE_MINUS_ONE();

    bool one_changed{};
    auto& stop_status = this->template create_workspace_array<stopping_status>(
        ws::stop, dense_b->get_size()[1]);
    exec->run(ir::make_initialize(&stop_status));
    if (guess != initial_guess_mode::zero) {
        residual->copy_from(dense_b);
        this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, residual);
    }
    // zero input the residual is dense_b
    const VectorType* residual_ptr =
        guess == initial_guess_mode::zero ? dense_b : residual;

    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
        residual_ptr);

    int iter = -1;
    while (true) {
        ++iter;

        if (iter == 0) {
            // In iter 0, the iteration and residual are updated.
            bool all_stopped = stop_criterion->update()
                                   .num_iterations(iter)
                                   .residual(residual_ptr)
                                   .solution(dense_x)
                                   .check(relative_stopping_id, true,
                                          &stop_status, &one_changed);
            this->template log<log::Logger::iteration_complete>(
                this, dense_b, dense_x, iter, residual_ptr, nullptr, nullptr,
                &stop_status, all_stopped);
            if (all_stopped) {
                break;
            }
        } else {
            // In the other iterations, the residual can be updated separately.
            bool all_stopped = stop_criterion->update()
                                   .num_iterations(iter)
                                   .solution(dense_x)
                                   // we have the residual check later
                                   .ignore_residual_check(true)
                                   .check(relative_stopping_id, false,
                                          &stop_status, &one_changed);
            if (all_stopped) {
                this->template log<log::Logger::iteration_complete>(
                    this, dense_b, dense_x, iter, nullptr, nullptr, nullptr,
                    &stop_status, all_stopped);
                break;
            }
            residual_ptr = residual;
            // residual = b - A * x
            residual->copy_from(dense_b);
            this->get_system_matrix()->apply(neg_one_op, dense_x, one_op,
                                             residual);
            all_stopped = stop_criterion->update()
                              .num_iterations(iter)
                              .residual(residual_ptr)
                              .solution(dense_x)
                              .check(relative_stopping_id, true, &stop_status,
                                     &one_changed);
            this->template log<log::Logger::iteration_complete>(
                this, dense_b, dense_x, iter, residual_ptr, nullptr, nullptr,
                &stop_status, all_stopped);
            if (all_stopped) {
                break;
            }
        }

        if (solver_->apply_uses_initial_guess()) {
            // Use the inner solver to solve
            // A * inner_solution = residual
            // with residual as initial guess.
            inner_solution->copy_from(residual_ptr);
            solver_->apply(residual_ptr, inner_solution);

            // x = x + relaxation_factor * inner_solution
            dense_x->add_scaled(relaxation_factor_, inner_solution);
        } else {
            // x = x + relaxation_factor * A \ residual
            solver_->apply(relaxation_factor_, residual_ptr, one_op, dense_x);
        }
    }
}


template <typename ValueType>
void Ir<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                               const LinOp* beta, LinOp* x) const
{
    this->apply_with_initial_guess_impl(alpha, b, beta, x,
                                        this->get_default_initial_guess());
}

template <typename ValueType>
void Ir<ValueType>::apply_with_initial_guess_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x,
    initial_guess_mode guess) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this, guess](auto dense_alpha, auto dense_b, auto dense_beta,
                      auto dense_x) {
            prepare_initial_guess(dense_b, dense_x, guess);
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get(), guess);
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone);
        },
        alpha, b, beta, x);
}


template <typename ValueType>
int workspace_traits<Ir<ValueType>>::num_arrays(const Solver&)
{
    return 1;
}


template <typename ValueType>
int workspace_traits<Ir<ValueType>>::num_vectors(const Solver&)
{
    return 4;
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Ir<ValueType>>::op_names(
    const Solver&)
{
    return {
        "residual",
        "inner_solution",
        "one",
        "minus_one",
    };
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Ir<ValueType>>::array_names(
    const Solver&)
{
    return {"stop"};
}


template <typename ValueType>
std::vector<int> workspace_traits<Ir<ValueType>>::scalars(const Solver&)
{
    return {};
}


template <typename ValueType>
std::vector<int> workspace_traits<Ir<ValueType>>::vectors(const Solver&)
{
    return {residual, inner_solution};
}


#define GKO_DECLARE_IR(_type) class Ir<_type>
#define GKO_DECLARE_IR_TRAITS(_type) struct workspace_traits<Ir<_type>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IR);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IR_TRAITS);


}  // namespace solver
}  // namespace gko
