/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/solver/chebyshev.hpp>


#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


#include "core/distributed/helpers.hpp"
#include "core/solver/ir_kernels.hpp"
#include "core/solver/solver_base.hpp"
#include "core/solver/solver_boilerplate.hpp"
#include "core/solver/update_residual.hpp"


namespace gko {
namespace solver {
namespace chebyshev {
namespace {


GKO_REGISTER_OPERATION(initialize, ir::initialize);


}  // anonymous namespace
}  // namespace chebyshev


template <typename ValueType>
Chebyshev<ValueType>::Chebyshev(const Factory* factory,
                                std::shared_ptr<const LinOp> system_matrix)
    : EnableLinOp<Chebyshev>(factory->get_executor(),
                             gko::transpose(system_matrix->get_size())),
      EnablePreconditionedIterativeSolver<ValueType, Chebyshev<ValueType>>{
          std::move(system_matrix), factory->get_parameters()},
      parameters_{factory->get_parameters()}
{
    this->set_default_initial_guess(parameters_.default_initial_guess);
    center_ = (std::get<0>(parameters_.foci) + std::get<1>(parameters_.foci)) /
              ValueType{2};
    foci_direction_ =
        (std::get<1>(parameters_.foci) - std::get<0>(parameters_.foci)) /
        ValueType{2};
    // if changing the lower/upper eig, need to reset it to zero
    num_generated_scalar_ = 0;
    num_max_generation_ = 3;
}


template <typename ValueType>
Chebyshev<ValueType>& Chebyshev<ValueType>::operator=(const Chebyshev& other)
{
    if (&other != this) {
        EnableLinOp<Chebyshev>::operator=(other);
        EnablePreconditionedIterativeSolver<
            ValueType, Chebyshev<ValueType>>::operator=(other);
        this->parameters_ = other.parameters_;
        // the workspace is not copied.
        this->num_generated_scalar_ = 0;
        this->num_max_generation_ = 3;
    }
    return *this;
}


template <typename ValueType>
Chebyshev<ValueType>& Chebyshev<ValueType>::operator=(Chebyshev&& other)
{
    if (&other != this) {
        EnableLinOp<Chebyshev>::operator=(std::move(other));
        EnablePreconditionedIterativeSolver<
            ValueType, Chebyshev<ValueType>>::operator=(std::move(other));
        // the workspace is not moved.
        this->num_generated_scalar_ = 0;
        this->num_max_generation_ = 3;
    }
    return *this;
}


template <typename ValueType>
Chebyshev<ValueType>::Chebyshev(const Chebyshev& other)
    : Chebyshev(other.get_executor())
{
    *this = other;
}


template <typename ValueType>
Chebyshev<ValueType>::Chebyshev(Chebyshev&& other)
    : Chebyshev(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType>
std::unique_ptr<LinOp> Chebyshev<ValueType>::transpose() const
{
    return build()
        .with_generated_preconditioner(
            share(as<Transposable>(this->get_preconditioner())->transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .with_foci(parameters_.foci)
        .on(this->get_executor())
        ->generate(
            share(as<Transposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<LinOp> Chebyshev<ValueType>::conj_transpose() const
{
    return build()
        .with_generated_preconditioner(share(
            as<Transposable>(this->get_preconditioner())->conj_transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .with_foci(conj(std::get<0>(parameters_.foci)),
                   conj(std::get<1>(parameters_.foci)))
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType>
void Chebyshev<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    this->apply_with_initial_guess(b, x, this->get_default_initial_guess());
}


template <typename ValueType>
void Chebyshev<ValueType>::apply_with_initial_guess_impl(
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
void Chebyshev<ValueType>::apply_dense_impl(const VectorType* dense_b,
                                            VectorType* dense_x,
                                            initial_guess_mode guess) const
{
    using Vector = matrix::Dense<ValueType>;
    using ws = workspace_traits<Chebyshev>;

    auto exec = this->get_executor();
    this->setup_workspace();

    GKO_SOLVER_VECTOR(residual, dense_b);
    GKO_SOLVER_VECTOR(inner_solution, dense_b);
    GKO_SOLVER_VECTOR(update_solution, dense_b);

    auto old_num_max_generation = num_max_generation_;
    // Use the scalar first
    // get the iteration information from stopping criterion.
    if (auto combined =
            std::dynamic_pointer_cast<const gko::stop::Combined::Factory>(
                this->get_stop_criterion_factory())) {
        for (const auto& factory : combined->get_parameters().criteria) {
            if (auto iter_stop = std::dynamic_pointer_cast<
                    const gko::stop::Iteration::Factory>(factory)) {
                num_max_generation_ = std::max(
                    num_max_generation_, iter_stop->get_parameters().max_iters);
            }
        }
    } else if (auto iter_stop = std::dynamic_pointer_cast<
                   const gko::stop::Iteration::Factory>(
                   this->get_stop_criterion_factory())) {
        num_max_generation_ = std::max(num_max_generation_,
                                       iter_stop->get_parameters().max_iters);
    }
    // Regenerate the vector if we realloc the memory.
    if (old_num_max_generation != num_max_generation_) {
        num_generated_scalar_ = 0;
    }
    auto alpha = this->template create_workspace_scalar<ValueType>(
        GKO_SOLVER_TRAITS::alpha, num_max_generation_ + 1);
    auto beta = this->template create_workspace_scalar<ValueType>(
        GKO_SOLVER_TRAITS::beta, num_max_generation_ + 1);

    GKO_SOLVER_ONE_MINUS_ONE();

    auto alpha_ref = ValueType{1} / center_;
    auto beta_ref = ValueType{0.5} * (foci_direction_ * alpha_ref) *
                    (foci_direction_ * alpha_ref);

    auto& stop_status = this->template create_workspace_array<stopping_status>(
        ws::stop, dense_b->get_size()[1]);
    exec->run(chebyshev::make_initialize(&stop_status));
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
        auto log_func = [this](auto solver, auto dense_b, auto dense_x,
                               auto iter, auto residual_ptr,
                               array<stopping_status>& stop_status,
                               bool all_stopped) {
            this->template log<log::Logger::iteration_complete>(
                solver, dense_b, dense_x, iter, residual_ptr, nullptr, nullptr,
                &stop_status, all_stopped);
        };
        bool all_stopped = update_residual(
            this, iter, dense_b, dense_x, residual, residual_ptr,
            stop_criterion, stop_status, log_func);
        if (all_stopped) {
            break;
        }

        if (this->get_preconditioner()->apply_uses_initial_guess()) {
            // Use the inner solver to solve
            // A * inner_solution = residual
            // with residual as initial guess.
            inner_solution->copy_from(residual_ptr);
        }
        this->get_preconditioner()->apply(residual_ptr, inner_solution);
        size_type index =
            (iter >= num_max_generation_) ? num_max_generation_ : iter;
        auto alpha_scalar =
            alpha->create_submatrix(span{0, 1}, span{index, index + 1});
        auto beta_scalar =
            beta->create_submatrix(span{0, 1}, span{index, index + 1});
        if (iter == 0) {
            if (num_generated_scalar_ < num_max_generation_) {
                alpha_scalar->fill(alpha_ref);
                // unused beta for first iteration, but fill zero
                beta_scalar->fill(zero<ValueType>());
                num_generated_scalar_++;
            }
            // x = x + alpha * inner_solution
            dense_x->add_scaled(alpha_scalar.get(), inner_solution);
            update_solution->copy_from(inner_solution);
            continue;
        }
        // beta_ref for iter == 1 is initialized in the beginning
        if (iter > 1) {
            beta_ref = (foci_direction_ * alpha_ref / ValueType{2.0}) *
                       (foci_direction_ * alpha_ref / ValueType{2.0});
        }
        alpha_ref = ValueType{1.0} / (center_ - beta_ref / alpha_ref);
        // The last one is always the updated one
        if (num_generated_scalar_ < num_max_generation_ ||
            iter >= num_max_generation_) {
            alpha_scalar->fill(alpha_ref);
            beta_scalar->fill(beta_ref);
        }
        if (num_generated_scalar_ < num_max_generation_) {
            num_generated_scalar_++;
        }
        // z = z + beta * p
        // p = z
        inner_solution->add_scaled(beta_scalar.get(), update_solution);
        update_solution->copy_from(inner_solution);
        // x + alpha * p
        dense_x->add_scaled(alpha_scalar.get(), update_solution);
    }
}


template <typename ValueType>
void Chebyshev<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                      const LinOp* beta, LinOp* x) const
{
    this->apply_with_initial_guess(alpha, b, beta, x,
                                   this->get_default_initial_guess());
}

template <typename ValueType>
void Chebyshev<ValueType>::apply_with_initial_guess_impl(
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
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


template <typename ValueType>
int workspace_traits<Chebyshev<ValueType>>::num_arrays(const Solver&)
{
    return 1;
}


template <typename ValueType>
int workspace_traits<Chebyshev<ValueType>>::num_vectors(const Solver&)
{
    return 7;
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Chebyshev<ValueType>>::op_names(
    const Solver&)
{
    return {
        "residual", "inner_solution", "update_solution", "alpha", "beta",
        "one",      "minus_one",
    };
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Chebyshev<ValueType>>::array_names(
    const Solver&)
{
    return {"stop"};
}


template <typename ValueType>
std::vector<int> workspace_traits<Chebyshev<ValueType>>::scalars(const Solver&)
{
    return {};
}


template <typename ValueType>
std::vector<int> workspace_traits<Chebyshev<ValueType>>::vectors(const Solver&)
{
    return {residual, inner_solution, update_solution};
}


#define GKO_DECLARE_CHEBYSHEV(_type) class Chebyshev<_type>
#define GKO_DECLARE_CHEBYSHEV_TRAITS(_type) \
    struct workspace_traits<Chebyshev<_type>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CHEBYSHEV);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CHEBYSHEV_TRAITS);


}  // namespace solver
}  // namespace gko
