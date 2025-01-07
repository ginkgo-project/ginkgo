// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/chebyshev.hpp"

#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/solver_base.hpp>

#include "core/config/solver_config.hpp"
#include "core/distributed/helpers.hpp"
#include "core/solver/chebyshev_kernels.hpp"
#include "core/solver/ir_kernels.hpp"
#include "core/solver/solver_base.hpp"
#include "core/solver/solver_boilerplate.hpp"
#include "core/solver/update_residual.hpp"


namespace gko {
namespace solver {
namespace chebyshev {
namespace {


GKO_REGISTER_OPERATION(initialize, ir::initialize);
GKO_REGISTER_OPERATION(init_update, chebyshev::init_update);
GKO_REGISTER_OPERATION(update, chebyshev::update);


}  // anonymous namespace
}  // namespace chebyshev

template <typename ValueType>
typename Chebyshev<ValueType>::parameters_type Chebyshev<ValueType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = solver::Chebyshev<ValueType>::build();
    common_solver_parse(params, config, context, td_for_child);
    if (auto& obj = config.get("foci")) {
        auto arr = obj.get_array();
        if (arr.size() != 2) {
            GKO_INVALID_CONFIG_VALUE("foci", "must contain two elements");
        }
        params.with_foci(gko::config::get_value<ValueType>(arr.at(0)),
                         gko::config::get_value<ValueType>(arr.at(1)));
    }
    if (auto& obj = config.get("default_initial_guess")) {
        params.with_default_initial_guess(
            gko::config::get_value<solver::initial_guess_mode>(obj));
    }
    return params;
}


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
}


template <typename ValueType>
Chebyshev<ValueType>& Chebyshev<ValueType>::operator=(const Chebyshev& other)
{
    if (&other != this) {
        EnableLinOp<Chebyshev>::operator=(other);
        EnablePreconditionedIterativeSolver<
            ValueType, Chebyshev<ValueType>>::operator=(other);
        this->parameters_ = other.parameters_;
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
        if (iter == 0) {
            // x = x + alpha * inner_solution
            // update_solultion = inner_solution
            exec->run(chebyshev::make_init_update(
                alpha_ref, gko::detail::get_local(inner_solution),
                gko::detail::get_local(update_solution),
                gko::detail::get_local(dense_x)));
            continue;
        }
        // beta_ref for iter == 1 is initialized in the beginning
        if (iter > 1) {
            beta_ref = (foci_direction_ * alpha_ref / ValueType{2.0}) *
                       (foci_direction_ * alpha_ref / ValueType{2.0});
        }
        alpha_ref = ValueType{1.0} / (center_ - beta_ref / alpha_ref);
        // z = z + beta * p
        // p = z
        // x += alpha * p
        exec->run(chebyshev::make_update(
            alpha_ref, beta_ref, gko::detail::get_local(inner_solution),
            gko::detail::get_local(update_solution),
            gko::detail::get_local(dense_x)));
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
    return 5;
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Chebyshev<ValueType>>::op_names(
    const Solver&)
{
    return {
        "residual", "inner_solution", "update_solution", "one", "minus_one",
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
