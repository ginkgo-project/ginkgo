// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/chebyshev.hpp"

#include <string>

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
namespace ir {
namespace {


GKO_REGISTER_OPERATION(initialize, ir::initialize);


}
}  // namespace ir


namespace chebyshev {
namespace {


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
    config::config_check_decorator config_check(config);
    config::common_solver_parse(params, config_check, context, td_for_child);
    if (auto& obj = config_check.get("foci")) {
        auto arr = obj.get_array();
        if (arr.size() != 2) {
            GKO_INVALID_CONFIG_VALUE("foci", "must contain two elements");
        }
        params.with_foci(
            gko::config::get_value<detail::coeff_type<ValueType>>(arr.at(0)),
            gko::config::get_value<detail::coeff_type<ValueType>>(arr.at(1)));
    }
    if (auto& obj = config_check.get("default_initial_guess")) {
        params.with_default_initial_guess(
            gko::config::get_value<solver::initial_guess_mode>(obj));
    }

    return params;
}


template <typename ValueType>
Chebyshev<ValueType>::Chebyshev(std::shared_ptr<const Executor> exec)
    : EnableLinOp<Chebyshev>(std::move(exec))
{}


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
    auto left_foci = std::get<0>(parameters_.foci);
    auto right_foci = std::get<1>(parameters_.foci);
    GKO_ASSERT(real(left_foci) <= real(right_foci));
    GKO_ASSERT(is_nonzero(left_foci) || is_nonzero(right_foci));
    center_ =
        (left_foci + right_foci) / solver::detail::coeff_type<ValueType>{2};
    foci_direction_ =
        (right_foci - left_foci) / solver::detail::coeff_type<ValueType>{2};
    // if the center is zero then the alpha will be inf
    GKO_ASSERT(is_nonzero(center_));
}


template <typename ValueType>
Chebyshev<ValueType>& Chebyshev<ValueType>::operator=(const Chebyshev& other)
{
    if (&other != this) {
        EnableLinOp<Chebyshev>::operator=(other);
        EnablePreconditionedIterativeSolver<
            ValueType, Chebyshev<ValueType>>::operator=(other);
        this->parameters_ = other.parameters_;
        this->center_ = other.center_;
        this->foci_direction_ = other.foci_direction_;
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
        this->parameters_ = std::exchange(other.parameters_, parameters_type{});
        this->center_ = std::exchange(other.center_,
                                      solver::detail::coeff_type<ValueType>{});
        this->foci_direction_ = std::exchange(
            other.foci_direction_, solver::detail::coeff_type<ValueType>{});
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
    using coeff_type = solver::detail::coeff_type<ValueType>;

    auto exec = this->get_executor();
    this->setup_workspace();

    GKO_SOLVER_VECTOR(residual, dense_b);
    GKO_SOLVER_VECTOR(inner_solution, dense_b);
    GKO_SOLVER_VECTOR(update_solution, dense_b);

    GKO_SOLVER_ONE_MINUS_ONE();

    auto alpha_host = coeff_type{1} / center_;
    auto beta_host = coeff_type{0.5} * (foci_direction_ * alpha_host) *
                     (foci_direction_ * alpha_host);

    auto& stop_status = this->template create_workspace_array<stopping_status>(
        ws::stop, dense_b->get_size()[1]);
    exec->run(ir::make_initialize(&stop_status));
    exec->run(ir::make_initialize(&stop_status));
    auto& stop_indicators =
        this->template create_workspace_array<bool>(ws::indicators, 2);
    stop_indicators.set_executor(this->get_executor()->get_master());
    stop_indicators.get_data()[0] = false;
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
            stop_criterion, stop_status, &stop_indicators, log_func);
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
                alpha_host, gko::detail::get_local(inner_solution),
                gko::detail::get_local(update_solution),
                gko::detail::get_local(dense_x)));
            continue;
        }
        // beta_host for iter == 1 is initialized in the beginning
        if (iter > 1) {
            beta_host = (foci_direction_ * alpha_host / coeff_type{2.0}) *
                        (foci_direction_ * alpha_host / coeff_type{2.0});
        }
        alpha_host = coeff_type{1.0} / (center_ - beta_host / alpha_host);
        // z = z + beta * p
        // p = z
        // x += alpha * p
        exec->run(chebyshev::make_update(
            alpha_host, beta_host, gko::detail::get_local(inner_solution),
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
    return 2;
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
