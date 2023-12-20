// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/log/convergence.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


#include "core/base/dispatch_helper.hpp"
#include "core/distributed/helpers.hpp"


namespace gko {
namespace log {


template <typename ValueType>
void Convergence<ValueType>::on_criterion_check_completed(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const LinOp* residual, const LinOp* residual_norm,
    const LinOp* implicit_sq_resnorm, const LinOp* solution,
    const uint8& stopping_id, const bool& set_finalized,
    const array<stopping_status>* status, const bool& one_changed,
    const bool& stopped) const
{
    this->on_iteration_complete(nullptr, nullptr, solution, num_iterations,
                                residual, residual_norm, implicit_sq_resnorm,
                                status, stopped);
}


template <typename ValueType>
void Convergence<ValueType>::on_criterion_check_completed(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const LinOp* residual, const LinOp* residual_norm, const LinOp* solution,
    const uint8& stopping_id, const bool& set_finalized,
    const array<stopping_status>* status, const bool& one_changed,
    const bool& stopped) const
{
    this->on_criterion_check_completed(
        criterion, num_iterations, residual, residual_norm, nullptr, solution,
        stopping_id, set_finalized, status, one_changed, stopped);
}


template <typename ValueType>
void Convergence<ValueType>::on_iteration_complete(
    const LinOp* solver, const LinOp* b, const LinOp* x,
    const size_type& num_iterations, const LinOp* residual,
    const LinOp* residual_norm, const LinOp* implicit_resnorm_sq,
    const array<stopping_status>* status, const bool stopped) const
{
    auto update_history = [&](auto& container, auto&& new_val, bool is_norm) {
        if (history_ == convergence_history::none && stopped) {
            container.back().reset(std::move(new_val));
            return;
        }
        if (is_norm || history_ == convergence_history::full) {
            container.emplace_back(std::move(new_val));
        }
    };
    if (stopped) {
        array<stopping_status> tmp(status->get_executor()->get_master(),
                                   *status);
        convergence_status_ = true;
        for (int i = 0; i < status->get_size(); i++) {
            if (!tmp.get_data()[i].has_converged()) {
                convergence_status_ = false;
                break;
            }
        }
        num_iterations_ = num_iterations;
    }
    if (residual != nullptr) {
        update_history(residual_, residual->clone(), false);
    }
    if (implicit_resnorm_sq != nullptr) {
        update_history(implicit_sq_resnorm_, implicit_resnorm_sq->clone(),
                       true);
    }
    if (residual_norm != nullptr) {
        update_history(residual_norm_, residual_norm->clone(), true);
    } else if (residual != nullptr) {
        using NormVector = matrix::Dense<remove_complex<ValueType>>;
        detail::vector_dispatch<ValueType>(residual, [&](const auto* dense_r) {
            update_history(
                residual_norm_,
                NormVector::create(residual->get_executor(),
                                   dim<2>{1, residual->get_size()[1]}),
                true);
            dense_r->compute_norm2(residual_norm_.back());
        });
    } else if (dynamic_cast<const solver::detail::SolverBaseLinOp*>(solver) &&
               b != nullptr && x != nullptr) {
        auto system_mtx =
            dynamic_cast<const solver::detail::SolverBaseLinOp*>(solver)
                ->get_system_matrix();
        using Vector = matrix::Dense<ValueType>;
        using NormVector = matrix::Dense<remove_complex<ValueType>>;
        detail::vector_dispatch<ValueType>(b, [&](const auto* dense_b) {
            detail::vector_dispatch<ValueType>(x, [&](const auto* dense_x) {
                auto exec = system_mtx->get_executor();
                update_history(residual_, dense_b->clone(), false);
                system_mtx->apply(initialize<Vector>({-1.0}, exec), dense_x,
                                  initialize<Vector>({1.0}, exec),
                                  residual_.back());
                update_history(
                    residual_norm_,
                    NormVector::create(
                        exec, dim<2>{1, residual_.back()->get_size()[1]}),
                    true);
                detail::vector_dispatch<ValueType>(
                    residual_.back(), [&](const auto* actual_residual) {
                        actual_residual->compute_norm2(residual_norm_.back());
                    });
            });
        });
    }
}


template <typename ValueType>
void Convergence<ValueType>::clear_history() const
{
    residual_ = {nullptr};
    residual_norm_ = {nullptr};
    implicit_sq_resnorm_ = {nullptr};
}


#define GKO_DECLARE_CONVERGENCE(_type) class Convergence<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONVERGENCE);


}  // namespace log
}  // namespace gko
