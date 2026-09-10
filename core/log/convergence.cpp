// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/log/convergence.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
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
    const AbstractMultiVector* residual,
    const AbstractMultiVector* residual_norm,
    const AbstractMultiVector* implicit_sq_resnorm,
    const AbstractMultiVector* solution, const uint8& stopping_id,
    const bool& set_finalized, const array<stopping_status>* status,
    const bool& one_changed, const bool& stopped) const
{
    this->on_iteration_complete(nullptr, nullptr, solution, num_iterations,
                                residual, residual_norm, implicit_sq_resnorm,
                                status, stopped);
}


template <typename ValueType>
void Convergence<ValueType>::on_criterion_check_completed(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const AbstractMultiVector* residual,
    const AbstractMultiVector* residual_norm,
    const AbstractMultiVector* solution, const uint8& stopping_id,
    const bool& set_finalized, const array<stopping_status>* status,
    const bool& one_changed, const bool& stopped) const
{
    this->on_criterion_check_completed(
        criterion, num_iterations, residual, residual_norm, nullptr, solution,
        stopping_id, set_finalized, status, one_changed, stopped);
}


template <typename ValueType>
void Convergence<ValueType>::on_iteration_complete(
    const LinOp* solver, const AbstractMultiVector* b,
    const AbstractMultiVector* x, const size_type& num_iterations,
    const AbstractMultiVector* residual,
    const AbstractMultiVector* residual_norm,
    const AbstractMultiVector* implicit_resnorm_sq,
    const array<stopping_status>* status, const bool stopped) const
{
    if (stopped) {
        array<stopping_status> tmp(status->get_executor()->get_master(),
                                   *status);
        this->convergence_status_ = true;
        for (int i = 0; i < status->get_size(); i++) {
            if (!tmp.get_data()[i].has_converged()) {
                this->convergence_status_ = false;
                break;
            }
        }
        this->num_iterations_ = num_iterations;
        if (residual != nullptr) {
            this->residual_ = residual->clone();
        }
        if (implicit_resnorm_sq != nullptr) {
            this->implicit_sq_resnorm_ = implicit_resnorm_sq->clone();
        }
        if (residual_norm != nullptr) {
            this->residual_norm_ = residual_norm->clone();
        } else if (residual != nullptr) {
            using NormVector = matrix::MultiVector<remove_complex<ValueType>>;
            this->residual_norm_ = NormVector::create(
                residual->get_executor(), dim<2>{1, residual->get_size()[1]});
            residual->compute_norm2(this->residual_norm_);
        } else if (dynamic_cast<const solver::detail::SolverBaseLinOp*>(
                       solver) &&
                   b != nullptr && x != nullptr) {
            auto system_mtx =
                dynamic_cast<const solver::detail::SolverBaseLinOp*>(solver)
                    ->get_system_matrix();
            using Vector = matrix::MultiVector<ValueType>;
            using NormVector = matrix::MultiVector<remove_complex<ValueType>>;
            auto converted_b = b->as_precision(precision_v<ValueType>);
            auto exec = system_mtx->get_executor();
            auto residual_tmp = converted_b->clone();
            this->residual_norm_ = NormVector::create(
                exec, dim<2>{1, residual_tmp->get_size()[1]});
            system_mtx->apply(initialize<Vector>({-1.0}, exec),
                              x->as_precision(precision_v<ValueType>).get(),
                              initialize<Vector>({1.0}, exec), residual_tmp);
            residual_tmp->compute_norm2(this->residual_norm_);
        }
    }
}


template <typename ValueType>
std::unique_ptr<Convergence<ValueType>> Convergence<ValueType>::create(
    std::shared_ptr<const Executor>, const mask_type& enabled_events)

{
    return std::unique_ptr<Convergence>(new Convergence(enabled_events));
}


template <typename ValueType>
std::unique_ptr<Convergence<ValueType>> Convergence<ValueType>::create(
    const mask_type& enabled_events)

{
    return std::unique_ptr<Convergence>(new Convergence(enabled_events));
}


template <typename ValueType>
bool Convergence<ValueType>::has_converged() const noexcept
{
    return convergence_status_;
}


template <typename ValueType>
void Convergence<ValueType>::reset_convergence_status()
{
    this->convergence_status_ = false;
}


template <typename ValueType>
const size_type& Convergence<ValueType>::get_num_iterations() const noexcept

{
    return num_iterations_;
}


template <typename ValueType>
const AbstractMultiVector* Convergence<ValueType>::get_residual() const noexcept
{
    return residual_.get();
}


template <typename ValueType>
const AbstractMultiVector* Convergence<ValueType>::get_residual_norm()
    const noexcept

{
    return residual_norm_.get();
}


template <typename ValueType>
const AbstractMultiVector* Convergence<ValueType>::get_implicit_sq_resnorm()
    const noexcept

{
    return implicit_sq_resnorm_.get();
}


template <typename ValueType>
Convergence<ValueType>::Convergence(std::shared_ptr<const gko::Executor>,
                                    const mask_type& enabled_events)

    : Logger(enabled_events)
{}


template <typename ValueType>
Convergence<ValueType>::Convergence(const mask_type& enabled_events)

    : Logger(enabled_events)
{}


#define GKO_DECLARE_CONVERGENCE(ValueType) class Convergence<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONVERGENCE);


}  // namespace log
}  // namespace gko
