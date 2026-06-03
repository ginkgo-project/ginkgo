// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/log/record.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


namespace gko {
namespace log {


template <typename T>
std::unique_ptr<T> clone_or_nullptr(T* input)
{
    // whether throw exception if input is not cloneable?
    if (auto tmp = dynamic_cast<const Cloneable*>(input)) {
        return as<T>(tmp->clone());
    }
    return nullptr;
}


iteration_complete_data::iteration_complete_data(
    const LinOp* solver, const AbstractMultiVector* right_hand_side,
    const AbstractMultiVector* solution, const size_type num_iterations,
    const AbstractMultiVector* residual,
    const AbstractMultiVector* residual_norm,
    const AbstractMultiVector* implicit_sq_residual_norm,
    const gko::array<stopping_status>* status, bool all_stopped)

    : num_iterations{num_iterations}, all_stopped(all_stopped)
{
    this->solver = clone_or_nullptr(solver);
    this->solution = clone_or_nullptr(solution);
    if (right_hand_side != nullptr) {
        this->right_hand_side = clone_or_nullptr(right_hand_side);
    }
    if (residual != nullptr) {
        this->residual = clone_or_nullptr(residual);
    }
    if (residual_norm != nullptr) {
        this->residual_norm = clone_or_nullptr(residual_norm);
    }
    if (implicit_sq_residual_norm != nullptr) {
        this->implicit_sq_residual_norm =
            clone_or_nullptr(implicit_sq_residual_norm);
    }
    if (status != nullptr) {
        this->status = *status;
    }
}


polymorphic_object_data::polymorphic_object_data(
    const Executor* exec, const PolymorphicObject* input,
    const PolymorphicObject* output)

    : exec{exec}
{
    this->input = clone_or_nullptr(input);
    if (output != nullptr) {
        this->output = clone_or_nullptr(output);
    }
}


linop_data::linop_data(const LinOp* A, const AbstractMultiVector* alpha,
                       const AbstractMultiVector* b,
                       const AbstractMultiVector* beta,
                       const AbstractMultiVector* x)

{
    this->A = clone_or_nullptr(A);
    if (alpha != nullptr) {
        this->alpha = clone_or_nullptr(alpha);
    }
    this->b = clone_or_nullptr(b);
    if (beta != nullptr) {
        this->beta = clone_or_nullptr(beta);
    }
    this->x = clone_or_nullptr(x);
}


linop_factory_data::linop_factory_data(const LinOpFactory* factory,
                                       const LinOp* input, const LinOp* output)

    : factory{factory}
{
    this->input = clone_or_nullptr(input);
    if (output != nullptr) {
        this->output = clone_or_nullptr(output);
    }
}


criterion_data::criterion_data(const stop::Criterion* criterion,
                               const size_type& num_iterations,
                               const AbstractMultiVector* residual,
                               const AbstractMultiVector* residual_norm,
                               const AbstractMultiVector* solution,
                               const uint8 stopping_id,
                               const bool set_finalized,
                               const array<stopping_status>* status,
                               const bool oneChanged, const bool converged)

    : criterion{criterion},
      num_iterations{num_iterations},
      residual{nullptr},
      residual_norm{nullptr},
      solution{nullptr},
      stopping_id{stopping_id},
      set_finalized{set_finalized},
      status{status},
      oneChanged{oneChanged},
      converged{converged}
{
    if (residual != nullptr) {
        this->residual = residual->clone();
    }
    if (residual_norm != nullptr) {
        this->residual_norm = residual_norm->clone();
    }
    if (solution != nullptr) {
        this->solution = solution->clone();
    }
}
void Record::on_allocation_started(const Executor* exec,
                                   const size_type& num_bytes) const
{
    append_deque(data_.allocation_started,
                 (std::unique_ptr<executor_data>(
                     new executor_data{exec, num_bytes, 0})));
}


void Record::on_allocation_completed(const Executor* exec,
                                     const size_type& num_bytes,
                                     const uintptr& location) const
{
    append_deque(data_.allocation_completed,
                 (std::unique_ptr<executor_data>(
                     new executor_data{exec, num_bytes, location})));
}


void Record::on_free_started(const Executor* exec,
                             const uintptr& location) const
{
    append_deque(
        data_.free_started,
        (std::unique_ptr<executor_data>(new executor_data{exec, 0, location})));
}


void Record::on_free_completed(const Executor* exec,
                               const uintptr& location) const
{
    append_deque(
        data_.free_completed,
        (std::unique_ptr<executor_data>(new executor_data{exec, 0, location})));
}


void Record::on_copy_started(const Executor* from, const Executor* to,
                             const uintptr& location_from,
                             const uintptr& location_to,
                             const size_type& num_bytes) const
{
    using tuple = std::tuple<executor_data, executor_data>;
    append_deque(
        data_.copy_started,
        (std::unique_ptr<tuple>(new tuple{{from, num_bytes, location_from},
                                          {to, num_bytes, location_to}})));
}


void Record::on_copy_completed(const Executor* from, const Executor* to,
                               const uintptr& location_from,
                               const uintptr& location_to,
                               const size_type& num_bytes) const
{
    using tuple = std::tuple<executor_data, executor_data>;
    append_deque(
        data_.copy_completed,
        (std::unique_ptr<tuple>(new tuple{{from, num_bytes, location_from},
                                          {to, num_bytes, location_to}})));
}


void Record::on_operation_launched(const Executor* exec,
                                   const Operation* operation) const
{
    append_deque(
        data_.operation_launched,
        (std::unique_ptr<operation_data>(new operation_data{exec, operation})));
}


void Record::on_operation_completed(const Executor* exec,
                                    const Operation* operation) const
{
    append_deque(
        data_.operation_completed,
        (std::unique_ptr<operation_data>(new operation_data{exec, operation})));
}


void Record::on_polymorphic_object_create_started(
    const Executor* exec, const PolymorphicObject* po) const
{
    append_deque(data_.polymorphic_object_create_started,
                 (std::unique_ptr<polymorphic_object_data>(
                     new polymorphic_object_data{exec, po})));
}


void Record::on_polymorphic_object_create_completed(
    const Executor* exec, const PolymorphicObject* input,
    const PolymorphicObject* output) const
{
    append_deque(data_.polymorphic_object_create_completed,
                 (std::unique_ptr<polymorphic_object_data>(
                     new polymorphic_object_data{exec, input, output})));
}


void Record::on_polymorphic_object_copy_started(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    append_deque(data_.polymorphic_object_copy_started,
                 (std::unique_ptr<polymorphic_object_data>(
                     new polymorphic_object_data{exec, from, to})));
}


void Record::on_polymorphic_object_copy_completed(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    append_deque(data_.polymorphic_object_copy_completed,
                 (std::unique_ptr<polymorphic_object_data>(
                     new polymorphic_object_data{exec, from, to})));
}


void Record::on_polymorphic_object_move_started(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    append_deque(data_.polymorphic_object_move_started,
                 (std::make_unique<polymorphic_object_data>(exec, from, to)));
}


void Record::on_polymorphic_object_move_completed(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    append_deque(data_.polymorphic_object_move_completed,
                 (std::make_unique<polymorphic_object_data>(exec, from, to)));
}


void Record::on_polymorphic_object_deleted(const Executor* exec,
                                           const PolymorphicObject* po) const
{
    append_deque(data_.polymorphic_object_deleted,
                 (std::unique_ptr<polymorphic_object_data>(
                     new polymorphic_object_data{exec, po})));
}


void Record::on_linop_apply_started(const LinOp* A,
                                    const AbstractMultiVector* b,
                                    const AbstractMultiVector* x) const
{
    append_deque(data_.linop_apply_started,
                 (std::unique_ptr<linop_data>(
                     new linop_data{A, nullptr, b, nullptr, x})));
}


void Record::on_linop_apply_completed(const LinOp* A,
                                      const AbstractMultiVector* b,
                                      const AbstractMultiVector* x) const
{
    append_deque(data_.linop_apply_completed,
                 (std::unique_ptr<linop_data>(
                     new linop_data{A, nullptr, b, nullptr, x})));
}


void Record::on_linop_advanced_apply_started(const LinOp* A,
                                             const AbstractMultiVector* alpha,
                                             const AbstractMultiVector* b,
                                             const AbstractMultiVector* beta,
                                             const AbstractMultiVector* x) const
{
    append_deque(
        data_.linop_advanced_apply_started,
        (std::unique_ptr<linop_data>(new linop_data{A, alpha, b, beta, x})));
}


void Record::on_linop_advanced_apply_completed(
    const LinOp* A, const AbstractMultiVector* alpha,
    const AbstractMultiVector* b, const AbstractMultiVector* beta,
    const AbstractMultiVector* x) const
{
    append_deque(
        data_.linop_advanced_apply_completed,
        (std::unique_ptr<linop_data>(new linop_data{A, alpha, b, beta, x})));
}


void Record::on_linop_factory_generate_started(const LinOpFactory* factory,
                                               const LinOp* input) const
{
    append_deque(data_.linop_factory_generate_started,
                 (std::unique_ptr<linop_factory_data>(
                     new linop_factory_data{factory, input, nullptr})));
}


void Record::on_linop_factory_generate_completed(const LinOpFactory* factory,
                                                 const LinOp* input,
                                                 const LinOp* output) const
{
    append_deque(data_.linop_factory_generate_completed,
                 (std::unique_ptr<linop_factory_data>(
                     new linop_factory_data{factory, input, output})));
}


std::unique_ptr<Record> Record::create(std::shared_ptr<const Executor> exec,
                                       const mask_type& enabled_events,
                                       size_type max_storage)

{
    return std::unique_ptr<Record>(new Record(enabled_events, max_storage));
}


std::unique_ptr<Record> Record::create(const mask_type& enabled_events,
                                       size_type max_storage)

{
    return std::unique_ptr<Record>(new Record(enabled_events, max_storage));
}


const Record::logged_data& Record::get() const noexcept { return data_; }


Record::logged_data& Record::get() noexcept { return data_; }


Record::Record(std::shared_ptr<const gko::Executor> exec,
               const mask_type& enabled_events, size_type max_storage)

    : Record(enabled_events, max_storage)
{}


Record::Record(const mask_type& enabled_events, size_type max_storage)
    : Logger(enabled_events), max_storage_{max_storage}
{}


void Record::on_criterion_check_started(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const AbstractMultiVector* residual,
    const AbstractMultiVector* residual_norm,
    const AbstractMultiVector* solution, const uint8& stopping_id,
    const bool& set_finalized) const
{
    append_deque(data_.criterion_check_started,
                 (std::unique_ptr<criterion_data>(new criterion_data{
                     criterion, num_iterations, residual, residual_norm,
                     solution, stopping_id, set_finalized})));
}


void Record::on_criterion_check_completed(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const AbstractMultiVector* residual,
    const AbstractMultiVector* residual_norm,
    const AbstractMultiVector* implicit_residual_norm_sq,
    const AbstractMultiVector* solution, const uint8& stopping_id,
    const bool& set_finalized, const array<stopping_status>* status,
    const bool& oneChanged, const bool& converged) const
{
    append_deque(
        data_.criterion_check_completed,
        (std::unique_ptr<criterion_data>(new criterion_data{
            criterion, num_iterations, residual, residual_norm, solution,
            stopping_id, set_finalized, status, oneChanged, converged})));
}


void Record::on_criterion_check_completed(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const AbstractMultiVector* residual,
    const AbstractMultiVector* residual_norm,
    const AbstractMultiVector* solution, const uint8& stopping_id,
    const bool& set_finalized, const array<stopping_status>* status,
    const bool& oneChanged, const bool& converged) const
{
    this->on_criterion_check_completed(
        criterion, num_iterations, residual, residual_norm, nullptr, solution,
        stopping_id, set_finalized, status, oneChanged, converged);
}


void Record::on_iteration_complete(
    const LinOp* solver, const size_type& num_iterations,
    const AbstractMultiVector* residual, const AbstractMultiVector* solution,
    const AbstractMultiVector* residual_norm) const
{
    this->on_iteration_complete(solver, nullptr, solution, num_iterations,
                                residual, residual_norm, nullptr, nullptr,
                                false);
}


void Record::on_iteration_complete(
    const LinOp* solver, const size_type& num_iterations,
    const AbstractMultiVector* residual, const AbstractMultiVector* solution,
    const AbstractMultiVector* residual_norm,
    const AbstractMultiVector* implicit_sq_residual_norm) const
{
    this->on_iteration_complete(solver, nullptr, solution, num_iterations,
                                residual, residual_norm,
                                implicit_sq_residual_norm, nullptr, false);
}


void Record::on_iteration_complete(
    const LinOp* solver, const AbstractMultiVector* right_hand_side,
    const AbstractMultiVector* solution, const size_type& num_iterations,
    const AbstractMultiVector* residual,
    const AbstractMultiVector* residual_norm,
    const AbstractMultiVector* implicit_resnorm_sq,
    const array<stopping_status>* status, bool stopped) const
{
    append_deque(
        data_.iteration_completed,
        (std::unique_ptr<iteration_complete_data>(new iteration_complete_data{
            solver, right_hand_side, solution, num_iterations, residual,
            residual_norm, implicit_resnorm_sq, status, stopped})));
}


}  // namespace log
}  // namespace gko
