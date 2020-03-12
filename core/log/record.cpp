/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/log/record.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


namespace gko {
namespace log {


void Record::on_allocation_started(const Executor *exec,
                                   const size_type &num_bytes) const
{
    append_deque(data_.allocation_started,
                 (std::unique_ptr<executor_data>(
                     new executor_data{exec, num_bytes, 0})));
}


void Record::on_allocation_completed(const Executor *exec,
                                     const size_type &num_bytes,
                                     const uintptr &location) const
{
    append_deque(data_.allocation_completed,
                 (std::unique_ptr<executor_data>(
                     new executor_data{exec, num_bytes, location})));
}


void Record::on_free_started(const Executor *exec,
                             const uintptr &location) const
{
    append_deque(
        data_.free_started,
        (std::unique_ptr<executor_data>(new executor_data{exec, 0, location})));
}


void Record::on_free_completed(const Executor *exec,
                               const uintptr &location) const
{
    append_deque(
        data_.free_completed,
        (std::unique_ptr<executor_data>(new executor_data{exec, 0, location})));
}


void Record::on_copy_started(const Executor *from, const Executor *to,
                             const uintptr &location_from,
                             const uintptr &location_to,
                             const size_type &num_bytes) const
{
    using tuple = std::tuple<executor_data, executor_data>;
    append_deque(
        data_.copy_started,
        (std::unique_ptr<tuple>(new tuple{{from, num_bytes, location_from},
                                          {to, num_bytes, location_to}})));
}


void Record::on_copy_completed(const Executor *from, const Executor *to,
                               const uintptr &location_from,
                               const uintptr &location_to,
                               const size_type &num_bytes) const
{
    using tuple = std::tuple<executor_data, executor_data>;
    append_deque(
        data_.copy_completed,
        (std::unique_ptr<tuple>(new tuple{{from, num_bytes, location_from},
                                          {to, num_bytes, location_to}})));
}


void Record::on_operation_launched(const Executor *exec,
                                   const Operation *operation) const
{
    append_deque(
        data_.operation_launched,
        (std::unique_ptr<operation_data>(new operation_data{exec, operation})));
}


void Record::on_operation_completed(const Executor *exec,
                                    const Operation *operation) const
{
    append_deque(
        data_.operation_completed,
        (std::unique_ptr<operation_data>(new operation_data{exec, operation})));
}


void Record::on_polymorphic_object_create_started(
    const Executor *exec, const PolymorphicObject *po) const
{
    append_deque(data_.polymorphic_object_create_started,
                 (std::unique_ptr<polymorphic_object_data>(
                     new polymorphic_object_data{exec, po})));
}


void Record::on_polymorphic_object_create_completed(
    const Executor *exec, const PolymorphicObject *input,
    const PolymorphicObject *output) const
{
    append_deque(data_.polymorphic_object_create_completed,
                 (std::unique_ptr<polymorphic_object_data>(
                     new polymorphic_object_data{exec, input, output})));
}


void Record::on_polymorphic_object_copy_started(
    const Executor *exec, const PolymorphicObject *from,
    const PolymorphicObject *to) const
{
    append_deque(data_.polymorphic_object_copy_started,
                 (std::unique_ptr<polymorphic_object_data>(
                     new polymorphic_object_data{exec, from, to})));
}


void Record::on_polymorphic_object_copy_completed(
    const Executor *exec, const PolymorphicObject *from,
    const PolymorphicObject *to) const
{
    append_deque(data_.polymorphic_object_copy_completed,
                 (std::unique_ptr<polymorphic_object_data>(
                     new polymorphic_object_data{exec, from, to})));
}


void Record::on_polymorphic_object_deleted(const Executor *exec,
                                           const PolymorphicObject *po) const
{
    append_deque(data_.polymorphic_object_deleted,
                 (std::unique_ptr<polymorphic_object_data>(
                     new polymorphic_object_data{exec, po})));
}


void Record::on_linop_apply_started(const LinOp *A, const LinOp *b,
                                    const LinOp *x) const
{
    append_deque(data_.linop_apply_started,
                 (std::unique_ptr<linop_data>(
                     new linop_data{A, nullptr, b, nullptr, x})));
}


void Record::on_linop_apply_completed(const LinOp *A, const LinOp *b,
                                      const LinOp *x) const
{
    append_deque(data_.linop_apply_completed,
                 (std::unique_ptr<linop_data>(
                     new linop_data{A, nullptr, b, nullptr, x})));
}


void Record::on_linop_advanced_apply_started(const LinOp *A, const LinOp *alpha,
                                             const LinOp *b, const LinOp *beta,
                                             const LinOp *x) const
{
    append_deque(
        data_.linop_advanced_apply_started,
        (std::unique_ptr<linop_data>(new linop_data{A, alpha, b, beta, x})));
}


void Record::on_linop_advanced_apply_completed(const LinOp *A,
                                               const LinOp *alpha,
                                               const LinOp *b,
                                               const LinOp *beta,
                                               const LinOp *x) const
{
    append_deque(
        data_.linop_advanced_apply_completed,
        (std::unique_ptr<linop_data>(new linop_data{A, alpha, b, beta, x})));
}


void Record::on_linop_factory_generate_started(const LinOpFactory *factory,
                                               const LinOp *input) const
{
    append_deque(data_.linop_factory_generate_started,
                 (std::unique_ptr<linop_factory_data>(
                     new linop_factory_data{factory, input, nullptr})));
}


void Record::on_linop_factory_generate_completed(const LinOpFactory *factory,
                                                 const LinOp *input,
                                                 const LinOp *output) const
{
    append_deque(data_.linop_factory_generate_completed,
                 (std::unique_ptr<linop_factory_data>(
                     new linop_factory_data{factory, input, output})));
}


void Record::on_criterion_check_started(
    const stop::Criterion *criterion, const size_type &num_iterations,
    const LinOp *residual, const LinOp *residual_norm, const LinOp *solution,
    const uint8 &stopping_id, const bool &set_finalized) const
{
    append_deque(data_.criterion_check_started,
                 (std::unique_ptr<criterion_data>(new criterion_data{
                     criterion, num_iterations, residual, residual_norm,
                     solution, stopping_id, set_finalized})));
}


void Record::on_criterion_check_completed(
    const stop::Criterion *criterion, const size_type &num_iterations,
    const LinOp *residual, const LinOp *residual_norm, const LinOp *solution,
    const uint8 &stopping_id, const bool &set_finalized,
    const Array<stopping_status> *status, const bool &oneChanged,
    const bool &converged) const
{
    append_deque(
        data_.criterion_check_completed,
        (std::unique_ptr<criterion_data>(new criterion_data{
            criterion, num_iterations, residual, residual_norm, solution,
            stopping_id, set_finalized, status, oneChanged, converged})));
}


void Record::on_iteration_complete(const LinOp *solver,
                                   const size_type &num_iterations,
                                   const LinOp *residual, const LinOp *solution,
                                   const LinOp *residual_norm) const
{
    append_deque(
        data_.iteration_completed,
        (std::unique_ptr<iteration_complete_data>(new iteration_complete_data{
            solver, num_iterations, residual, solution, residual_norm})));
}


}  // namespace log
}  // namespace gko
