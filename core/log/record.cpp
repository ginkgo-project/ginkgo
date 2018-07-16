/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/


#include "core/log/record.hpp"


#include "core/base/array.hpp"
#include "core/stop/criterion.hpp"
#include "core/stop/stopping_status.hpp"


namespace gko {
namespace log {


#define GKO_APPEND_DEQUE(deque_name_, object_)                \
    if (max_storage_ && deque_name_.size() == max_storage_) { \
        deque_name_.pop_front();                              \
    }                                                         \
    deque_name_.push_back(object_)


void Record::on_iteration_complete(const LinOp *solver,
                                   const size_type &num_iterations,
                                   const LinOp *residual, const LinOp *solution,
                                   const LinOp *residual_norm) const
{
    /* TODO: remove this part */
    data_.num_iterations = num_iterations;

    GKO_APPEND_DEQUE(data_.iteration_completed,
                     (iteration_complete_data{solver, num_iterations, residual,
                                              solution, residual_norm}));
}


void Record::on_apply(const std::string &name) const
{
    GKO_APPEND_DEQUE(data_.applies, name);
}


/* TODO: improve this whenever the criterion class hierarchy MR is merged */
void Record::on_converged(const size_type &at_iteration,
                          const LinOp *residual) const
{
    data_.converged_at_iteration = at_iteration;
    if (residual != nullptr) {
        GKO_APPEND_DEQUE(data_.residuals, residual->clone());
    }
}


void Record::on_allocation_started(const Executor *exec,
                                   const size_type &num_bytes) const
{
    GKO_APPEND_DEQUE(data_.allocation_started,
                     (executor_data{exec, num_bytes, 0}));
}


void Record::on_allocation_completed(const Executor *exec,
                                     const size_type &num_bytes,
                                     const uintptr &location) const
{
    GKO_APPEND_DEQUE(data_.allocation_completed,
                     (executor_data{exec, num_bytes, location}));
}


void Record::on_free_started(const Executor *exec,
                             const uintptr &location) const
{
    GKO_APPEND_DEQUE(data_.free_started, (executor_data{exec, 0, location}));
}


void Record::on_free_completed(const Executor *exec,
                               const uintptr &location) const
{
    GKO_APPEND_DEQUE(data_.free_completed, (executor_data{exec, 0, location}));
}


void Record::on_copy_started(const Executor *from, const Executor *to,
                             const uintptr &location_from,
                             const uintptr &location_to,
                             const size_type &num_bytes) const
{
    GKO_APPEND_DEQUE(
        data_.copy_started,
        (std::tuple<executor_data, executor_data>{
            {from, num_bytes, location_from}, {to, num_bytes, location_to}}));
}


void Record::on_copy_completed(const Executor *from, const Executor *to,
                               const uintptr &location_from,
                               const uintptr &location_to,
                               const size_type &num_bytes) const
{
    GKO_APPEND_DEQUE(
        data_.copy_completed,
        (std::tuple<executor_data, executor_data>{
            {from, num_bytes, location_from}, {to, num_bytes, location_to}}));
}


void Record::on_operation_launched(const Executor *exec,
                                   const Operation *operation) const
{
    GKO_APPEND_DEQUE(data_.operation_launched,
                     (operation_data{exec, operation}));
}


void Record::on_operation_completed(const Executor *exec,
                                    const Operation *operation) const
{
    GKO_APPEND_DEQUE(data_.operation_completed,
                     (operation_data{exec, operation}));
}


void Record::on_polymorphic_object_create_started(
    const Executor *exec, const PolymorphicObject *po) const
{
    GKO_APPEND_DEQUE(data_.polymorphic_object_create_started,
                     (polymorphic_object_data{exec, po}));
}


void Record::on_polymorphic_object_create_completed(
    const Executor *exec, const PolymorphicObject *input,
    const PolymorphicObject *output) const
{
    GKO_APPEND_DEQUE(data_.polymorphic_object_create_completed,
                     (polymorphic_object_data{exec, input, output}));
}


void Record::on_polymorphic_object_copy_started(
    const Executor *exec, const PolymorphicObject *from,
    const PolymorphicObject *to) const
{
    GKO_APPEND_DEQUE(data_.polymorphic_object_copy_started,
                     (polymorphic_object_data{exec, from, to}));
}


void Record::on_polymorphic_object_copy_completed(
    const Executor *exec, const PolymorphicObject *from,
    const PolymorphicObject *to) const
{
    GKO_APPEND_DEQUE(data_.polymorphic_object_copy_completed,
                     (polymorphic_object_data{exec, from, to}));
}


void Record::on_polymorphic_object_deleted(const Executor *exec,
                                           const PolymorphicObject *po) const
{
    GKO_APPEND_DEQUE(data_.polymorphic_object_deleted,
                     (polymorphic_object_data{exec, po}));
}


void Record::on_linop_apply_started(const LinOp *A, const LinOp *b,
                                    const LinOp *x) const
{
    GKO_APPEND_DEQUE(data_.linop_apply_started,
                     (linop_data{A, nullptr, b, nullptr, x}));
}


void Record::on_linop_apply_completed(const LinOp *A, const LinOp *b,
                                      const LinOp *x) const
{
    GKO_APPEND_DEQUE(data_.linop_apply_completed,
                     (linop_data{A, nullptr, b, nullptr, x}));
}


void Record::on_linop_advanced_apply_started(const LinOp *A, const LinOp *alpha,
                                             const LinOp *b, const LinOp *beta,
                                             const LinOp *x) const
{
    GKO_APPEND_DEQUE(data_.linop_advanced_apply_started,
                     (linop_data{A, alpha, b, beta, x}));
}


void Record::on_linop_advanced_apply_completed(const LinOp *A,
                                               const LinOp *alpha,
                                               const LinOp *b,
                                               const LinOp *beta,
                                               const LinOp *x) const
{
    GKO_APPEND_DEQUE(data_.linop_advanced_apply_completed,
                     (linop_data{A, alpha, b, beta, x}));
}


void Record::on_linop_factory_generate_started(const LinOpFactory *factory,
                                               const LinOp *input) const
{
    GKO_APPEND_DEQUE(data_.linop_factory_generate_started,
                     (linop_factory_data{factory, input, nullptr}));
}


void Record::on_linop_factory_generate_completed(const LinOpFactory *factory,
                                                 const LinOp *input,
                                                 const LinOp *output) const
{
    GKO_APPEND_DEQUE(data_.linop_factory_generate_completed,
                     (linop_factory_data{factory, input, output}));
}


void Record::on_criterion_check_started(const stop::Criterion *criterion,
                                        const uint8 &stoppingId,
                                        const bool &setFinalized) const
{
    GKO_APPEND_DEQUE(data_.criterion_check_started,
                     (criterion_data{criterion, stoppingId, setFinalized}));
}


void Record::on_criterion_check_completed(const stop::Criterion *criterion,
                                          const uint8 &stoppingId,
                                          const bool &setFinalized,
                                          const Array<stopping_status> *status,
                                          const bool &oneChanged,
                                          const bool &converged) const
{
    GKO_APPEND_DEQUE(data_.criterion_check_completed,
                     (criterion_data{criterion, stoppingId, setFinalized,
                                     status, oneChanged, converged}));
}


}  // namespace log
}  // namespace gko
