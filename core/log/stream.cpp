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


#include "core/log/stream.hpp"


#include <iomanip>


#include "core/base/array.hpp"
#include "core/base/lin_op.hpp"
#include "core/matrix/dense.hpp"
#include "core/stop/criterion.hpp"
#include "core/stop/stopping_status.hpp"


namespace gko {
namespace log {


namespace {


template <typename ValueType = default_precision>
std::ostream &operator<<(std::ostream &os, const matrix::Dense<ValueType> *mtx)
{
    auto exec = mtx->get_executor();
    auto tmp = make_temporary_clone(exec->get_master(), mtx);
    os << "[" << std::endl;
    for (int i = 0; i < mtx->get_size()[0]; ++i) {
        for (int j = 0; j < mtx->get_size()[1]; ++j) {
            os << '\t' << mtx->at(i, j);
        }
        os << std::endl;
    }
    return os << "]" << std::endl;
}


std::ostream &operator<<(std::ostream &os, const stopping_status *status)
{
    os << "[" << std::endl;
    os << "\tConverged: " << status->has_converged() << std::endl;
    os << "\tStopped: " << status->has_stopped() << " with id "
       << static_cast<int>(status->get_id()) << std::endl;
    os << "\tFinalized: " << status->is_finalized() << std::endl;
    return os << "]" << std::endl;
}


}  // namespace


template <typename ValueType>
void Stream<ValueType>::on_allocation_started(const Executor *exec,
                                              const size_type &num_bytes) const
{
    os_ << prefix_ << "allocation started on " << executor_name(exec)
        << " with " << bytes_name(num_bytes) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_allocation_completed(const Executor *exec,
                                                const size_type &num_bytes,
                                                const uintptr &location) const
{
    os_ << prefix_ << "allocation completed on " << executor_name(exec)
        << " at " << location_name(location) << " with "
        << bytes_name(num_bytes) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_free_started(const Executor *exec,
                                        const uintptr &location) const
{
    os_ << prefix_ << "free started on " << executor_name(exec) << " at "
        << location_name(location) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_free_completed(const Executor *exec,
                                          const uintptr &location) const
{
    os_ << prefix_ << "free completed on " << executor_name(exec) << " at "
        << location_name(location) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_copy_started(const Executor *from,
                                        const Executor *to,
                                        const uintptr &location_from,
                                        const uintptr &location_to,
                                        const size_type &num_bytes) const
{
    os_ << prefix_ << "copy started from " << executor_name(from) << " to "
        << executor_name(to) << " from " << location_name(location_from)
        << " to " << location_name(location_to) << " with "
        << bytes_name(num_bytes) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_copy_completed(const Executor *from,
                                          const Executor *to,
                                          const uintptr &location_from,
                                          const uintptr &location_to,
                                          const size_type &num_bytes) const
{
    os_ << prefix_ << "copy completed from " << executor_name(from) << " to "
        << executor_name(to) << " from " << location_name(location_from)
        << " to " << location_name(location_to) << " with "
        << bytes_name(num_bytes) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_operation_launched(const Executor *exec,
                                              const Operation *operation) const
{
    os_ << prefix_ << operation_name(operation) << " started on "
        << executor_name(exec) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_operation_completed(const Executor *exec,
                                               const Operation *operation) const
{
    os_ << prefix_ << operation_name(operation) << " completed on "
        << executor_name(exec) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_polymorphic_object_create_started(
    const Executor *exec, const PolymorphicObject *po) const
{
    os_ << prefix_ << "PolymorphicObject create started from " << po_name(po)
        << " on " << executor_name(exec) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_polymorphic_object_create_completed(
    const Executor *exec, const PolymorphicObject *input,
    const PolymorphicObject *output) const
{
    os_ << prefix_ << po_name(output) << " create completed from "
        << po_name(input) << " on " << executor_name(exec) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_polymorphic_object_copy_started(
    const Executor *exec, const PolymorphicObject *from,
    const PolymorphicObject *to) const
{
    os_ << prefix_ << po_name(from) << " copy started to " << po_name(to)
        << " on " << executor_name(exec) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_polymorphic_object_copy_completed(
    const Executor *exec, const PolymorphicObject *from,
    const PolymorphicObject *to) const
{
    os_ << prefix_ << po_name(from) << " copy completed to " << po_name(to)
        << " on " << executor_name(exec) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_polymorphic_object_deleted(
    const Executor *exec, const PolymorphicObject *po) const
{
    os_ << prefix_ << po_name(po) << " deleted on " << executor_name(exec)
        << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_linop_apply_started(const LinOp *A, const LinOp *b,
                                               const LinOp *x) const
{
    os_ << prefix_ << "apply started on A " << linop_name(A) << " with b "
        << linop_name(b) << " and x " << linop_name(x) << std::endl;
    if (verbose_) {
        os_ << linop_name(A) << as<gko::matrix::Dense<ValueType>>(A)
            << std::endl;
        os_ << linop_name(b) << as<gko::matrix::Dense<ValueType>>(b)
            << std::endl;
        os_ << linop_name(x) << as<gko::matrix::Dense<ValueType>>(x)
            << std::endl;
    }
}


template <typename ValueType>
void Stream<ValueType>::on_linop_apply_completed(const LinOp *A, const LinOp *b,
                                                 const LinOp *x) const
{
    os_ << prefix_ << "apply completed on A " << linop_name(A) << " with b "
        << linop_name(b) << " and x " << linop_name(x) << std::endl;
    if (verbose_) {
        os_ << linop_name(A) << as<gko::matrix::Dense<ValueType>>(A)
            << std::endl;
        os_ << linop_name(b) << as<gko::matrix::Dense<ValueType>>(b)
            << std::endl;
        os_ << linop_name(x) << as<gko::matrix::Dense<ValueType>>(x)
            << std::endl;
    }
}


template <typename ValueType>
void Stream<ValueType>::on_linop_advanced_apply_started(const LinOp *A,
                                                        const LinOp *alpha,
                                                        const LinOp *b,
                                                        const LinOp *beta,
                                                        const LinOp *x) const
{
    os_ << prefix_ << "advanced apply started on A " << linop_name(A)
        << " with alpha " << linop_name(alpha) << " b " << linop_name(b)
        << " beta " << linop_name(beta) << " and x " << linop_name(x)
        << std::endl;
    if (verbose_) {
        os_ << linop_name(A) << as<gko::matrix::Dense<ValueType>>(A)
            << std::endl;
        os_ << linop_name(alpha) << as<gko::matrix::Dense<ValueType>>(alpha)
            << std::endl;
        os_ << linop_name(b) << as<gko::matrix::Dense<ValueType>>(b)
            << std::endl;
        os_ << linop_name(beta) << as<gko::matrix::Dense<ValueType>>(beta)
            << std::endl;
        os_ << linop_name(x) << as<gko::matrix::Dense<ValueType>>(x)
            << std::endl;
    }
}


template <typename ValueType>
void Stream<ValueType>::on_linop_advanced_apply_completed(const LinOp *A,
                                                          const LinOp *alpha,
                                                          const LinOp *b,
                                                          const LinOp *beta,
                                                          const LinOp *x) const
{
    os_ << prefix_ << "advanced apply completed on A " << linop_name(A)
        << " with alpha " << linop_name(alpha) << " b " << linop_name(b)
        << " beta " << linop_name(beta) << " and x " << linop_name(x)
        << std::endl;
    if (verbose_) {
        os_ << linop_name(A) << as<gko::matrix::Dense<ValueType>>(A)
            << std::endl;
        os_ << linop_name(alpha) << as<gko::matrix::Dense<ValueType>>(alpha)
            << std::endl;
        os_ << linop_name(b) << as<gko::matrix::Dense<ValueType>>(b)
            << std::endl;
        os_ << linop_name(beta) << as<gko::matrix::Dense<ValueType>>(beta)
            << std::endl;
        os_ << linop_name(x) << as<gko::matrix::Dense<ValueType>>(x)
            << std::endl;
    }
}


template <typename ValueType>
void Stream<ValueType>::on_linop_factory_generate_started(
    const LinOpFactory *factory, const LinOp *input) const
{
    os_ << prefix_ << "generate started for " << linop_factory_name(factory)
        << " with input " << linop_name(input) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_linop_factory_generate_completed(
    const LinOpFactory *factory, const LinOp *input, const LinOp *output) const
{
    os_ << prefix_ << "generate completed for " << linop_factory_name(factory)
        << " with input " << linop_name(input) << " produced "
        << linop_name(output) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_criterion_check_started(
    const stop::Criterion *criterion, const uint8 &stoppingId,
    const bool &setFinalized) const
{
    os_ << prefix_ << "check started for " << criterion_name(criterion)
        << " with ID " << static_cast<int>(stoppingId)
        << " and finalized set to " << setFinalized << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_criterion_check_completed(
    const stop::Criterion *criterion, const uint8 &stoppingId,
    const bool &setFinalized, const Array<stopping_status> *status,
    const bool &oneChanged, const bool &converged) const
{
    os_ << prefix_ << "check completed for " << criterion_name(criterion)
        << " with ID " << static_cast<int>(stoppingId)
        << " and finalized set to " << setFinalized << ". It changed one RHS "
        << oneChanged << ", stopped the iteration process " << converged
        << std::endl;

    if (verbose_) {
        Array<stopping_status> tmp(status->get_executor()->get_master(),
                                   *status);
        os_ << tmp.get_const_data();
    }
}


template <typename ValueType>
void Stream<ValueType>::on_iteration_complete(const LinOp *solver,
                                              const size_type &num_iterations,
                                              const LinOp *residual,
                                              const LinOp *solution,
                                              const LinOp *residual_norm) const
{
    os_ << prefix_ << "iteration " << num_iterations
        << " completed with solver " << linop_name(solver) << " with residual "
        << linop_name(residual) << ", solution " << linop_name(solution)
        << " and residual_norm " << linop_name(residual_norm) << std::endl;
    if (verbose_) {
        os_ << linop_name(residual)
            << as<gko::matrix::Dense<ValueType>>(residual) << std::endl;
        if (solution != nullptr) {
            os_ << linop_name(solution)
                << as<gko::matrix::Dense<ValueType>>(solution) << std::endl;
        }
        if (residual_norm != nullptr) {
            os_ << linop_name(residual_norm)
                << as<gko::matrix::Dense<ValueType>>(residual_norm)
                << std::endl;
        }
    }
}


#define GKO_DECLARE_STREAM(_type) class Stream<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_STREAM);
#undef GKO_DECLARE_STREAM


}  // namespace log
}  // namespace gko
