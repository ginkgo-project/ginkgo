// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/log/stream.hpp>


#include <iomanip>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


namespace gko {
namespace log {


namespace {


template <typename ValueType = default_precision>
std::ostream& operator<<(std::ostream& os, const matrix::Dense<ValueType>* mtx)
{
    auto exec = mtx->get_executor();
    auto tmp = make_temporary_clone(exec->get_master(), mtx);
    os << "[" << std::endl;
    for (size_type i = 0; i < mtx->get_size()[0]; ++i) {
        for (size_type j = 0; j < mtx->get_size()[1]; ++j) {
            os << '\t' << mtx->at(i, j);
        }
        os << std::endl;
    }
    return os << "]" << std::endl;
}


std::ostream& operator<<(std::ostream& os, const stopping_status* status)
{
    os << "[" << std::endl;
    os << "\tConverged: " << status->has_converged() << std::endl;
    os << "\tStopped: " << status->has_stopped() << " with id "
       << static_cast<int>(status->get_id()) << std::endl;
    os << "\tFinalized: " << status->is_finalized() << std::endl;
    return os << "]" << std::endl;
}


std::string bytes_name(const size_type& num_bytes)
{
    std::ostringstream oss;
    oss << "Bytes[" << num_bytes << "]";
    return oss.str();
}


std::string location_name(const uintptr& location)
{
    std::ostringstream oss;
    oss << "Location[" << std::hex << "0x" << location << "]" << std::dec;
    return oss.str();
}


#define GKO_ENABLE_DEMANGLE_NAME(_object_type)                               \
    std::string demangle_name(const _object_type* object)                    \
    {                                                                        \
        std::ostringstream oss;                                              \
        oss << #_object_type "[";                                            \
        if (object == nullptr) {                                             \
            oss << name_demangling::get_dynamic_type(object);                \
        } else {                                                             \
            oss << name_demangling::get_dynamic_type(*object);               \
        }                                                                    \
        oss << "," << object << "]";                                         \
        return oss.str();                                                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_ENABLE_DEMANGLE_NAME(PolymorphicObject);
GKO_ENABLE_DEMANGLE_NAME(LinOp);
GKO_ENABLE_DEMANGLE_NAME(LinOpFactory);
GKO_ENABLE_DEMANGLE_NAME(stop::Criterion);
GKO_ENABLE_DEMANGLE_NAME(Executor);
GKO_ENABLE_DEMANGLE_NAME(Operation);


}  // namespace


template <typename ValueType>
void Stream<ValueType>::on_allocation_started(const Executor* exec,
                                              const size_type& num_bytes) const
{
    *os_ << prefix_ << "allocation started on " << demangle_name(exec)
         << " with " << bytes_name(num_bytes) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_allocation_completed(const Executor* exec,
                                                const size_type& num_bytes,
                                                const uintptr& location) const
{
    *os_ << prefix_ << "allocation completed on " << demangle_name(exec)
         << " at " << location_name(location) << " with "
         << bytes_name(num_bytes) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_free_started(const Executor* exec,
                                        const uintptr& location) const
{
    *os_ << prefix_ << "free started on " << demangle_name(exec) << " at "
         << location_name(location) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_free_completed(const Executor* exec,
                                          const uintptr& location) const
{
    *os_ << prefix_ << "free completed on " << demangle_name(exec) << " at "
         << location_name(location) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_copy_started(const Executor* from,
                                        const Executor* to,
                                        const uintptr& location_from,
                                        const uintptr& location_to,
                                        const size_type& num_bytes) const
{
    *os_ << prefix_ << "copy started from " << demangle_name(from) << " to "
         << demangle_name(to) << " from " << location_name(location_from)
         << " to " << location_name(location_to) << " with "
         << bytes_name(num_bytes) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_copy_completed(const Executor* from,
                                          const Executor* to,
                                          const uintptr& location_from,
                                          const uintptr& location_to,
                                          const size_type& num_bytes) const
{
    *os_ << prefix_ << "copy completed from " << demangle_name(from) << " to "
         << demangle_name(to) << " from " << location_name(location_from)
         << " to " << location_name(location_to) << " with "
         << bytes_name(num_bytes) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_operation_launched(const Executor* exec,
                                              const Operation* operation) const
{
    *os_ << prefix_ << demangle_name(operation) << " started on "
         << demangle_name(exec) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_operation_completed(const Executor* exec,
                                               const Operation* operation) const
{
    *os_ << prefix_ << demangle_name(operation) << " completed on "
         << demangle_name(exec) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_polymorphic_object_create_started(
    const Executor* exec, const PolymorphicObject* po) const
{
    *os_ << prefix_ << "PolymorphicObject create started from "
         << demangle_name(po) << " on " << demangle_name(exec) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_polymorphic_object_create_completed(
    const Executor* exec, const PolymorphicObject* input,
    const PolymorphicObject* output) const
{
    *os_ << prefix_ << demangle_name(output) << " create completed from "
         << demangle_name(input) << " on " << demangle_name(exec) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_polymorphic_object_copy_started(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    *os_ << prefix_ << demangle_name(from) << " copy started to "
         << demangle_name(to) << " on " << demangle_name(exec) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_polymorphic_object_copy_completed(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    *os_ << prefix_ << demangle_name(from) << " copy completed to "
         << demangle_name(to) << " on " << demangle_name(exec) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_polymorphic_object_move_started(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    *os_ << prefix_ << demangle_name(from) << " move started to "
         << demangle_name(to) << " on " << demangle_name(exec) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_polymorphic_object_move_completed(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    *os_ << prefix_ << demangle_name(from) << " move completed to "
         << demangle_name(to) << " on " << demangle_name(exec) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_polymorphic_object_deleted(
    const Executor* exec, const PolymorphicObject* po) const
{
    *os_ << prefix_ << demangle_name(po) << " deleted on "
         << demangle_name(exec) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_linop_apply_started(const LinOp* A, const LinOp* b,
                                               const LinOp* x) const
{
    *os_ << prefix_ << "apply started on A " << demangle_name(A) << " with b "
         << demangle_name(b) << " and x " << demangle_name(x) << std::endl;
    if (verbose_) {
        *os_ << demangle_name(A) << as<gko::matrix::Dense<ValueType>>(A)
             << std::endl;
        *os_ << demangle_name(b) << as<gko::matrix::Dense<ValueType>>(b)
             << std::endl;
        *os_ << demangle_name(x) << as<gko::matrix::Dense<ValueType>>(x)
             << std::endl;
    }
}


template <typename ValueType>
void Stream<ValueType>::on_linop_apply_completed(const LinOp* A, const LinOp* b,
                                                 const LinOp* x) const
{
    *os_ << prefix_ << "apply completed on A " << demangle_name(A) << " with b "
         << demangle_name(b) << " and x " << demangle_name(x) << std::endl;
    if (verbose_) {
        *os_ << demangle_name(A) << as<gko::matrix::Dense<ValueType>>(A)
             << std::endl;
        *os_ << demangle_name(b) << as<gko::matrix::Dense<ValueType>>(b)
             << std::endl;
        *os_ << demangle_name(x) << as<gko::matrix::Dense<ValueType>>(x)
             << std::endl;
    }
}


template <typename ValueType>
void Stream<ValueType>::on_linop_advanced_apply_started(const LinOp* A,
                                                        const LinOp* alpha,
                                                        const LinOp* b,
                                                        const LinOp* beta,
                                                        const LinOp* x) const
{
    *os_ << prefix_ << "advanced apply started on A " << demangle_name(A)
         << " with alpha " << demangle_name(alpha) << " b " << demangle_name(b)
         << " beta " << demangle_name(beta) << " and x " << demangle_name(x)
         << std::endl;
    if (verbose_) {
        *os_ << demangle_name(A) << as<gko::matrix::Dense<ValueType>>(A)
             << std::endl;
        *os_ << demangle_name(alpha) << as<gko::matrix::Dense<ValueType>>(alpha)
             << std::endl;
        *os_ << demangle_name(b) << as<gko::matrix::Dense<ValueType>>(b)
             << std::endl;
        *os_ << demangle_name(beta) << as<gko::matrix::Dense<ValueType>>(beta)
             << std::endl;
        *os_ << demangle_name(x) << as<gko::matrix::Dense<ValueType>>(x)
             << std::endl;
    }
}


template <typename ValueType>
void Stream<ValueType>::on_linop_advanced_apply_completed(const LinOp* A,
                                                          const LinOp* alpha,
                                                          const LinOp* b,
                                                          const LinOp* beta,
                                                          const LinOp* x) const
{
    *os_ << prefix_ << "advanced apply completed on A " << demangle_name(A)
         << " with alpha " << demangle_name(alpha) << " b " << demangle_name(b)
         << " beta " << demangle_name(beta) << " and x " << demangle_name(x)
         << std::endl;
    if (verbose_) {
        *os_ << demangle_name(A) << as<gko::matrix::Dense<ValueType>>(A)
             << std::endl;
        *os_ << demangle_name(alpha) << as<gko::matrix::Dense<ValueType>>(alpha)
             << std::endl;
        *os_ << demangle_name(b) << as<gko::matrix::Dense<ValueType>>(b)
             << std::endl;
        *os_ << demangle_name(beta) << as<gko::matrix::Dense<ValueType>>(beta)
             << std::endl;
        *os_ << demangle_name(x) << as<gko::matrix::Dense<ValueType>>(x)
             << std::endl;
    }
}


template <typename ValueType>
void Stream<ValueType>::on_linop_factory_generate_started(
    const LinOpFactory* factory, const LinOp* input) const
{
    *os_ << prefix_ << "generate started for " << demangle_name(factory)
         << " with input " << demangle_name(input) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_linop_factory_generate_completed(
    const LinOpFactory* factory, const LinOp* input, const LinOp* output) const
{
    *os_ << prefix_ << "generate completed for " << demangle_name(factory)
         << " with input " << demangle_name(input) << " produced "
         << demangle_name(output) << std::endl;
}


template <typename ValueType>
void Stream<ValueType>::on_criterion_check_started(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const LinOp* residual, const LinOp* residual_norm, const LinOp* solution,
    const uint8& stopping_id, const bool& set_finalized) const
{
    *os_ << prefix_ << "check started for " << demangle_name(criterion)
         << " at iteration " << num_iterations << " with ID "
         << static_cast<int>(stopping_id) << " and finalized set to "
         << set_finalized << std::endl;
    if (verbose_) {
        if (residual != nullptr) {
            *os_ << demangle_name(residual)
                 << as<gko::matrix::Dense<ValueType>>(residual) << std::endl;
        }
        if (residual_norm != nullptr) {
            *os_ << demangle_name(residual_norm)
                 << as<gko::matrix::Dense<ValueType>>(residual_norm)
                 << std::endl;
        }
        if (solution != nullptr) {
            *os_ << demangle_name(solution)
                 << as<gko::matrix::Dense<ValueType>>(solution) << std::endl;
        }
    }
}


template <typename ValueType>
void Stream<ValueType>::on_criterion_check_completed(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const LinOp* residual, const LinOp* residual_norm, const LinOp* solution,
    const uint8& stoppingId, const bool& setFinalized,
    const array<stopping_status>* status, const bool& oneChanged,
    const bool& converged) const
{
    *os_ << prefix_ << "check completed for " << demangle_name(criterion)
         << " at iteration " << num_iterations << " with ID "
         << static_cast<int>(stoppingId) << " and finalized set to "
         << setFinalized << ". It changed one RHS " << oneChanged
         << ", stopped the iteration process " << converged << std::endl;

    if (verbose_) {
        array<stopping_status> tmp(status->get_executor()->get_master(),
                                   *status);
        *os_ << tmp.get_const_data();
        if (residual != nullptr) {
            *os_ << demangle_name(residual)
                 << as<gko::matrix::Dense<ValueType>>(residual) << std::endl;
        }
        if (residual_norm != nullptr) {
            *os_ << demangle_name(residual_norm)
                 << as<gko::matrix::Dense<ValueType>>(residual_norm)
                 << std::endl;
        }
        if (solution != nullptr) {
            *os_ << demangle_name(solution)
                 << as<gko::matrix::Dense<ValueType>>(solution) << std::endl;
        }
    }
}


template <typename ValueType>
void Stream<ValueType>::on_iteration_complete(
    const LinOp* solver, const LinOp* right_hand_side, const LinOp* solution,
    const size_type& num_iterations, const LinOp* residual,
    const LinOp* residual_norm, const LinOp* implicit_resnorm_sq,
    const array<stopping_status>* status, bool stopped) const
{
    *os_ << prefix_ << "iteration " << num_iterations
         << " completed with solver " << demangle_name(solver)
         << " and right-hand-side " << demangle_name(right_hand_side)
         << " with residual " << demangle_name(residual) << ", solution "
         << demangle_name(solution) << ", residual_norm "
         << demangle_name(residual_norm) << " and implicit_sq_residual_norm "
         << demangle_name(implicit_resnorm_sq);
    if (status) {
        *os_ << ". Stopped the iteration process " << std::boolalpha << stopped;
    }
    *os_ << std::endl;

    if (verbose_) {
        *os_ << demangle_name(residual)
             << as<gko::matrix::Dense<ValueType>>(residual) << std::endl;
        *os_ << demangle_name(solution)
             << as<gko::matrix::Dense<ValueType>>(solution) << std::endl;
        if (residual_norm != nullptr) {
            *os_ << demangle_name(residual_norm)
                 << as<gko::matrix::Dense<ValueType>>(residual_norm)
                 << std::endl;
        }
        if (implicit_resnorm_sq != nullptr) {
            *os_ << demangle_name(implicit_resnorm_sq)
                 << as<gko::matrix::Dense<ValueType>>(implicit_resnorm_sq)
                 << std::endl;
        }
        if (status != nullptr) {
            array<stopping_status> tmp(status->get_executor()->get_master(),
                                       *status);
            *os_ << tmp.get_const_data();
        }
        *os_ << demangle_name(right_hand_side)
             << as<gko::matrix::Dense<ValueType>>(right_hand_side) << std::endl;
    }
}


template <typename ValueType>
void Stream<ValueType>::on_iteration_complete(const LinOp* solver,
                                              const size_type& num_iterations,
                                              const LinOp* residual,
                                              const LinOp* solution,
                                              const LinOp* residual_norm) const
{
    on_iteration_complete(solver, nullptr, solution, num_iterations, residual,
                          residual_norm, nullptr, nullptr, false);
}


template <typename ValueType>
void Stream<ValueType>::on_iteration_complete(
    const LinOp* solver, const size_type& num_iterations, const LinOp* residual,
    const LinOp* solution, const LinOp* residual_norm,
    const LinOp* implicit_sq_residual_norm) const
{
    on_iteration_complete(solver, nullptr, solution, num_iterations, residual,
                          residual_norm, implicit_sq_residual_norm, nullptr,
                          false);
}


#define GKO_DECLARE_STREAM(_type) class Stream<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_STREAM);


}  // namespace log
}  // namespace gko
