// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/log/papi.hpp>


#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/distributed/helpers.hpp"


namespace gko {
namespace log {


template <typename ValueType>
void Papi<ValueType>::on_allocation_started(const Executor* exec,
                                            const size_type& num_bytes) const
{
    allocation_started.get_counter(exec) += num_bytes;
}


template <typename ValueType>
void Papi<ValueType>::on_allocation_completed(const Executor* exec,
                                              const size_type& num_bytes,
                                              const uintptr& location) const
{
    allocation_completed.get_counter(exec) += num_bytes;
}


template <typename ValueType>
void Papi<ValueType>::on_free_started(const Executor* exec,
                                      const uintptr& location) const
{
    free_started.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_free_completed(const Executor* exec,
                                        const uintptr& location) const
{
    free_completed.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_copy_started(const Executor* from, const Executor* to,
                                      const uintptr& location_from,
                                      const uintptr& location_to,
                                      const size_type& num_bytes) const
{
    copy_started_from.get_counter(from) += num_bytes;
    copy_started_to.get_counter(to) += num_bytes;
}


template <typename ValueType>
void Papi<ValueType>::on_copy_completed(const Executor* from,
                                        const Executor* to,
                                        const uintptr& location_from,
                                        const uintptr& location_to,
                                        const size_type& num_bytes) const
{
    copy_completed_from.get_counter(from) += num_bytes;
    copy_completed_to.get_counter(to) += num_bytes;
}


template <typename ValueType>
void Papi<ValueType>::on_operation_launched(const Executor* exec,
                                            const Operation* operation) const
{
    operation_launched.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_operation_completed(const Executor* exec,
                                             const Operation* operation) const
{
    operation_completed.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_create_started(
    const Executor* exec, const PolymorphicObject* po) const
{
    polymorphic_object_create_started.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_create_completed(
    const Executor* exec, const PolymorphicObject* input,
    const PolymorphicObject* output) const
{
    polymorphic_object_create_completed.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_copy_started(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    polymorphic_object_copy_started.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_copy_completed(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    polymorphic_object_copy_completed.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_move_started(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    polymorphic_object_move_started.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_move_completed(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    polymorphic_object_move_completed.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_deleted(
    const Executor* exec, const PolymorphicObject* po) const
{
    polymorphic_object_deleted.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_linop_apply_started(const LinOp* A, const LinOp* b,
                                             const LinOp* x) const
{
    linop_apply_started.get_counter(A) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_linop_apply_completed(const LinOp* A, const LinOp* b,
                                               const LinOp* x) const
{
    linop_apply_completed.get_counter(A) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_linop_advanced_apply_started(const LinOp* A,
                                                      const LinOp* alpha,
                                                      const LinOp* b,
                                                      const LinOp* beta,
                                                      const LinOp* x) const
{
    linop_advanced_apply_started.get_counter(A) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_linop_advanced_apply_completed(const LinOp* A,
                                                        const LinOp* alpha,
                                                        const LinOp* b,
                                                        const LinOp* beta,
                                                        const LinOp* x) const
{
    linop_advanced_apply_completed.get_counter(A) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_linop_factory_generate_started(
    const LinOpFactory* factory, const LinOp* input) const
{
    linop_factory_generate_started.get_counter(factory) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_linop_factory_generate_completed(
    const LinOpFactory* factory, const LinOp* input, const LinOp* output) const
{
    linop_factory_generate_completed.get_counter(factory) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_criterion_check_completed(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const LinOp* residual, const LinOp* residual_norm, const LinOp* solution,
    const uint8& stoppingId, const bool& setFinalized,
    const array<stopping_status>* status, const bool& oneChanged,
    const bool& converged) const
{
    using Vector = matrix::Dense<ValueType>;
    double residual_norm_d = 0.0;
    if (residual_norm != nullptr) {
        auto dense_r_norm = as<Vector>(residual_norm);
        residual_norm_d =
            static_cast<double>(std::real(dense_r_norm->at(0, 0)));
    } else if (residual != nullptr) {
        detail::vector_dispatch<ValueType>(residual, [&](const auto* dense_r) {
            auto tmp_res_norm = Vector::create(
                residual->get_executor(), dim<2>{1, residual->get_size()[1]});
            dense_r->compute_norm2(tmp_res_norm);
            residual_norm_d =
                static_cast<double>(std::real(tmp_res_norm->at(0, 0)));
        });
    }

    const auto tmp = reinterpret_cast<uintptr>(criterion);
    auto& map = this->criterion_check_completed;
    if (map.find(tmp) == map.end()) {
        map[tmp] = NULL;
    }
    void* handle = map[tmp];
    if (!handle) {
        std::ostringstream oss;
        oss << "criterion_check_completed_" << tmp;
        papi_sde_create_recorder(this->papi_handle, oss.str().c_str(),
                                 sizeof(double), papi_sde_compare_double,
                                 &handle);
    }
    papi_sde_record(handle, sizeof(double), &residual_norm_d);
}


template <typename ValueType>
void Papi<ValueType>::on_iteration_complete(
    const LinOp* solver, const LinOp* b, const LinOp* solution,
    const size_type& num_iterations, const LinOp* residual,
    const LinOp* residual_norm, const LinOp* implicit_resnorm_sq,
    const array<stopping_status>* status, bool stopped) const
{
    iteration_complete.get_counter(solver) = num_iterations;
}


template <typename ValueType>
void Papi<ValueType>::on_iteration_complete(const LinOp* solver,
                                            const size_type& num_iterations,
                                            const LinOp* residual,
                                            const LinOp* solution,
                                            const LinOp* residual_norm) const
{
    this->on_iteration_complete(solver, nullptr, solution, num_iterations,
                                residual, residual_norm, nullptr, nullptr,
                                false);
}


template <typename ValueType>
void Papi<ValueType>::on_iteration_complete(
    const LinOp* solver, const size_type& num_iterations, const LinOp* residual,
    const LinOp* solution, const LinOp* residual_norm,
    const LinOp* implicit_sq_residual_norm) const
{
    this->on_iteration_complete(solver, nullptr, solution, num_iterations,
                                residual, residual_norm,
                                implicit_sq_residual_norm, nullptr, false);
}


#define GKO_DECLARE_PAPI(_type) class Papi<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PAPI);


}  // namespace log
}  // namespace gko
