/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/log/distributed_stream.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


#include "core/distributed/helpers.hpp"


namespace gko {
namespace log {


template <typename ValueType>
std::unique_ptr<gko::matrix::Dense<ValueType>> distributed_matrix_to_dense(
    const LinOp* op)
{
    using Dense = gko::matrix::Dense<ValueType>;
    auto exec = op->get_executor();

    std::unique_ptr<Dense> out;

    gko::detail::dispatch_distributed_matrix(op, [&](const auto* concrete_mat) {
        auto mat = make_temporary_clone(exec->get_master(), concrete_mat);

        auto local_mat = mat->get_local_matrix();
        auto non_local_mat = mat->get_non_local_matrix();

        auto dense_local = Dense::create(exec->get_master());
        auto dense_non_local = Dense::create(exec->get_master());

        as<gko::ConvertibleTo<Dense>>(local_mat)->convert_to(dense_local.get());
        as<gko::ConvertibleTo<Dense>>(non_local_mat)
            ->convert_to(dense_non_local.get());

        out = gko::matrix::Dense<ValueType>::create(
            exec->get_master(), gko::dim<2>{local_mat->get_size()[0],
                                            local_mat->get_size()[1] +
                                                non_local_mat->get_size()[1]});

        for (size_type row = 0; row < out->get_size()[0]; ++row) {
            std::copy(&dense_local->at(row, 0),
                      &dense_local->at(row, dense_local->get_size()[1]),
                      &out->at(row, 0));
            std::copy(&dense_non_local->at(row, 0),
                      &dense_non_local->at(row, dense_non_local->get_size()[1]),
                      &out->at(row, dense_local->get_size()[1]));
        }
    });

    return out;
}


template <typename ValueType>
void DistributedStream<ValueType>::on_allocation_started(
    const Executor* exec, const size_type& num_bytes) const
{
    region_wrapper_(
        [&] { local_logger_->on_allocation_started(exec, num_bytes); });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_allocation_completed(
    const Executor* exec, const size_type& num_bytes,
    const uintptr& location) const
{
    region_wrapper_([&] {
        local_logger_->on_allocation_completed(exec, num_bytes, location);
    });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_free_started(
    const Executor* exec, const uintptr& location) const
{
    region_wrapper_([&] { local_logger_->on_free_started(exec, location); });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_free_completed(
    const Executor* exec, const uintptr& location) const
{
    region_wrapper_([&] { local_logger_->on_free_completed(exec, location); });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_copy_started(
    const Executor* from, const Executor* to, const uintptr& location_from,
    const uintptr& location_to, const size_type& num_bytes) const
{
    region_wrapper_([&] {
        local_logger_->on_copy_started(from, to, location_from, location_to,
                                       num_bytes);
    });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_copy_completed(
    const Executor* from, const Executor* to, const uintptr& location_from,
    const uintptr& location_to, const size_type& num_bytes) const
{
    region_wrapper_([&] {
        local_logger_->on_copy_completed(from, to, location_from, location_to,
                                         num_bytes);
    });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_operation_launched(
    const Executor* exec, const Operation* operation) const
{
    region_wrapper_(
        [&] { local_logger_->on_operation_launched(exec, operation); });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_operation_completed(
    const Executor* exec, const Operation* operation) const
{
    region_wrapper_(
        [&] { local_logger_->on_operation_completed(exec, operation); });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_polymorphic_object_create_started(
    const Executor* exec, const PolymorphicObject* po) const
{
    region_wrapper_(
        [&] { local_logger_->on_polymorphic_object_create_started(exec, po); });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_polymorphic_object_create_completed(
    const Executor* exec, const PolymorphicObject* input,
    const PolymorphicObject* output) const
{
    region_wrapper_([&] {
        local_logger_->on_polymorphic_object_create_completed(exec, input,
                                                              output);
    });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_polymorphic_object_copy_started(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    region_wrapper_([&] {
        local_logger_->on_polymorphic_object_copy_started(exec, from, to);
    });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_polymorphic_object_copy_completed(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    region_wrapper_([&] {
        local_logger_->on_polymorphic_object_copy_completed(exec, from, to);
    });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_polymorphic_object_move_started(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    region_wrapper_([&] {
        local_logger_->on_polymorphic_object_move_started(exec, from, to);
    });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_polymorphic_object_move_completed(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    region_wrapper_([&] {
        local_logger_->on_polymorphic_object_move_completed(exec, from, to);
    });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_polymorphic_object_deleted(
    const Executor* exec, const PolymorphicObject* po) const
{
    region_wrapper_(
        [&] { local_logger_->on_polymorphic_object_deleted(exec, po); });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_linop_apply_started(const LinOp* A,
                                                          const LinOp* b,
                                                          const LinOp* x) const
{
    region_wrapper_([&] {
        local_logger_->on_linop_apply_started(
            distributed_matrix_to_dense<ValueType>(A).get(),
            gko::detail::get_local(b), gko::detail::get_local(x));
    });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_linop_apply_completed(
    const LinOp* A, const LinOp* b, const LinOp* x) const
{
    region_wrapper_([&] {
        local_logger_->on_linop_apply_completed(
            distributed_matrix_to_dense<ValueType>(A).get(),
            gko::detail::get_local(b), gko::detail::get_local(x));
    });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_linop_advanced_apply_started(
    const LinOp* A, const LinOp* alpha, const LinOp* b, const LinOp* beta,
    const LinOp* x) const
{
    region_wrapper_([&] {
        local_logger_->on_linop_advanced_apply_started(
            distributed_matrix_to_dense<ValueType>(A).get(), alpha,
            gko::detail::get_local(b), beta, gko::detail::get_local(x));
    });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_linop_advanced_apply_completed(
    const LinOp* A, const LinOp* alpha, const LinOp* b, const LinOp* beta,
    const LinOp* x) const
{
    region_wrapper_([&] {
        local_logger_->on_linop_advanced_apply_completed(
            distributed_matrix_to_dense<ValueType>(A).get(), alpha,
            gko::detail::get_local(b), beta, gko::detail::get_local(x));
    });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_linop_factory_generate_started(
    const LinOpFactory* factory, const LinOp* input) const
{
    region_wrapper_([&] {
        local_logger_->on_linop_factory_generate_started(factory, input);
    });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_linop_factory_generate_completed(
    const LinOpFactory* factory, const LinOp* input, const LinOp* output) const
{
    region_wrapper_([&] {
        local_logger_->on_linop_factory_generate_completed(factory, input,
                                                           output);
    });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_criterion_check_started(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const LinOp* residual, const LinOp* residual_norm, const LinOp* solution,
    const uint8& stopping_id, const bool& set_finalized) const
{
    region_wrapper_([&] {
        local_logger_->on_criterion_check_started(
            criterion, num_iterations, gko::detail::get_local(residual),
            residual_norm, gko::detail::get_local(solution), stopping_id,
            set_finalized);
    });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_criterion_check_completed(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const LinOp* residual, const LinOp* residual_norm, const LinOp* solution,
    const uint8& stoppingId, const bool& setFinalized,
    const array<stopping_status>* status, const bool& oneChanged,
    const bool& converged) const
{
    region_wrapper_([&] {
        local_logger_->on_criterion_check_completed(
            criterion, num_iterations, gko::detail::get_local(residual),
            residual_norm, gko::detail::get_local(solution), stoppingId,
            setFinalized, status, oneChanged, converged);
    });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_iteration_complete(
    const LinOp* solver, const size_type& num_iterations, const LinOp* residual,
    const LinOp* solution, const LinOp* residual_norm) const
{
    region_wrapper_([&] {
        local_logger_->on_iteration_complete(
            solver, num_iterations, gko::detail::get_local(residual),
            gko::detail::get_local(solution), residual_norm);
    });
}


template <typename ValueType>
void DistributedStream<ValueType>::on_iteration_complete(
    const LinOp* solver, const size_type& num_iterations, const LinOp* residual,
    const LinOp* solution, const LinOp* residual_norm,
    const LinOp* implicit_sq_residual_norm) const
{
    region_wrapper_([&] {
        local_logger_->on_iteration_complete(
            solver, num_iterations, gko::detail::get_local(residual),
            gko::detail::get_local(solution), residual_norm,
            implicit_sq_residual_norm);
    });
}


#define GKO_DECLARE_STREAM(_type) class DistributedStream<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_STREAM);


}  // namespace log
}  // namespace gko
