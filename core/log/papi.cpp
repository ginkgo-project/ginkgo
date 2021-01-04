/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/log/papi.hpp>


#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace log {


template <typename ValueType>
void Papi<ValueType>::on_allocation_started(const Executor *exec,
                                            const size_type &num_bytes) const
{
    allocation_started.get_counter(exec) += num_bytes;
}


template <typename ValueType>
void Papi<ValueType>::on_allocation_completed(const Executor *exec,
                                              const size_type &num_bytes,
                                              const uintptr &location) const
{
    allocation_completed.get_counter(exec) += num_bytes;
}


template <typename ValueType>
void Papi<ValueType>::on_free_started(const Executor *exec,
                                      const uintptr &location) const
{
    free_started.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_free_completed(const Executor *exec,
                                        const uintptr &location) const
{
    free_completed.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_copy_started(const Executor *from, const Executor *to,
                                      const uintptr &location_from,
                                      const uintptr &location_to,
                                      const size_type &num_bytes) const
{
    copy_started_from.get_counter(from) += num_bytes;
    copy_started_to.get_counter(to) += num_bytes;
}


template <typename ValueType>
void Papi<ValueType>::on_copy_completed(const Executor *from,
                                        const Executor *to,
                                        const uintptr &location_from,
                                        const uintptr &location_to,
                                        const size_type &num_bytes) const
{
    copy_completed_from.get_counter(from) += num_bytes;
    copy_completed_to.get_counter(to) += num_bytes;
}


template <typename ValueType>
void Papi<ValueType>::on_operation_launched(const Executor *exec,
                                            const Operation *operation) const
{
    operation_launched.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_operation_completed(const Executor *exec,
                                             const Operation *operation) const
{
    operation_completed.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_create_started(
    const Executor *exec, const PolymorphicObject *po) const
{
    polymorphic_object_create_started.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_create_completed(
    const Executor *exec, const PolymorphicObject *input,
    const PolymorphicObject *output) const
{
    polymorphic_object_create_completed.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_copy_started(
    const Executor *exec, const PolymorphicObject *from,
    const PolymorphicObject *to) const
{
    polymorphic_object_copy_started.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_copy_completed(
    const Executor *exec, const PolymorphicObject *from,
    const PolymorphicObject *to) const
{
    polymorphic_object_copy_completed.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_deleted(
    const Executor *exec, const PolymorphicObject *po) const
{
    polymorphic_object_deleted.get_counter(exec) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_linop_apply_started(const LinOp *A, const LinOp *b,
                                             const LinOp *x) const
{
    linop_apply_started.get_counter(A) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_linop_apply_completed(const LinOp *A, const LinOp *b,
                                               const LinOp *x) const
{
    linop_apply_completed.get_counter(A) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_linop_advanced_apply_started(const LinOp *A,
                                                      const LinOp *alpha,
                                                      const LinOp *b,
                                                      const LinOp *beta,
                                                      const LinOp *x) const
{
    linop_advanced_apply_started.get_counter(A) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_linop_advanced_apply_completed(const LinOp *A,
                                                        const LinOp *alpha,
                                                        const LinOp *b,
                                                        const LinOp *beta,
                                                        const LinOp *x) const
{
    linop_advanced_apply_completed.get_counter(A) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_linop_factory_generate_started(
    const LinOpFactory *factory, const LinOp *input) const
{
    linop_factory_generate_started.get_counter(factory) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_linop_factory_generate_completed(
    const LinOpFactory *factory, const LinOp *input, const LinOp *output) const
{
    linop_factory_generate_completed.get_counter(factory) += 1;
}


template <typename ValueType>
void Papi<ValueType>::on_criterion_check_completed(
    const stop::Criterion *criterion, const size_type &num_iterations,
    const LinOp *residual, const LinOp *residual_norm, const LinOp *solution,
    const uint8 &stoppingId, const bool &setFinalized,
    const Array<stopping_status> *status, const bool &oneChanged,
    const bool &converged) const
{
    using Vector = matrix::Dense<ValueType>;
    double residual_norm_d = 0.0;
    if (residual_norm != nullptr) {
        auto dense_r_norm = as<Vector>(residual_norm);
        residual_norm_d =
            static_cast<double>(std::real(dense_r_norm->at(0, 0)));
    } else if (residual != nullptr) {
        auto tmp_res_norm = Vector::create(residual->get_executor(),
                                           dim<2>{1, residual->get_size()[1]});
        auto dense_r = as<Vector>(residual);
        dense_r->compute_norm2(tmp_res_norm.get());
        residual_norm_d =
            static_cast<double>(std::real(tmp_res_norm->at(0, 0)));
    }

    const auto tmp = reinterpret_cast<uintptr>(criterion);
    auto &map = this->criterion_check_completed;
    if (map.find(tmp) == map.end()) {
        map[tmp] = NULL;
    }
    void *handle = map[tmp];
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
void Papi<ValueType>::on_iteration_complete(const LinOp *solver,
                                            const size_type &num_iterations,
                                            const LinOp *residual,
                                            const LinOp *solution,
                                            const LinOp *residual_norm) const
{
    iteration_complete.get_counter(solver) = num_iterations;
}


#define GKO_DECLARE_PAPI(_type) class Papi<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PAPI);


}  // namespace log
}  // namespace gko
