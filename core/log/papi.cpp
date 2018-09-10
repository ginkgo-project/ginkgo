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


#include <ginkgo/core/log/papi.hpp>


namespace gko {
namespace log {


const papi_handle_t papi_handle = papi_sde_init("ginkgo");


template <typename ValueType>
void Papi<ValueType>::on_allocation_started(const Executor *exec,
                                            const size_type &num_bytes) const
{}


template <typename ValueType>
void Papi<ValueType>::on_allocation_completed(const Executor *exec,
                                              const size_type &num_bytes,
                                              const uintptr &location) const
{}


template <typename ValueType>
void Papi<ValueType>::on_free_started(const Executor *exec,
                                      const uintptr &location) const
{}


template <typename ValueType>
void Papi<ValueType>::on_free_completed(const Executor *exec,
                                        const uintptr &location) const
{}


template <typename ValueType>
void Papi<ValueType>::on_copy_started(const Executor *from, const Executor *to,
                                      const uintptr &location_from,
                                      const uintptr &location_to,
                                      const size_type &num_bytes) const
{}


template <typename ValueType>
void Papi<ValueType>::on_copy_completed(const Executor *from,
                                        const Executor *to,
                                        const uintptr &location_from,
                                        const uintptr &location_to,
                                        const size_type &num_bytes) const
{}


template <typename ValueType>
void Papi<ValueType>::on_operation_launched(const Executor *exec,
                                            const Operation *operation) const
{}


template <typename ValueType>
void Papi<ValueType>::on_operation_completed(const Executor *exec,
                                             const Operation *operation) const
{}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_create_started(
    const Executor *exec, const PolymorphicObject *po) const
{}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_create_completed(
    const Executor *exec, const PolymorphicObject *input,
    const PolymorphicObject *output) const
{}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_copy_started(
    const Executor *exec, const PolymorphicObject *from,
    const PolymorphicObject *to) const
{}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_copy_completed(
    const Executor *exec, const PolymorphicObject *from,
    const PolymorphicObject *to) const
{}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_deleted(
    const Executor *exec, const PolymorphicObject *po) const
{}


template <typename ValueType>
void Papi<ValueType>::on_linop_apply_started(const LinOp *A, const LinOp *b,
                                             const LinOp *x) const
{}


template <typename ValueType>
void Papi<ValueType>::on_linop_apply_completed(const LinOp *A, const LinOp *b,
                                               const LinOp *x) const
{}


template <typename ValueType>
void Papi<ValueType>::on_linop_advanced_apply_started(const LinOp *A,
                                                      const LinOp *alpha,
                                                      const LinOp *b,
                                                      const LinOp *beta,
                                                      const LinOp *x) const
{}


template <typename ValueType>
void Papi<ValueType>::on_linop_advanced_apply_completed(const LinOp *A,
                                                        const LinOp *alpha,
                                                        const LinOp *b,
                                                        const LinOp *beta,
                                                        const LinOp *x) const
{}


template <typename ValueType>
void Papi<ValueType>::on_linop_factory_generate_started(
    const LinOpFactory *factory, const LinOp *input) const
{}


template <typename ValueType>
void Papi<ValueType>::on_linop_factory_generate_completed(
    const LinOpFactory *factory, const LinOp *input, const LinOp *output) const
{}


template <typename ValueType>
void Papi<ValueType>::on_criterion_check_started(
    const stop::Criterion *criterion, const size_type &num_iterations,
    const LinOp *residual, const LinOp *residual_norm, const LinOp *solution,
    const uint8 &stopping_id, const bool &set_finalized) const
{}


template <typename ValueType>
void Papi<ValueType>::on_criterion_check_completed(
    const stop::Criterion *criterion, const size_type &num_iterations,
    const LinOp *residual, const LinOp *residual_norm, const LinOp *solution,
    const uint8 &stoppingId, const bool &setFinalized,
    const Array<stopping_status> *status, const bool &oneChanged,
    const bool &converged) const
{}


template <typename ValueType>
void Papi<ValueType>::on_iteration_complete(const LinOp *solver,
                                            const size_type &num_iterations,
                                            const LinOp *residual,
                                            const LinOp *solution,
                                            const LinOp *residual_norm) const
{}


#define GKO_DECLARE_PAPI(_type) class Papi<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PAPI);
#undef GKO_DECLARE_PAPI


}  // namespace log
}  // namespace gko
