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

#include <ginkgo/core/log/gpu_tracer.hpp>


#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include <iostream>
#include "core/log/gpu_tracer_kernels.hpp"


namespace gko {
namespace log {
namespace gpu_tracer {
namespace {


GKO_REGISTER_OPERATION(push, gpu_tracer::push);
GKO_REGISTER_OPERATION(pop, gpu_tracer::pop);


}  // anonymous namespace
}  // namespace gpu_tracer


void GpuTracer::on_allocation_started(const Executor* exec,
                                      const size_type& num_bytes) const
{
    std::stringstream ss;
    ss << "allocation(" << num_bytes << ")";
    exec->run(gpu_tracer::make_push(ss.str()));
}


void GpuTracer::on_allocation_completed(const Executor* exec,
                                        const size_type& num_bytes,
                                        const uintptr& location) const
{
    exec->run(gpu_tracer::make_pop());
}


void GpuTracer::on_free_started(const Executor* exec,
                                const uintptr& location) const
{
    std::string str("free");
    exec->run(gpu_tracer::make_push(str));
}


void GpuTracer::on_free_completed(const Executor* exec,
                                  const uintptr& location) const
{
    exec->run(gpu_tracer::make_pop());
}


void GpuTracer::on_copy_started(const Executor* from, const Executor* to,
                                const uintptr& location_from,
                                const uintptr& location_to,
                                const size_type& num_bytes) const
{
    std::stringstream ss;
    ss << "copy( " << name_demangling::get_dynamic_type(*from) << ", "
       << name_demangling::get_dynamic_type(*to) << " - " << num_bytes << ")";
    if (from == to) {
        from->run(gpu_tracer::make_push(ss.str()));
    }
}


void GpuTracer::on_copy_completed(const Executor* from, const Executor* to,
                                  const uintptr& location_from,
                                  const uintptr& location_to,
                                  const size_type& num_bytes) const
{
    if (from == to) {
        from->run(gpu_tracer::make_pop());
    }
}


void GpuTracer::on_operation_launched(const Executor* exec,
                                      const Operation* operation) const
{
    std::string op_name = operation->get_name();
    if (op_name.find("push") == std::string::npos &&
        op_name.find("pop") == std::string::npos) {
        exec->run(gpu_tracer::make_push(op_name));
    }
}


void GpuTracer::on_operation_completed(const Executor* exec,
                                       const Operation* operation) const
{
    std::string op_name = operation->get_name();
    if (op_name.find("push") == std::string::npos &&
        op_name.find("pop") == std::string::npos) {
        exec->run(gpu_tracer::make_pop());
    }
}


void GpuTracer::on_polymorphic_object_create_started(
    const Executor* exec, const PolymorphicObject* po) const
{
    std::stringstream ss;
    ss << "create( " << name_demangling::get_dynamic_type(*po) << ")";
    exec->run(gpu_tracer::make_push(ss.str()));
}


void GpuTracer::on_polymorphic_object_create_completed(
    const Executor* exec, const PolymorphicObject* input,
    const PolymorphicObject* output) const
{
    exec->run(gpu_tracer::make_pop());
}


void GpuTracer::on_polymorphic_object_copy_started(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    std::stringstream ss;
    ss << "copy(" << name_demangling::get_dynamic_type(*from) << ","
       << name_demangling::get_dynamic_type(*to) << ")";
    exec->run(gpu_tracer::make_push(ss.str()));
}


void GpuTracer::on_polymorphic_object_copy_completed(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    exec->run(gpu_tracer::make_pop());
}


void GpuTracer::on_polymorphic_object_move_started(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    std::stringstream ss;
    ss << "copy(" << name_demangling::get_dynamic_type(*from) << ","
       << name_demangling::get_dynamic_type(*to) << ")";
    exec->run(gpu_tracer::make_push(ss.str()));
}


void GpuTracer::on_polymorphic_object_move_completed(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    exec->run(gpu_tracer::make_pop());
}


void GpuTracer::on_linop_apply_started(const LinOp* A, const LinOp* b,
                                       const LinOp* x) const
{
    std::stringstream ss;
    ss << "apply(" << name_demangling::get_dynamic_type(*A) << " - "
       << b->get_size()[1] << ")";
    A->get_executor()->run(gpu_tracer::make_push(ss.str()));
}


void GpuTracer::on_linop_apply_completed(const LinOp* A, const LinOp* b,
                                         const LinOp* x) const
{
    A->get_executor()->run(gpu_tracer::make_pop());
}


void GpuTracer::on_linop_advanced_apply_started(const LinOp* A,
                                                const LinOp* alpha,
                                                const LinOp* b,
                                                const LinOp* beta,
                                                const LinOp* x) const
{
    std::stringstream ss;
    ss << "advanced_apply(" << name_demangling::get_dynamic_type(*A) << " - "
       << b->get_size()[1] << ")";
    A->get_executor()->run(gpu_tracer::make_push(ss.str()));
}


void GpuTracer::on_linop_advanced_apply_completed(const LinOp* A,
                                                  const LinOp* alpha,
                                                  const LinOp* b,
                                                  const LinOp* beta,
                                                  const LinOp* x) const
{
    A->get_executor()->run(gpu_tracer::make_pop());
}


void GpuTracer::on_linop_factory_generate_started(const LinOpFactory* factory,
                                                  const LinOp* input) const
{
    std::stringstream ss;
    ss << "generate(" << name_demangling::get_dynamic_type(*factory) << ")";
    factory->get_executor()->run(gpu_tracer::make_push(ss.str()));
}


void GpuTracer::on_linop_factory_generate_completed(const LinOpFactory* factory,
                                                    const LinOp* input,
                                                    const LinOp* output) const
{
    factory->get_executor()->run(gpu_tracer::make_pop());
}


}  // namespace log
}  // namespace gko
