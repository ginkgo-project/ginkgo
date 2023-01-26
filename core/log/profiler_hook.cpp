/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/log/profiler_hook.hpp>


#include <memory>
#include <mutex>
#include <sstream>


#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/stop/criterion.hpp>


#include "core/log/profiler_hook.hpp"


namespace gko {
namespace log {


void ProfilerHook::on_allocation_started(const gko::Executor* exec,
                                         const gko::size_type&) const
{
    this->maybe_synchronize(exec);
    this->begin_hook_("allocate", profile_event_category::memory);
}


void ProfilerHook::on_allocation_completed(const gko::Executor* exec,
                                           const gko::size_type&,
                                           const gko::uintptr&) const
{
    this->maybe_synchronize(exec);
    this->end_hook_("allocate", profile_event_category::memory);
}


void ProfilerHook::on_free_started(const gko::Executor* exec,
                                   const gko::uintptr&) const
{
    this->maybe_synchronize(exec);
    this->begin_hook_("free", profile_event_category::memory);
}


void ProfilerHook::on_free_completed(const gko::Executor* exec,
                                     const gko::uintptr&) const
{
    this->maybe_synchronize(exec);
    this->end_hook_("free", profile_event_category::memory);
}


void ProfilerHook::on_copy_started(const gko::Executor* from,
                                   const gko::Executor* to, const gko::uintptr&,
                                   const gko::uintptr&,
                                   const gko::size_type&) const
{
    this->maybe_synchronize(from);
    this->maybe_synchronize(to);
    this->begin_hook_("copy", profile_event_category::operation);
}


void ProfilerHook::on_copy_completed(const gko::Executor* from,
                                     const gko::Executor* to,
                                     const gko::uintptr&, const gko::uintptr&,
                                     const gko::size_type&) const
{
    this->maybe_synchronize(from);
    this->maybe_synchronize(to);
    this->end_hook_("copy", profile_event_category::operation);
}


void ProfilerHook::on_operation_launched(const Executor* exec,
                                         const Operation* operation) const
{
    this->maybe_synchronize(exec);
    this->begin_hook_(operation->get_name(), profile_event_category::operation);
}


void ProfilerHook::on_operation_completed(const Executor* exec,
                                          const Operation* operation) const
{
    this->maybe_synchronize(exec);
    this->end_hook_(operation->get_name(), profile_event_category::operation);
}


void ProfilerHook::on_polymorphic_object_copy_started(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    std::stringstream ss;
    ss << "copy(" << stringify_object(from) << "," << stringify_object(to)
       << ")";
    this->begin_hook_(ss.str().c_str(), profile_event_category::object);
}


void ProfilerHook::on_polymorphic_object_copy_completed(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    std::stringstream ss;
    ss << "copy(" << stringify_object(from) << "," << stringify_object(to)
       << ")";
    this->end_hook_(ss.str().c_str(), profile_event_category::object);
}


void ProfilerHook::on_polymorphic_object_move_started(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    std::stringstream ss;
    ss << "move(" << stringify_object(from) << "," << stringify_object(to)
       << ")";
    this->begin_hook_(ss.str().c_str(), profile_event_category::object);
}


void ProfilerHook::on_polymorphic_object_move_completed(
    const Executor* exec, const PolymorphicObject* from,
    const PolymorphicObject* to) const
{
    std::stringstream ss;
    ss << "move(" << stringify_object(from) << "," << stringify_object(to)
       << ")";
    this->end_hook_(ss.str().c_str(), profile_event_category::object);
}


void ProfilerHook::on_linop_apply_started(const LinOp* A, const LinOp* b,
                                          const LinOp* x) const
{
    std::stringstream ss;
    ss << "apply(" << stringify_object(A) << ")";
    this->begin_hook_(ss.str().c_str(), profile_event_category::linop);
}


void ProfilerHook::on_linop_apply_completed(const LinOp* A, const LinOp* b,
                                            const LinOp* x) const
{
    std::stringstream ss;
    ss << "apply(" << stringify_object(A) << ")";
    this->end_hook_(ss.str().c_str(), profile_event_category::linop);
}


void ProfilerHook::on_linop_advanced_apply_started(const LinOp* A,
                                                   const LinOp* alpha,
                                                   const LinOp* b,
                                                   const LinOp* beta,
                                                   const LinOp* x) const
{
    std::stringstream ss;
    ss << "advanced_apply(" << stringify_object(A) << ")";
    this->begin_hook_(ss.str().c_str(), profile_event_category::linop);
}


void ProfilerHook::on_linop_advanced_apply_completed(const LinOp* A,
                                                     const LinOp* alpha,
                                                     const LinOp* b,
                                                     const LinOp* beta,
                                                     const LinOp* x) const
{
    std::stringstream ss;
    ss << "advanced_apply(" << stringify_object(A) << ")";
    this->end_hook_(ss.str().c_str(), profile_event_category::linop);
}


void ProfilerHook::on_linop_factory_generate_started(
    const LinOpFactory* factory, const LinOp* input) const
{
    std::stringstream ss;
    ss << "generate(" << stringify_object(factory) << ")";
    this->begin_hook_(ss.str().c_str(), profile_event_category::factory);
}


void ProfilerHook::on_linop_factory_generate_completed(
    const LinOpFactory* factory, const LinOp* input, const LinOp* output) const
{
    std::stringstream ss;
    ss << "generate(" << stringify_object(factory) << ")";
    this->end_hook_(ss.str().c_str(), profile_event_category::factory);
}


void ProfilerHook::on_criterion_check_started(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const LinOp* residual, const LinOp* residual_norm, const LinOp* solution,
    const uint8& stopping_id, const bool& set_finalized) const
{
    std::stringstream ss;
    ss << "check(" << stringify_object(criterion) << ")";
    this->begin_hook_(ss.str().c_str(), profile_event_category::criterion);
}


void ProfilerHook::on_criterion_check_completed(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const LinOp* residual, const LinOp* residual_norm, const LinOp* solution,
    const uint8& stopping_id, const bool& set_finalized,
    const array<stopping_status>* status, const bool& one_changed,
    const bool& all_converged) const
{
    this->on_criterion_check_completed(
        criterion, num_iterations, residual, residual_norm, nullptr, solution,
        stopping_id, set_finalized, status, one_changed, all_converged);
}

void ProfilerHook::on_criterion_check_completed(
    const stop::Criterion* criterion, const size_type& num_iterations,
    const LinOp* residual, const LinOp* residual_norm,
    const LinOp* implicit_sq_resnorm, const LinOp* solution,
    const uint8& stopping_id, const bool& set_finalized,
    const array<stopping_status>* status, const bool& one_changed,
    const bool& all_stopped) const
{
    std::stringstream ss;
    ss << "check(" << stringify_object(criterion) << ")";
    this->end_hook_(ss.str().c_str(), profile_event_category::criterion);
}


bool ProfilerHook::needs_propagation() const { return true; }


void ProfilerHook::set_object_name(const PolymorphicObject* obj,
                                   std::string name)
{
    name_map_[obj] = name;
}


void ProfilerHook::set_synchronization(bool synchronize)
{
    synchronize_ = synchronize;
}


void ProfilerHook::maybe_synchronize(const Executor* exec) const
{
    if (synchronize_) {
        exec->synchronize();
    }
}


std::string ProfilerHook::stringify_object(const PolymorphicObject* obj) const
{
    if (!obj) {
        return "nullptr";
    }
    auto it = name_map_.find(obj);
    if (it != name_map_.end()) {
        return it->second;
    }
    return name_demangling::get_dynamic_type(*obj);
}


ProfilerHook::ProfilerHook(hook_function begin, hook_function end)
    : synchronize_{false}, begin_hook_{begin}, end_hook_{end}
{}


struct tau_finalize_deleter {
    void operator()(int* ptr)
    {
        finalize_tau();
        delete ptr;
    }
};


std::unique_ptr<ProfilerHook> ProfilerHook::create_tau(bool initialize)
{
    static std::mutex tau_mutex{};
    static std::unique_ptr<int, tau_finalize_deleter>
        tau_finalize_scope_guard{};
    if (initialize) {
        std::lock_guard<std::mutex> guard{tau_mutex};
        if (!tau_finalize_scope_guard) {
            init_tau();
            tau_finalize_scope_guard =
                std::unique_ptr<int, tau_finalize_deleter>{
                    new int, tau_finalize_deleter{}};
        }
    }
    return std::unique_ptr<ProfilerHook>{new ProfilerHook{begin_tau, end_tau}};
}


std::unique_ptr<ProfilerHook> ProfilerHook::create_nvtx(uint32 color_rgb)
{
    init_nvtx();
    return std::unique_ptr<ProfilerHook>{
        new ProfilerHook{begin_nvtx_fn(color_rgb), end_nvtx}};
}


std::unique_ptr<ProfilerHook> ProfilerHook::create_roctx()
{
    return std::unique_ptr<ProfilerHook>{
        new ProfilerHook{begin_roctx, end_roctx}};
}


std::unique_ptr<ProfilerHook> ProfilerHook::create_for_executor(
    std::shared_ptr<const Executor> exec)
{
    if (std::dynamic_pointer_cast<const CudaExecutor>(exec)) {
        return create_nvtx();
    }
#if (GINKGO_HIP_PLATFORM_NVCC == 0)
    if (std::dynamic_pointer_cast<const HipExecutor>(exec)) {
        return create_roctx();
    }
#endif
    return create_tau();
}


std::unique_ptr<ProfilerHook> ProfilerHook::create_custom(hook_function begin,
                                                          hook_function end)
{
    return std::unique_ptr<ProfilerHook>{new ProfilerHook{begin, end}};
}


}  // namespace log
}  // namespace gko
