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


#include <mutex>
#include <sstream>


#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace log {


using hook_function = std::function<void(const char*, profile_event_category)>;


void init_tau();
void init_nvtx();
void begin_tau(const char*, profile_event_category);
hook_function begin_nvtx_fn(uint32 color_rgb);
void begin_roctx(const char*, profile_event_category);
void end_tau(const char*, profile_event_category);
void end_nvtx(const char*, profile_event_category);
void end_roctx(const char*, profile_event_category);
void finalize_tau();


class ProfilerHook : public Logger {
public:
    void on_allocation_started(const gko::Executor* exec,
                               const gko::size_type&) const override
    {
        this->begin_hook_("allocate", profile_event_category::memory);
    }

    void on_allocation_completed(const gko::Executor* exec,
                                 const gko::size_type&,
                                 const gko::uintptr&) const override
    {
        this->end_hook_("allocate", profile_event_category::memory);
    }

    void on_free_started(const gko::Executor* exec,
                         const gko::uintptr&) const override
    {
        this->begin_hook_("free", profile_event_category::memory);
    }

    void on_free_completed(const gko::Executor* exec,
                           const gko::uintptr&) const override
    {
        this->end_hook_("free", profile_event_category::memory);
    }

    void on_copy_started(const gko::Executor* from, const gko::Executor* to,
                         const gko::uintptr&, const gko::uintptr&,
                         const gko::size_type&) const override
    {
        this->begin_hook_("copy", profile_event_category::operation);
    }

    void on_copy_completed(const gko::Executor* from, const gko::Executor* to,
                           const gko::uintptr&, const gko::uintptr&,
                           const gko::size_type&) const override
    {
        this->end_hook_("copy", profile_event_category::operation);
    }

    /* Operation events */
    void on_operation_launched(const Executor* exec,
                               const Operation* operation) const override
    {
        this->begin_hook_(operation->get_name(),
                          profile_event_category::operation);
    }

    void on_operation_completed(const Executor* exec,
                                const Operation* operation) const override
    {
        this->end_hook_(operation->get_name(),
                        profile_event_category::operation);
    }

    /* PolymorphicObject events */
    void on_polymorphic_object_copy_started(
        const Executor* exec, const PolymorphicObject* from,
        const PolymorphicObject* to) const override
    {
        std::stringstream ss;
        ss << "copy(" << name_demangling::get_dynamic_type(*from) << ","
           << name_demangling::get_dynamic_type(*to) << ")";
        this->begin_hook_(ss.str().c_str(), profile_event_category::object);
    }

    void on_polymorphic_object_copy_completed(
        const Executor* exec, const PolymorphicObject* from,
        const PolymorphicObject* to) const override
    {
        std::stringstream ss;
        ss << "copy(" << name_demangling::get_dynamic_type(*from) << ","
           << name_demangling::get_dynamic_type(*to) << ")";
        this->end_hook_(ss.str().c_str(), profile_event_category::object);
    }

    void on_polymorphic_object_move_started(
        const Executor* exec, const PolymorphicObject* from,
        const PolymorphicObject* to) const override
    {
        std::stringstream ss;
        ss << "move(" << name_demangling::get_dynamic_type(*from) << ","
           << name_demangling::get_dynamic_type(*to) << ")";
        this->begin_hook_(ss.str().c_str(), profile_event_category::object);
    }

    void on_polymorphic_object_move_completed(
        const Executor* exec, const PolymorphicObject* from,
        const PolymorphicObject* to) const override
    {
        std::stringstream ss;
        ss << "move(" << name_demangling::get_dynamic_type(*from) << ","
           << name_demangling::get_dynamic_type(*to) << ")";
        this->end_hook_(ss.str().c_str(), profile_event_category::object);
    }

    /* LinOp events */
    void on_linop_apply_started(const LinOp* A, const LinOp* b,
                                const LinOp* x) const override
    {
        std::stringstream ss;
        ss << "apply(" << name_demangling::get_dynamic_type(*A) << ")";
        this->begin_hook_(ss.str().c_str(), profile_event_category::linop);
    }

    void on_linop_apply_completed(const LinOp* A, const LinOp* b,
                                  const LinOp* x) const override
    {
        std::stringstream ss;
        ss << "apply(" << name_demangling::get_dynamic_type(*A) << ")";
        this->end_hook_(ss.str().c_str(), profile_event_category::linop);
    }

    void on_linop_advanced_apply_started(const LinOp* A, const LinOp* alpha,
                                         const LinOp* b, const LinOp* beta,
                                         const LinOp* x) const override
    {
        std::stringstream ss;
        ss << "advanced_apply(" << name_demangling::get_dynamic_type(*A) << ")";
        this->begin_hook_(ss.str().c_str(), profile_event_category::linop);
    }

    void on_linop_advanced_apply_completed(const LinOp* A, const LinOp* alpha,
                                           const LinOp* b, const LinOp* beta,
                                           const LinOp* x) const override
    {
        std::stringstream ss;
        ss << "advanced_apply(" << name_demangling::get_dynamic_type(*A) << ")";
        this->end_hook_(ss.str().c_str(), profile_event_category::linop);
    }

    /* LinOpFactory events */
    void on_linop_factory_generate_started(const LinOpFactory* factory,
                                           const LinOp* input) const override
    {
        std::stringstream ss;
        ss << "generate(" << name_demangling::get_dynamic_type(*factory) << ")";
        this->begin_hook_(ss.str().c_str(), profile_event_category::factory);
    }

    void on_linop_factory_generate_completed(const LinOpFactory* factory,
                                             const LinOp* input,
                                             const LinOp* output) const override
    {
        std::stringstream ss;
        ss << "generate(" << name_demangling::get_dynamic_type(*factory) << ")";
        this->end_hook_(ss.str().c_str(), profile_event_category::factory);
    }

    /* Criterion events */
    void on_criterion_check_started(const stop::Criterion* criterion,
                                    const size_type& num_iterations,
                                    const LinOp* residual,
                                    const LinOp* residual_norm,
                                    const LinOp* solution,
                                    const uint8& stopping_id,
                                    const bool& set_finalized) const override
    {
        std::stringstream ss;
        ss << "check(" << name_demangling::get_dynamic_type(*criterion) << ")";
        this->begin_hook_(ss.str().c_str(), profile_event_category::criterion);
    }

    void on_criterion_check_completed(
        const stop::Criterion* criterion, const size_type& num_iterations,
        const LinOp* residual, const LinOp* residual_norm,
        const LinOp* solutino, const uint8& stopping_id,
        const bool& set_finalized, const array<stopping_status>* status,
        const bool& one_changed, const bool& all_converged) const override
    {
        std::stringstream ss;
        ss << "check(" << name_demangling::get_dynamic_type(*criterion) << ")";
        this->end_hook_(ss.str().c_str(), profile_event_category::criterion);
    }

    ProfilerHook(hook_function begin, hook_function end)
        : begin_hook_{begin}, end_hook_{end}
    {}

private:
    hook_function begin_hook_;
    hook_function end_hook_;
};


std::mutex profiler_hook_mutex{};


std::shared_ptr<Logger> get_tau_hook(bool initialize)
{
    static std::shared_ptr<Logger> logger;
    std::lock_guard<std::mutex> lock{profiler_hook_mutex};
    if (!logger) {
        inc_global_logger_refcount();
        if (initialize) {
            init_tau();
            logger = std::shared_ptr<ProfilerHook>(
                new ProfilerHook{begin_tau, end_tau}, [](auto ptr) {
                    delete ptr;
                    finalize_tau();
                    dec_global_logger_refcount();
                });
        } else {
            logger = std::shared_ptr<ProfilerHook>(
                new ProfilerHook{begin_tau, end_tau}, [](auto ptr) {
                    delete ptr;
                    dec_global_logger_refcount();
                });
        }
    }
    return logger;
}


std::shared_ptr<Logger> create_nvtx_hook(uint32 color_rgb)
{
    inc_global_logger_refcount();
    init_nvtx();
    return std::shared_ptr<ProfilerHook>(
        new ProfilerHook{begin_nvtx_fn(color_rgb), end_nvtx}, [](auto ptr) {
            delete ptr;
            dec_global_logger_refcount();
        });
}


std::shared_ptr<Logger> create_roctx_hook()
{
    inc_global_logger_refcount();
    return std::shared_ptr<ProfilerHook>(
        new ProfilerHook{begin_roctx, end_roctx}, [](auto ptr) {
            delete ptr;
            dec_global_logger_refcount();
        });
}


}  // namespace log
}  // namespace gko
