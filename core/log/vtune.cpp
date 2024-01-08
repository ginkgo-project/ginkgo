// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/log/profiler_hook.hpp>


#if GKO_HAVE_VTUNE
#include <ittnotify.h>
#endif


namespace gko {
namespace log {


#if GKO_HAVE_VTUNE


std::pair<ProfilerHook::hook_function, ProfilerHook::hook_function>
create_vtune_fns()
{
    auto domain = __itt_domain_create("Ginkgo");
    return std::make_pair(
        [domain](const char* name, profile_event_category) {
            auto handle = __itt_string_handle_create(name);
            __itt_task_begin(domain, __itt_null, __itt_null, handle);
        },
        [domain](const char*, profile_event_category) {
            __itt_task_end(domain);
        });
}


#else


std::pair<ProfilerHook::hook_function, ProfilerHook::hook_function>
create_vtune_fns() GKO_NOT_COMPILED(vtune);


#endif


}  // namespace log
}  // namespace gko
