// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/log/profiler_hook.hpp>

#if GKO_HAVE_TAU
#define PERFSTUBS_USE_TIMERS
#include <perfstubs_api/timer.h>
#endif

namespace gko {
namespace log {


#if GKO_HAVE_TAU


void init_tau() { PERFSTUBS_INITIALIZE(); }


void begin_tau(const char* name, profile_event_category)
{
    PERFSTUBS_START_STRING(name);
}


void end_tau(const char* name, profile_event_category)
{
    PERFSTUBS_STOP_STRING(name);
}


void finalize_tau() { PERFSTUBS_FINALIZE(); }


#else


void init_tau() GKO_NOT_COMPILED(tau);


void begin_tau(const char*, profile_event_category) GKO_NOT_COMPILED(tau);


void end_tau(const char*, profile_event_category) GKO_NOT_COMPILED(tau);


void finalize_tau() GKO_NOT_COMPILED(tau);


#endif


}  // namespace log
}  // namespace gko
