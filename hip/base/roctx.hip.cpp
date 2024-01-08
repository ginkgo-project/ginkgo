// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <hip/hip_runtime.h>


#include <ginkgo/config.hpp>


#if GINKGO_HIP_PLATFORM_HCC && GKO_HAVE_ROCTX
#if HIP_VERSION >= 50200000
#include <roctracer/roctx.h>
#else
#include <roctx.h>
#endif
#endif


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/log/profiler_hook.hpp>


namespace gko {
namespace log {


#if GINKGO_HIP_PLATFORM_HCC && GKO_HAVE_ROCTX

void begin_roctx(const char* name, profile_event_category)
{
    roctxRangePush(name);
}


void end_roctx(const char*, profile_event_category) { roctxRangePop(); }

#else

void begin_roctx(const char* name, profile_event_category)
    GKO_NOT_COMPILED(roctx);


void end_roctx(const char*, profile_event_category) GKO_NOT_COMPILED(roctx);

#endif


}  // namespace log
}  // namespace gko
