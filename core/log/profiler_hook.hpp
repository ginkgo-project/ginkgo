// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_LOG_PROFILER_HOOK_HPP_
#define GKO_CORE_LOG_PROFILER_HOOK_HPP_


#include <ginkgo/core/log/profiler_hook.hpp>


namespace gko {
namespace log {


/** Global initialization of TAU through PerfStubs. */
void init_tau();


/**
 * Global initialization of NVTX event categories
 * (can be called safely multiple times).
 */
void init_nvtx();


/** Starts a TAU profiling range through PerfStubs. */
void begin_tau(const char*, profile_event_category);


/** Returns a function starting an NVTX profiling range with the given color. */
ProfilerHook::hook_function begin_nvtx_fn(uint32 color_rgb);


/**
 * Returns a pair of functions starting and ending a VTune ITT profiling range.
 */
std::pair<ProfilerHook::hook_function, ProfilerHook::hook_function>
create_vtune_fns();


/** Starts a ROCTX profiling range. */
void begin_roctx(const char*, profile_event_category);


/** Ends a TAU profiling range through PerfStubs. */
void end_tau(const char*, profile_event_category);


/** Ends an NVTX profiling range. */
void end_nvtx(const char*, profile_event_category);


/** Ends a ROCTX profiling range. */
void end_roctx(const char*, profile_event_category);


/** Global finalization of TAU through PerfStubs*/
void finalize_tau();


class default_profiling_scope_guard : log::profiling_scope_guard {
public:
    default_profiling_scope_guard(const char* name)
        : log::profiling_scope_guard{name,
                                     log::profile_event_category::internal,
                                     log::begin_tau, log::begin_tau}
    {}
};


}  // namespace log

namespace kernels {
namespace reference {


using profiling_scope_guard = log::default_profiling_scope_guard;


}  // namespace reference


namespace omp {


using profiling_scope_guard = log::default_profiling_scope_guard;


}  // namespace omp


namespace cuda {


class profiling_scope_guard : log::profiling_scope_guard {
public:
    profiling_scope_guard(const char* name)
        : log::profiling_scope_guard{
              name, log::profile_event_category::internal,
              log::begin_nvtx_fn(log::ProfilerHook::color_yellow_argb),
              log::end_nvtx}
    {}
};


}  // namespace cuda


namespace hip {


#if (GINKGO_HIP_PLATFORM_NVCC == 0)
class profiling_scope_guard : log::profiling_scope_guard {
public:
    profiling_scope_guard(const char* name)
        : log::profiling_scope_guard{name,
                                     log::profile_event_category::internal,
                                     log::begin_roctx, log::end_nvtx}
    {}
};
#else
using profiling_scope_guard = log::default_profiling_scope_guard;
#endif


}  // namespace hip


namespace dpcpp {


class profiling_scope_guard {
public:
    profiling_scope_guard(const char* name)
    {
        auto functions = log::create_vtune_fns();
        guard_ = std::make_unique<log::profiling_scope_guard>(
            name, log::profile_event_category::internal,
            std::move(functions.first), std::move(functions.second));
    }

private:
    // TODO17: use std::optional
    std::unique_ptr<log::profiling_scope_guard> guard_;
};


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#define GKO_MACRO_CONCAT(a, b) GKO_MACRO_CONCAT2(a, b)
#define GKO_MACRO_CONCAT2(a, b) a##b
/**
 * @internal
 * This macro can be used inside Ginkgo kernel implementations to annotate a
 * scope for the profiler corresponding to the backend namespace:
 * ```
 * {
 *     GKO_PROFILING_RANGE(costly operation inside a kernel);
 *     // ...
 * }
 * ```
 */
#define GKO_PROFILE_RANGE(_name) \
    profiling_scope_guard GKO_MACRO_CONCAT(guard, __LINE__) { #_name }


#endif  // GKO_CORE_LOG_PROFILER_HOOK_HPP_
