// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>
#include <string>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/memory.hpp>
#include <ginkgo/core/base/timer.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/version.hpp>


namespace gko {


version version_info::get_csl_version() noexcept
{
    // We just return the version with a special "not compiled" tag in
    // placeholder modules.
    return {GKO_VERSION_STR, "not compiled"};
}


std::shared_ptr<CslExecutor> CslExecutor::create(
    int device_id, std::shared_ptr<Executor> master)
{
    return std::shared_ptr<CslExecutor>(
        new CslExecutor(device_id, std::move(master)));
}


void CslExecutor::populate_exec_info(const machine_topology* mach_topo)
{
    // This method is always called, so cannot throw when not compiled.
}


void OmpExecutor::raw_copy_to(const CslExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(csl);


void DpcppExecutor::raw_copy_to(const CslExecutor*, size_type num_bytes,
                                const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(csl);


void HipExecutor::raw_copy_to(const CslExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(csl);


void CudaExecutor::raw_copy_to(const CslExecutor*, size_type num_bytes,
                               const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(csl);


void CslExecutor::raw_free(void* ptr) const noexcept
{
    // Free must never fail, as it can be called in destructors.
    // If the nvidia module was not compiled, the library couldn't have
    // allocated the memory, so there is no need to deallocate it.
}


void* CslExecutor::raw_alloc(size_type num_bytes) const GKO_NOT_COMPILED(csl);


void CslExecutor::raw_copy_to(const OmpExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(csl);


void CslExecutor::raw_copy_to(const DpcppExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(csl);


void CslExecutor::raw_copy_to(const CudaExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(csl);


void CslExecutor::raw_copy_to(const HipExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(csl);


void CslExecutor::raw_copy_to(const CslExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(csl);


void CslExecutor::synchronize() const GKO_NOT_COMPILED(csl);


scoped_device_id_guard CslExecutor::get_scoped_device_id_guard() const
    GKO_NOT_COMPILED(csl);


std::string CslExecutor::get_description() const GKO_NOT_COMPILED(csl);


int CslExecutor::get_num_devices(std::string) { return 0; }


bool CslExecutor::verify_memory_to(const OmpExecutor* dest_exec) const
{
    // Dummy check
    // TODO
    return false;
}

bool CslExecutor::verify_memory_to(const CslExecutor* dest_exec) const
{
    // Dummy check
    // TODO
    return false;
}


scoped_device_id_guard::scoped_device_id_guard(const CslExecutor* exec,
                                               int device_id)
    GKO_NOT_COMPILED(csl);


CslTimer::CslTimer(std::shared_ptr<const CslExecutor> exec)
    GKO_NOT_COMPILED(csl);


void CslTimer::init_time_point(time_point&) GKO_NOT_COMPILED(csl);


void CslTimer::record(time_point&) GKO_NOT_COMPILED(csl);


void CslTimer::wait(time_point& time) GKO_NOT_COMPILED(csl);


std::chrono::nanoseconds CslTimer::difference_async(const time_point& start,
                                                    const time_point& stop)
    GKO_NOT_COMPILED(csl);


namespace kernels {
namespace csl {


void reset_device(int device_id) GKO_NOT_COMPILED(csl);


void destroy_event(::csl::event* event) GKO_NOT_COMPILED(csl);


}  // namespace csl
}  // namespace kernels


}  // namespace gko


#define GKO_HOOK_MODULE csl
#include "core/device_hooks/common_kernels.inc.cpp"
#undef GKO_HOOK_MODULE
