// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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


version version_info::get_poplar_version() noexcept
{
    // We just return the version with a special "not compiled" tag in
    // placeholder modules.
    return {GKO_VERSION_STR, "not compiled"};
}


std::shared_ptr<PoplarExecutor> PoplarExecutor::create(
    int device_id, std::shared_ptr<Executor> master)
{
    return std::shared_ptr<PoplarExecutor>(
        new PoplarExecutor(device_id, std::move(master)));
}


void PoplarExecutor::populate_exec_info(const machine_topology* mach_topo)
{
    // This method is always called, so cannot throw when not compiled.
}


void OmpExecutor::raw_copy_to(const PoplarExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(poplar);


bool OmpExecutor::verify_memory_to(const PoplarExecutor* dest_exec) const
{
    // Dummy check
    // TODO
    return false;
}


void PoplarExecutor::raw_free(void* ptr) const noexcept
{
    // Free must never fail, as it can be called in destructors.
    // If the nvidia module was not compiled, the library couldn't have
    // allocated the memory, so there is no need to deallocate it.
}


void* PoplarExecutor::raw_alloc(size_type num_bytes) const
    GKO_NOT_COMPILED(poplar);


void PoplarExecutor::raw_copy_to(const OmpExecutor*, size_type num_bytes,
                                 const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(poplar);


void PoplarExecutor::raw_copy_to(const DpcppExecutor*, size_type num_bytes,
                                 const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(poplar);


void PoplarExecutor::raw_copy_to(const CudaExecutor*, size_type num_bytes,
                                 const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(poplar);


void PoplarExecutor::raw_copy_to(const HipExecutor*, size_type num_bytes,
                                 const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(poplar);


void PoplarExecutor::raw_copy_to(const PoplarExecutor*, size_type num_bytes,
                                 const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(poplar);


void PoplarExecutor::synchronize() const GKO_NOT_COMPILED(poplar);


scoped_device_id_guard PoplarExecutor::get_scoped_device_id_guard() const
    GKO_NOT_COMPILED(poplar);


std::string PoplarExecutor::get_description() const GKO_NOT_COMPILED(poplar);


int PoplarExecutor::get_num_devices(std::string) { return 0; }


void PoplarExecutor::set_device_property(poplar_queue_property property) {}


bool PoplarExecutor::verify_memory_to(const OmpExecutor* dest_exec) const
{
    // Dummy check
    // TODO
    return false;
}

bool PoplarExecutor::verify_memory_to(const PoplarExecutor* dest_exec) const
{
    // Dummy check
    // TODO
    return false;
}


scoped_device_id_guard::scoped_device_id_guard(const PoplarExecutor* exec,
                                               int device_id)
    GKO_NOT_COMPILED(poplar);


PoplarTimer::PoplarTimer(std::shared_ptr<const PoplarExecutor> exec)
    GKO_NOT_COMPILED(poplar);


void PoplarTimer::init_time_point(time_point&) GKO_NOT_COMPILED(poplar);


void PoplarTimer::record(time_point&) GKO_NOT_COMPILED(poplar);


void PoplarTimer::wait(time_point& time) GKO_NOT_COMPILED(poplar);


std::chrono::nanoseconds PoplarTimer::difference_async(const time_point& start,
                                                       const time_point& stop)
    GKO_NOT_COMPILED(poplar);


}  // namespace gko


#define GKO_HOOK_MODULE poplar
#include "core/device_hooks/common_kernels.inc.cpp"
#undef GKO_HOOK_MODULE
