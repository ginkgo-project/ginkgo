// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>
#include <string>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/memory.hpp>
#include <ginkgo/core/base/stream.hpp>
#include <ginkgo/core/base/timer.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/version.hpp>
#include <ginkgo/core/log/profiler_hook.hpp>


namespace gko {


version version_info::get_hip_version() noexcept
{
    // We just return 1.1.0 with a special "not compiled" tag in placeholder
    // modules.
    return {1, 1, 0, "not compiled"};
}


void* HipAllocator::allocate(size_type num_bytes) GKO_NOT_COMPILED(hip);


void HipAllocator::deallocate(void* dev_ptr) GKO_NOT_COMPILED(hip);


HipAsyncAllocator::HipAsyncAllocator(GKO_HIP_STREAM_STRUCT* stream)
    GKO_NOT_COMPILED(hip);


void* HipAsyncAllocator::allocate(size_type num_bytes) GKO_NOT_COMPILED(hip);


void HipAsyncAllocator::deallocate(void* dev_ptr) GKO_NOT_COMPILED(hip);


bool HipAsyncAllocator::check_environment(int device_id,
                                          GKO_HIP_STREAM_STRUCT* stream) const
    GKO_NOT_COMPILED(hip);


HipUnifiedAllocator::HipUnifiedAllocator(int device_id) GKO_NOT_COMPILED(hip);


HipUnifiedAllocator::HipUnifiedAllocator(int device_id, unsigned int flags)
    GKO_NOT_COMPILED(hip);


void* HipUnifiedAllocator::allocate(size_type num_bytes) GKO_NOT_COMPILED(hip);


void HipUnifiedAllocator::deallocate(void* dev_ptr) GKO_NOT_COMPILED(hip);


bool HipUnifiedAllocator::check_environment(int device_id,
                                            GKO_HIP_STREAM_STRUCT* stream) const
    GKO_NOT_COMPILED(hip);


HipHostAllocator::HipHostAllocator(int device_id) GKO_NOT_COMPILED(hip);


void* HipHostAllocator::allocate(size_type num_bytes) GKO_NOT_COMPILED(hip);


void HipHostAllocator::deallocate(void* dev_ptr) GKO_NOT_COMPILED(hip);


bool HipHostAllocator::check_environment(int device_id,
                                         GKO_HIP_STREAM_STRUCT* stream) const
    GKO_NOT_COMPILED(hip);


std::shared_ptr<HipExecutor> HipExecutor::create(
    int device_id, std::shared_ptr<Executor> master, bool device_reset,
    allocation_mode alloc_mode, GKO_HIP_STREAM_STRUCT* stream)
{
    return std::shared_ptr<HipExecutor>(
        new HipExecutor(device_id, std::move(master),
                        std::make_shared<HipAllocator>(), stream));
}


std::shared_ptr<HipExecutor> HipExecutor::create(
    int device_id, std::shared_ptr<Executor> master,
    std::shared_ptr<HipAllocatorBase> alloc, GKO_HIP_STREAM_STRUCT* stream)
{
    return std::shared_ptr<HipExecutor>(
        new HipExecutor(device_id, std::move(master), alloc, stream));
}


void HipExecutor::populate_exec_info(const machine_topology* mach_topo)
{
    // This method is always called, so cannot throw when not compiled.
}


void OmpExecutor::raw_copy_to(const HipExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(hip);


void HipExecutor::raw_free(void* ptr) const noexcept
{
    // Free must never fail, as it can be called in destructors.
    // If the nvidia module was not compiled, the library couldn't have
    // allocated the memory, so there is no need to deallocate it.
}


void* HipExecutor::raw_alloc(size_type num_bytes) const GKO_NOT_COMPILED(hip);


void HipExecutor::raw_copy_to(const OmpExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(hip);


void HipExecutor::raw_copy_to(const CudaExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(hip);


void HipExecutor::raw_copy_to(const HipExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(hip);


void HipExecutor::raw_copy_to(const DpcppExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(hip);


void HipExecutor::synchronize() const GKO_NOT_COMPILED(hip);


scoped_device_id_guard HipExecutor::get_scoped_device_id_guard() const
    GKO_NOT_COMPILED(hip);


std::string HipError::get_error(int64)
{
    return "ginkgo HIP module is not compiled";
}


std::string HipblasError::get_error(int64)
{
    return "ginkgo HIP module is not compiled";
}


std::string HiprandError::get_error(int64)
{
    return "ginkgo HIP module is not compiled";
}


std::string HipsparseError::get_error(int64)
{
    return "ginkgo HIP module is not compiled";
}


std::string HipfftError::get_error(int64)
{
    return "ginkgo HIP module is not compiled";
}


int HipExecutor::get_num_devices() { return 0; }


void HipExecutor::set_gpu_property() {}


void HipExecutor::init_handles() {}


scoped_device_id_guard::scoped_device_id_guard(const HipExecutor* exec,
                                               int device_id)
    GKO_NOT_COMPILED(hip);


hip_stream::hip_stream() GKO_NOT_COMPILED(hip);


hip_stream::hip_stream(int device_id) GKO_NOT_COMPILED(hip);


hip_stream::~hip_stream() {}


hip_stream::hip_stream(hip_stream&&) GKO_NOT_COMPILED(hip);


GKO_HIP_STREAM_STRUCT* hip_stream::get() const GKO_NOT_COMPILED(hip);


HipTimer::HipTimer(std::shared_ptr<const HipExecutor> exec)
    GKO_NOT_COMPILED(hip);


void HipTimer::init_time_point(time_point& time) GKO_NOT_COMPILED(hip);


void HipTimer::record(time_point&) GKO_NOT_COMPILED(hip);


void HipTimer::wait(time_point& time) GKO_NOT_COMPILED(hip);


std::chrono::nanoseconds HipTimer::difference_async(const time_point& start,
                                                    const time_point& stop)
    GKO_NOT_COMPILED(hip);


namespace kernels {
namespace hip {


void reset_device(int device_id) GKO_NOT_COMPILED(hip);


void destroy_event(GKO_HIP_EVENT_STRUCT* event) GKO_NOT_COMPILED(hip);


}  // namespace hip
}  // namespace kernels


namespace log {


void begin_roctx(const char*, profile_event_category) GKO_NOT_COMPILED(hip);


void end_roctx(const char*, profile_event_category) GKO_NOT_COMPILED(hip);


}  // namespace log
}  // namespace gko


#define GKO_HOOK_MODULE hip
#include "core/device_hooks/common_kernels.inc.cpp"
#undef GKO_HOOK_MODULE
