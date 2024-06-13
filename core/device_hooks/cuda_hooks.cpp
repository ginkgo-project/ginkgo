// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/memory.hpp>
#include <ginkgo/core/base/stream.hpp>
#include <ginkgo/core/base/timer.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/version.hpp>
#include <ginkgo/core/log/profiler_hook.hpp>


namespace gko {


version version_info::get_cuda_version() noexcept
{
    // We just return the version with a special "not compiled" tag in
    // placeholder modules.
    return {GKO_VERSION_STR, "not compiled"};
}


void* CudaAllocator::allocate(size_type num_bytes) GKO_NOT_COMPILED(cuda);


void CudaAllocator::deallocate(void* dev_ptr) GKO_NOT_COMPILED(cuda);


CudaAsyncAllocator::CudaAsyncAllocator(CUstream_st* stream)
    GKO_NOT_COMPILED(cuda);


void* CudaAsyncAllocator::allocate(size_type num_bytes) GKO_NOT_COMPILED(cuda);


void CudaAsyncAllocator::deallocate(void* dev_ptr) GKO_NOT_COMPILED(cuda);


bool CudaAsyncAllocator::check_environment(int device_id,
                                           CUstream_st* stream) const
    GKO_NOT_COMPILED(cuda);


CudaUnifiedAllocator::CudaUnifiedAllocator(int device_id)
    GKO_NOT_COMPILED(cuda);


CudaUnifiedAllocator::CudaUnifiedAllocator(int device_id, unsigned int flags)
    GKO_NOT_COMPILED(cuda);


void* CudaUnifiedAllocator::allocate(size_type num_bytes)
    GKO_NOT_COMPILED(cuda);


void CudaUnifiedAllocator::deallocate(void* dev_ptr) GKO_NOT_COMPILED(cuda);


bool CudaUnifiedAllocator::check_environment(int device_id,
                                             CUstream_st* stream) const
    GKO_NOT_COMPILED(cuda);


CudaHostAllocator::CudaHostAllocator(int device_id) GKO_NOT_COMPILED(cuda);


void* CudaHostAllocator::allocate(size_type num_bytes) GKO_NOT_COMPILED(cuda);


void CudaHostAllocator::deallocate(void* dev_ptr) GKO_NOT_COMPILED(cuda);


bool CudaHostAllocator::check_environment(int device_id,
                                          CUstream_st* stream) const
    GKO_NOT_COMPILED(cuda);


std::shared_ptr<CudaExecutor> CudaExecutor::create(
    int device_id, std::shared_ptr<Executor> master, bool device_reset,
    allocation_mode alloc_mode, CUstream_st* stream)
{
    return std::shared_ptr<CudaExecutor>(
        new CudaExecutor(device_id, std::move(master),
                         std::make_shared<CudaAllocator>(), stream));
}


std::shared_ptr<CudaExecutor> CudaExecutor::create(
    int device_id, std::shared_ptr<Executor> master,
    std::shared_ptr<CudaAllocatorBase> alloc, CUstream_st* stream)
{
    return std::shared_ptr<CudaExecutor>(new CudaExecutor(
        device_id, std::move(master), std::move(alloc), stream));
}


void CudaExecutor::populate_exec_info(const machine_topology* mach_topo)
{
    // This method is always called, so cannot throw when not compiled.
}


void OmpExecutor::raw_copy_to(const CudaExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(cuda);


void CudaExecutor::raw_free(void* ptr) const noexcept
{
    // Free must never fail, as it can be called in destructors.
    // If the nvidia module was not compiled, the library couldn't have
    // allocated the memory, so there is no need to deallocate it.
}


void* CudaExecutor::raw_alloc(size_type num_bytes) const GKO_NOT_COMPILED(cuda);


void CudaExecutor::raw_copy_to(const OmpExecutor*, size_type num_bytes,
                               const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(cuda);


void CudaExecutor::raw_copy_to(const CudaExecutor*, size_type num_bytes,
                               const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(cuda);


void CudaExecutor::raw_copy_to(const HipExecutor*, size_type num_bytes,
                               const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(cuda);


void CudaExecutor::raw_copy_to(const DpcppExecutor*, size_type num_bytes,
                               const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(cuda);


void CudaExecutor::synchronize() const GKO_NOT_COMPILED(cuda);


scoped_device_id_guard CudaExecutor::get_scoped_device_id_guard() const
    GKO_NOT_COMPILED(cuda);


std::string CudaError::get_error(int64)
{
    return "ginkgo CUDA module is not compiled";
}


std::string CublasError::get_error(int64)
{
    return "ginkgo CUDA module is not compiled";
}


std::string CurandError::get_error(int64)
{
    return "ginkgo CUDA module is not compiled";
}


std::string CusparseError::get_error(int64)
{
    return "ginkgo CUDA module is not compiled";
}


std::string CufftError::get_error(int64)
{
    return "ginkgo CUDA module is not compiled";
}


int CudaExecutor::get_num_devices() { return 0; }


void CudaExecutor::set_gpu_property() {}


void CudaExecutor::init_handles() {}


scoped_device_id_guard::scoped_device_id_guard(const CudaExecutor* exec,
                                               int device_id)
    GKO_NOT_COMPILED(cuda);


cuda_stream::cuda_stream() GKO_NOT_COMPILED(cuda);


cuda_stream::cuda_stream(int device_id) GKO_NOT_COMPILED(cuda);


cuda_stream::~cuda_stream() {}


cuda_stream::cuda_stream(cuda_stream&&) GKO_NOT_COMPILED(cuda);


CUstream_st* cuda_stream::get() const GKO_NOT_COMPILED(cuda);


CudaTimer::CudaTimer(std::shared_ptr<const CudaExecutor> exec)
    GKO_NOT_COMPILED(cuda);


void CudaTimer::init_time_point(time_point& time) GKO_NOT_COMPILED(cuda);


void CudaTimer::record(time_point&) GKO_NOT_COMPILED(cuda);


void CudaTimer::wait(time_point& time) GKO_NOT_COMPILED(cuda);


std::chrono::nanoseconds CudaTimer::difference_async(const time_point& start,
                                                     const time_point& stop)
    GKO_NOT_COMPILED(cuda);


namespace kernels {
namespace cuda {


void reset_device(int device_id) GKO_NOT_COMPILED(cuda);


void destroy_event(CUevent_st* event) GKO_NOT_COMPILED(cuda);


}  // namespace cuda
}  // namespace kernels


namespace log {


void init_nvtx() GKO_NOT_COMPILED(cuda);


std::function<void(const char*, profile_event_category)> begin_nvtx_fn(
    uint32_t color_rgb) GKO_NOT_COMPILED(cuda);


void end_nvtx(const char*, profile_event_category) GKO_NOT_COMPILED(cuda);


}  // namespace log
}  // namespace gko


#define GKO_HOOK_MODULE cuda
#include "core/device_hooks/common_kernels.inc.cpp"
#undef GKO_HOOK_MODULE
