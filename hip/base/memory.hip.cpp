// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/memory.hpp>


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "hip/base/scoped_device_id.hip.hpp"


namespace gko {


#define GKO_ASSERT_NO_HIP_ALLOCATION_ERRORS(_operation, _size)       \
    {                                                                \
        auto error_code = _operation;                                \
        if (error_code == hipErrorMemoryAllocation) {                \
            throw AllocationError(__FILE__, __LINE__, "hip", _size); \
        } else {                                                     \
            GKO_ASSERT_NO_HIP_ERRORS(error_code);                    \
        }                                                            \
    }


#if GKO_VERBOSE_LEVEL >= 1
#define GKO_EXIT_ON_HIP_ERROR(_operation)                                  \
    {                                                                      \
        const auto error_code = _operation;                                \
        if (error_code != hipSuccess) {                                    \
            int device_id{-1};                                             \
            hipGetDevice(&device_id);                                      \
            std::cerr << "Unrecoverable HIP error on device " << device_id \
                      << " in " << __func__ << ": "                        \
                      << hipGetErrorName(error_code) << ": "               \
                      << hipGetErrorString(error_code) << std::endl        \
                      << "Exiting program" << std::endl;                   \
            std::exit(error_code);                                         \
        }                                                                  \
    }
#else
#define GKO_EXIT_ON_HIP_ERROR(_operation)   \
    {                                       \
        const auto error_code = _operation; \
        if (error_code != hipSuccess) {     \
            std::exit(error_code);          \
        }                                   \
    }
#endif


void* HipAllocator::allocate(size_type num_bytes)
{
    void* dev_ptr{};
    GKO_ASSERT_NO_HIP_ALLOCATION_ERRORS(hipMalloc(&dev_ptr, num_bytes),
                                        num_bytes);
    return dev_ptr;
}


void HipAllocator::deallocate(void* dev_ptr)
{
    GKO_EXIT_ON_HIP_ERROR(hipFree(dev_ptr));
}


#if HIP_VERSION >= 50200000


HipAsyncAllocator::HipAsyncAllocator(hipStream_t stream) : stream_{stream} {}


void* HipAsyncAllocator::allocate(size_type num_bytes)
{
    void* ptr{};
    GKO_ASSERT_NO_HIP_ALLOCATION_ERRORS(
        hipMallocAsync(&ptr, num_bytes, stream_), num_bytes);
    return ptr;
}


void HipAsyncAllocator::deallocate(void* ptr)
{
    GKO_EXIT_ON_HIP_ERROR(hipFreeAsync(ptr, stream_));
}


#else  // Fall back to regular allocation


HipAsyncAllocator::HipAsyncAllocator(hipStream_t stream) : stream_{stream}
{
#if GKO_VERBOSE_LEVEL >= 1
    std::cerr << "This version of HIP does not support hipMallocAsync, "
                 "please use HipAllocator instead of HipAsyncAllocator.\n";
#endif
}


void* HipAsyncAllocator::allocate(size_type num_bytes)
{
    void* ptr{};
    GKO_ASSERT_NO_HIP_ALLOCATION_ERRORS(hipMalloc(&ptr, num_bytes), num_bytes);
    return ptr;
}


void HipAsyncAllocator::deallocate(void* ptr)
{
    GKO_EXIT_ON_HIP_ERROR(hipFree(ptr));
}


#endif


bool HipAsyncAllocator::check_environment(int device_id,
                                          hipStream_t stream) const
{
    return stream == stream_;
}


HipUnifiedAllocator::HipUnifiedAllocator(int device_id)
    : HipUnifiedAllocator{device_id, hipMemAttachGlobal}
{}


HipUnifiedAllocator::HipUnifiedAllocator(int device_id, unsigned int flags)
    : device_id_{device_id}, flags_{flags}
{}


void* HipUnifiedAllocator::allocate(size_type num_bytes)
{
    // we need to set the device ID in case this gets used in a host executor
    detail::hip_scoped_device_id_guard g(device_id_);
    void* ptr{};
    GKO_ASSERT_NO_HIP_ALLOCATION_ERRORS(
        hipMallocManaged(&ptr, num_bytes, flags_), num_bytes);
    return ptr;
}


void HipUnifiedAllocator::deallocate(void* ptr)
{
    // we need to set the device ID in case this gets used in a host executor
    detail::hip_scoped_device_id_guard g(device_id_);
    GKO_EXIT_ON_HIP_ERROR(hipFree(ptr));
}


bool HipUnifiedAllocator::check_environment(int device_id,
                                            hipStream_t stream) const
{
    return device_id == device_id_;
}


HipHostAllocator::HipHostAllocator(int device_id) : device_id_{device_id} {}


void* HipHostAllocator::allocate(size_type num_bytes)
{
    // we need to set the device ID in case this gets used in a host executor
    detail::hip_scoped_device_id_guard g(device_id_);
    void* ptr{};
    GKO_ASSERT_NO_HIP_ALLOCATION_ERRORS(hipHostMalloc(&ptr, num_bytes),
                                        num_bytes);
    return ptr;
}


void HipHostAllocator::deallocate(void* ptr)
{
    // we need to set the device ID in case this gets used in a host executor
    detail::hip_scoped_device_id_guard g(device_id_);
    GKO_EXIT_ON_HIP_ERROR(hipHostFree(ptr));
}


bool HipHostAllocator::check_environment(int device_id,
                                         hipStream_t stream) const
{
    return device_id == device_id_;
}


}  // namespace gko
