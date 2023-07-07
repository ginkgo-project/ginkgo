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


#if HIP_VERSION_MAJOR >= 5


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


HipAsyncAllocator::HipAsyncAllocator(hipStream_t stream) : stream_{stream} {}


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
