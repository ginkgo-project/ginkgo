/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/base/memory_space.hpp>


#include <iostream>


#include <hip/hip_runtime.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include "hip/base/device_guard.hip.hpp"
#include "hip/base/stream_bindings.hip.hpp"


namespace gko {


HipMemorySpace::HipMemorySpace(int device_id) : device_id_(device_id)
{
    assert(device_id < max_devices);

    this->default_input_stream_ = HipAsyncHandle::create();
    this->default_output_stream_ = HipAsyncHandle::create();
}


std::shared_ptr<AsyncHandle> HostMemorySpace::raw_copy_to(
    const HipMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
    auto stream = as<HipAsyncHandle>(handle)->get_handle();
    if (num_bytes > 0) {
        hip::device_guard g(dest->get_device_id());
        GKO_ASSERT_NO_HIP_ERRORS(hipMemcpyAsync(dest_ptr, src_ptr, num_bytes,
                                                hipMemcpyHostToDevice, stream));
    }
    return handle;
}


std::shared_ptr<AsyncHandle> ReferenceMemorySpace::raw_copy_to(
    const HipMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
    auto stream = as<HipAsyncHandle>(handle)->get_handle();
    if (num_bytes > 0) {
        hip::device_guard g(dest->get_device_id());
        GKO_ASSERT_NO_HIP_ERRORS(hipMemcpyAsync(dest_ptr, src_ptr, num_bytes,
                                                hipMemcpyHostToDevice, stream));
    }
    return handle;
}


void HipMemorySpace::raw_free(void* ptr) const noexcept
{
    hip::device_guard g(this->get_device_id());
    auto error_code = hipFree(ptr);
    if (error_code != hipSuccess) {
#if GKO_VERBOSE_LEVEL >= 1
        // Unfortunately, if memory free fails, there's not much we can do
        std::cerr << "Unrecoverable HIP error on device " << this->device_id_
                  << " in " << __func__ << ": " << hipGetErrorName(error_code)
                  << ": " << hipGetErrorString(error_code) << std::endl
                  << "Exiting program" << std::endl;
#endif
        std::exit(error_code);
    }
}


void* HipMemorySpace::raw_alloc(size_type num_bytes) const
{
    void* dev_ptr = nullptr;
    hip::device_guard g(this->get_device_id());
    auto error_code = hipMalloc(&dev_ptr, num_bytes);
    if (error_code != hipErrorMemoryAllocation) {
        GKO_ASSERT_NO_HIP_ERRORS(error_code);
    }
    GKO_ENSURE_ALLOCATED(dev_ptr, "hip", num_bytes);
    return dev_ptr;
}


std::shared_ptr<AsyncHandle> HipMemorySpace::raw_copy_to(
    const HostMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
    auto stream =
        as<HipAsyncHandle>(this->get_default_output_stream())->get_handle();
    if (num_bytes > 0) {
        hip::device_guard g(this->get_device_id());
        GKO_ASSERT_NO_HIP_ERRORS(hipMemcpyAsync(dest_ptr, src_ptr, num_bytes,
                                                hipMemcpyDeviceToHost, stream));
    }
    return this->get_default_output_stream();
}


std::shared_ptr<AsyncHandle> HipMemorySpace::raw_copy_to(
    const ReferenceMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
    auto stream =
        as<HipAsyncHandle>(this->get_default_output_stream())->get_handle();
    if (num_bytes > 0) {
        hip::device_guard g(this->get_device_id());
        GKO_ASSERT_NO_HIP_ERRORS(hipMemcpyAsync(dest_ptr, src_ptr, num_bytes,
                                                hipMemcpyDeviceToHost, stream));
    }
    return this->get_default_output_stream();
}


std::shared_ptr<AsyncHandle> HipMemorySpace::raw_copy_to(
    const CudaMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
#if GINKGO_HIP_PLATFORM_NVCC == 1
    auto stream = as<HipAsyncHandle>(handle)->get_handle();
    if (num_bytes > 0) {
        hip::device_guard g(this->get_device_id());
        GKO_ASSERT_NO_HIP_ERRORS(
            hipMemcpyPeerAsync(dest_ptr, dest->get_device_id(), src_ptr,
                               this->get_device_id(), num_bytes, stream));
    }
    return handle;
#else
    GKO_NOT_SUPPORTED(this);
#endif
}


std::shared_ptr<AsyncHandle> HipMemorySpace::raw_copy_to(
    const CudaUVMSpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
#if GINKGO_HIP_PLATFORM_NVCC == 1
    auto stream = as<HipAsyncHandle>(handle)->get_handle();
    if (num_bytes > 0) {
        hip::device_guard g(this->get_device_id());
        GKO_ASSERT_NO_HIP_ERRORS(
            hipMemcpyPeerAsync(dest_ptr, dest->get_device_id(), src_ptr,
                               this->get_device_id(), num_bytes, stream));
    }
    return handle;
#else
    GKO_NOT_SUPPORTED(this);
#endif
}


std::shared_ptr<AsyncHandle> HipMemorySpace::raw_copy_to(
    const DpcppMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
    GKO_NOT_SUPPORTED(this);


std::shared_ptr<AsyncHandle> HipMemorySpace::raw_copy_to(
    const HipMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
    auto stream = as<HipAsyncHandle>(handle)->get_handle();
    if (num_bytes > 0) {
        hip::device_guard g(this->get_device_id());
        GKO_ASSERT_NO_HIP_ERRORS(
            hipMemcpyPeerAsync(dest_ptr, dest->get_device_id(), src_ptr,
                               this->get_device_id(), num_bytes, stream));
    }
    return handle;
}


void HipMemorySpace::synchronize() const
{
    hip::device_guard g(this->get_device_id());
    GKO_ASSERT_NO_HIP_ERRORS(hipDeviceSynchronize());
}


int HipMemorySpace::get_num_devices()
{
    int deviceCount = 0;
    auto error_code = hipGetDeviceCount(&deviceCount);
    if (error_code == hipErrorNoDevice) {
        return 0;
    }
    GKO_ASSERT_NO_HIP_ERRORS(error_code);
    return deviceCount;
}


}  // namespace gko
