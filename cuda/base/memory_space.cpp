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


#include <cuda_runtime.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include "cuda/base/device_guard.hpp"
#include "cuda/base/stream_bindings.hpp"


namespace gko {


CudaMemorySpace::CudaMemorySpace(int device_id) : device_id_(device_id)
{
    assert(device_id < max_devices);

    this->default_input_stream_ = CudaAsyncHandle::create();
    this->default_output_stream_ = CudaAsyncHandle::create();
}


CudaUVMSpace::CudaUVMSpace(int device_id) : device_id_(device_id)
{
    assert(device_id < max_devices);

    this->default_input_stream_ = CudaAsyncHandle::create();
    this->default_output_stream_ = CudaAsyncHandle::create();
}


void CudaMemorySpace::synchronize() const
{
    cuda::device_guard g(this->get_device_id());
    GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
}


void CudaUVMSpace::synchronize() const
{
    cuda::device_guard g(this->get_device_id());
    GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
}


int CudaMemorySpace::get_num_devices()
{
    int deviceCount = 0;
    auto error_code = cudaGetDeviceCount(&deviceCount);
    if (error_code == cudaErrorNoDevice) {
        return 0;
    }
    GKO_ASSERT_NO_CUDA_ERRORS(error_code);
    return deviceCount;
}


int CudaUVMSpace::get_num_devices()
{
    int deviceCount = 0;
    auto error_code = cudaGetDeviceCount(&deviceCount);
    if (error_code == cudaErrorNoDevice) {
        return 0;
    }
    GKO_ASSERT_NO_CUDA_ERRORS(error_code);
    return deviceCount;
}


std::shared_ptr<AsyncHandle> HostMemorySpace::raw_copy_to(
    const CudaMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
    auto stream = as<CudaAsyncHandle>(handle)->get_handle();
    if (num_bytes > 0) {
        cuda::device_guard g(dest->get_device_id());
        GKO_ASSERT_NO_CUDA_ERRORS(cudaMemcpyAsync(
            dest_ptr, src_ptr, num_bytes, cudaMemcpyHostToDevice, stream));
    }
    return handle;
}


std::shared_ptr<AsyncHandle> ReferenceMemorySpace::raw_copy_to(
    const CudaMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
    auto stream = as<CudaAsyncHandle>(handle)->get_handle();
    if (num_bytes > 0) {
        cuda::device_guard g(dest->get_device_id());
        GKO_ASSERT_NO_CUDA_ERRORS(cudaMemcpyAsync(
            dest_ptr, src_ptr, num_bytes, cudaMemcpyHostToDevice, stream));
    }
    return handle;
}


void CudaMemorySpace::raw_free(void* ptr) const noexcept
{
    cuda::device_guard g(this->get_device_id());
    auto error_code = cudaFree(ptr);
    if (error_code != cudaSuccess) {
#if GKO_VERBOSE_LEVEL >= 1
        // Unfortunately, if memory free fails, there's not much we can do
        std::cerr << "Unrecoverable CUDA error on device " << this->device_id_
                  << " in " << __func__ << ": " << cudaGetErrorName(error_code)
                  << ": " << cudaGetErrorString(error_code) << std::endl
                  << "Exiting program" << std::endl;
#endif
        std::exit(error_code);
    }
}


void CudaMemorySpace::raw_free_pinned_host(void* ptr) const
{
    cuda::device_guard g(this->get_device_id());
    auto error_code = cudaFreeHost(ptr);
    if (error_code != cudaSuccess) {
#if GKO_VERBOSE_LEVEL >= 1
        // Unfortunately, if memory free fails, there's not much we can do
        std::cerr << "Unrecoverable CUDA error on device " << this->device_id_
                  << " in " << __func__ << ": " << cudaGetErrorName(error_code)
                  << ": " << cudaGetErrorString(error_code) << std::endl
                  << "Exiting program" << std::endl;
#endif
        std::exit(error_code);
    }
}


void CudaUVMSpace::raw_free(void* ptr) const noexcept
{
    cuda::device_guard g(this->get_device_id());
    auto error_code = cudaFree(ptr);
    if (error_code != cudaSuccess) {
#if GKO_VERBOSE_LEVEL >= 1
        // Unfortunately, if memory free fails, there's not much we can do
        std::cerr << "Unrecoverable CUDA error on device " << this->device_id_
                  << " in " << __func__ << ": " << cudaGetErrorName(error_code)
                  << ": " << cudaGetErrorString(error_code) << std::endl
                  << "Exiting program" << std::endl;
#endif
        std::exit(error_code);
    }
}


void* CudaMemorySpace::raw_pinned_host_alloc(size_type num_bytes) const
{
    void* dev_ptr = nullptr;
    cuda::device_guard g(this->get_device_id());
    auto error_code = cudaHostAlloc(&dev_ptr, num_bytes, 0);
    if (error_code != cudaErrorMemoryAllocation) {
        GKO_ASSERT_NO_CUDA_ERRORS(error_code);
    }
    GKO_ENSURE_ALLOCATED(dev_ptr, "cuda", num_bytes);
    return dev_ptr;
}


void* CudaMemorySpace::raw_alloc(size_type num_bytes) const
{
    void* dev_ptr = nullptr;
    cuda::device_guard g(this->get_device_id());
    auto error_code = cudaMalloc(&dev_ptr, num_bytes);
    if (error_code != cudaErrorMemoryAllocation) {
        GKO_ASSERT_NO_CUDA_ERRORS(error_code);
    }
    GKO_ENSURE_ALLOCATED(dev_ptr, "cuda", num_bytes);
    return dev_ptr;
}


std::shared_ptr<AsyncHandle> CudaMemorySpace::raw_copy_to(
    const HostMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
    if (auto cuda_handle = dynamic_cast<CudaAsyncHandle*>(handle.get())) {
        auto stream = (cuda_handle)->get_handle();
        if (num_bytes > 0) {
            cuda::device_guard g(this->get_device_id());
            GKO_ASSERT_NO_CUDA_ERRORS(cudaMemcpyAsync(
                dest_ptr, src_ptr, num_bytes, cudaMemcpyDeviceToHost, stream));
        }
        std::cout << " Here " << __LINE__ << std::endl;
        return handle;
    } else {
        auto stream = as<CudaAsyncHandle>(this->get_default_output_stream())
                          ->get_handle();
        if (num_bytes > 0) {
            cuda::device_guard g(this->get_device_id());
            GKO_ASSERT_NO_CUDA_ERRORS(cudaMemcpyAsync(
                dest_ptr, src_ptr, num_bytes, cudaMemcpyDeviceToHost, stream));
        }
        std::cout << " Here " << __LINE__ << std::endl;
        return this->get_default_output_stream();
    }
}


std::shared_ptr<AsyncHandle> CudaMemorySpace::raw_copy_to(
    const ReferenceMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
    if (auto cuda_handle = dynamic_cast<CudaAsyncHandle*>(handle.get())) {
        auto stream = (cuda_handle)->get_handle();
        if (num_bytes > 0) {
            cuda::device_guard g(this->get_device_id());
            GKO_ASSERT_NO_CUDA_ERRORS(
                cudaMemcpyAsync(dest_ptr, src_ptr, num_bytes, cudaMemcpyDefault,
                                stream););
        }
        return handle;
    } else {
        auto stream = as<CudaAsyncHandle>(this->get_default_output_stream())
                          ->get_handle();
        if (num_bytes > 0) {
            cuda::device_guard g(this->get_device_id());
            GKO_ASSERT_NO_CUDA_ERRORS(cudaMemcpyAsync(
                dest_ptr, src_ptr, num_bytes, cudaMemcpyDeviceToHost, stream));
        }
        return this->get_default_output_stream();
    }
}


std::shared_ptr<AsyncHandle> CudaMemorySpace::raw_copy_to(
    const CudaMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
    auto stream = as<CudaAsyncHandle>(handle)->get_handle();
    if (num_bytes > 0) {
        cuda::device_guard g(this->get_device_id());
        GKO_ASSERT_NO_CUDA_ERRORS(
            cudaMemcpyPeerAsync(dest_ptr, dest->get_device_id(), src_ptr,
                                this->get_device_id(), num_bytes, stream));
    }
    return handle;
}


std::shared_ptr<AsyncHandle> CudaUVMSpace::raw_copy_to(
    const HipMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
#if GINKGO_HIP_PLATFORM_NVCC == 1
    auto stream = as<CudaAsyncHandle>(handle)->get_handle();
    if (num_bytes > 0) {
        cuda::device_guard g(this->get_device_id());
        GKO_ASSERT_NO_CUDA_ERRORS(
            cudaMemcpyPeerAsync(dest_ptr, dest->get_device_id(), src_ptr,
                                this->get_device_id(), num_bytes, stream));
    }
    return handle;
#else
    GKO_NOT_SUPPORTED(this);
#endif
}


std::shared_ptr<AsyncHandle> CudaMemorySpace::raw_copy_to(
    const HipMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
#if GINKGO_HIP_PLATFORM_NVCC == 1
    auto stream = as<CudaAsyncHandle>(handle)->get_handle();
    if (num_bytes > 0) {
        cuda::device_guard g(this->get_device_id());
        GKO_ASSERT_NO_CUDA_ERRORS(
            cudaMemcpyPeerAsync(dest_ptr, dest->get_device_id(), src_ptr,
                                this->get_device_id(), num_bytes, stream));
    }
    return handle;
#else
    GKO_NOT_SUPPORTED(this);
#endif
}


std::shared_ptr<AsyncHandle> CudaMemorySpace::raw_copy_to(
    const CudaUVMSpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
    auto stream = as<CudaAsyncHandle>(handle)->get_handle();
    if (num_bytes > 0) {
        cuda::device_guard g(this->get_device_id());
        GKO_ASSERT_NO_CUDA_ERRORS(
            cudaMemcpyPeerAsync(dest_ptr, dest->get_device_id(), src_ptr,
                                this->get_device_id(), num_bytes, stream));
    }
    return handle;
}


std::shared_ptr<AsyncHandle> CudaMemorySpace::raw_copy_to(
    const DpcppMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
    GKO_NOT_SUPPORTED(this);


std::shared_ptr<AsyncHandle> CudaUVMSpace::raw_copy_to(
    const CudaMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
    auto stream = as<CudaAsyncHandle>(handle)->get_handle();
    if (num_bytes > 0) {
        cuda::device_guard g(this->get_device_id());
        GKO_ASSERT_NO_CUDA_ERRORS(
            cudaMemcpyPeerAsync(dest_ptr, dest->get_device_id(), src_ptr,
                                this->get_device_id(), num_bytes, stream));
    }
    return handle;
}


std::shared_ptr<AsyncHandle> CudaUVMSpace::raw_copy_to(
    const CudaUVMSpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
    auto stream = as<CudaAsyncHandle>(handle)->get_handle();
    if (num_bytes > 0) {
        cuda::device_guard g(this->get_device_id());
        GKO_ASSERT_NO_CUDA_ERRORS(
            cudaMemcpyPeerAsync(dest_ptr, dest->get_device_id(), src_ptr,
                                this->get_device_id(), num_bytes, stream));
    }
    return handle;
}


std::shared_ptr<AsyncHandle> CudaUVMSpace::raw_copy_to(
    const DpcppMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
    GKO_NOT_SUPPORTED(this);


std::shared_ptr<AsyncHandle> HostMemorySpace::raw_copy_to(
    const CudaUVMSpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
    auto stream = as<CudaAsyncHandle>(handle)->get_handle();
    if (num_bytes > 0) {
        cuda::device_guard g(dest->get_device_id());
        GKO_ASSERT_NO_CUDA_ERRORS(cudaMemcpyAsync(
            dest_ptr, src_ptr, num_bytes, cudaMemcpyHostToDevice, stream));
    }
    return handle;
}


std::shared_ptr<AsyncHandle> ReferenceMemorySpace::raw_copy_to(
    const CudaUVMSpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
    auto stream = as<CudaAsyncHandle>(handle)->get_handle();
    if (num_bytes > 0) {
        cuda::device_guard g(dest->get_device_id());
        GKO_ASSERT_NO_CUDA_ERRORS(cudaMemcpyAsync(
            dest_ptr, src_ptr, num_bytes, cudaMemcpyHostToDevice, stream));
    }
    return handle;
}


std::shared_ptr<AsyncHandle> CudaUVMSpace::raw_copy_to(
    const HostMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
    auto stream = as<CudaAsyncHandle>(handle)->get_handle();
    if (num_bytes > 0) {
        cuda::device_guard g(this->get_device_id());
        GKO_ASSERT_NO_CUDA_ERRORS(cudaMemcpyAsync(
            dest_ptr, src_ptr, num_bytes, cudaMemcpyDeviceToHost, stream));
    }
    return handle;
}


std::shared_ptr<AsyncHandle> CudaUVMSpace::raw_copy_to(
    const ReferenceMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr, std::shared_ptr<AsyncHandle> handle) const
{
    auto stream = as<CudaAsyncHandle>(handle)->get_handle();
    if (num_bytes > 0) {
        cuda::device_guard g(this->get_device_id());
        GKO_ASSERT_NO_CUDA_ERRORS(cudaMemcpyAsync(
            dest_ptr, src_ptr, num_bytes, cudaMemcpyDeviceToHost, stream));
    }
    return handle;
}


void* CudaUVMSpace::raw_alloc(size_type num_bytes) const
{
    void* dev_ptr = nullptr;
    cuda::device_guard g(this->get_device_id());
    auto error_code = cudaMallocManaged(&dev_ptr, num_bytes);
    if (error_code != cudaErrorMemoryAllocation) {
        GKO_ASSERT_NO_CUDA_ERRORS(error_code);
    }
    GKO_ENSURE_ALLOCATED(dev_ptr, "cuda", num_bytes);
    return dev_ptr;
}


}  // namespace gko
