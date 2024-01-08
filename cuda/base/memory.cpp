// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/memory.hpp>


#include <cuda.h>
#include <cuda_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "cuda/base/scoped_device_id.hpp"


namespace gko {


#define GKO_ASSERT_NO_CUDA_ALLOCATION_ERRORS(_operation, _size)       \
    {                                                                 \
        auto error_code = _operation;                                 \
        if (error_code == cudaErrorMemoryAllocation) {                \
            throw AllocationError(__FILE__, __LINE__, "cuda", _size); \
        } else {                                                      \
            GKO_ASSERT_NO_CUDA_ERRORS(error_code);                    \
        }                                                             \
    }


#if GKO_VERBOSE_LEVEL >= 1
#define GKO_EXIT_ON_CUDA_ERROR(_operation)                                  \
    {                                                                       \
        const auto error_code = _operation;                                 \
        if (error_code != cudaSuccess) {                                    \
            int device_id{-1};                                              \
            cudaGetDevice(&device_id);                                      \
            std::cerr << "Unrecoverable CUDA error on device " << device_id \
                      << " in " << __func__ << ":" << __LINE__ << ": "      \
                      << cudaGetErrorName(error_code) << ": "               \
                      << cudaGetErrorString(error_code) << std::endl        \
                      << "Exiting program" << std::endl;                    \
            std::exit(error_code);                                          \
        }                                                                   \
    }
#else
#define GKO_EXIT_ON_CUDA_ERROR(_operation)  \
    {                                       \
        const auto error_code = _operation; \
        if (error_code != cudaSuccess) {    \
            std::exit(error_code);          \
        }                                   \
    }
#endif


void* CudaAllocator::allocate(size_type num_bytes)
{
    void* ptr{};
    GKO_ASSERT_NO_CUDA_ALLOCATION_ERRORS(cudaMalloc(&ptr, num_bytes),
                                         num_bytes);
    return ptr;
}


void CudaAllocator::deallocate(void* ptr)
{
    GKO_EXIT_ON_CUDA_ERROR(cudaFree(ptr));
}


#if CUDA_VERSION >= 11020


CudaAsyncAllocator::CudaAsyncAllocator(cudaStream_t stream) : stream_{stream} {}


void* CudaAsyncAllocator::allocate(size_type num_bytes)
{
    void* ptr{};
    GKO_ASSERT_NO_CUDA_ALLOCATION_ERRORS(
        cudaMallocAsync(&ptr, num_bytes, stream_), num_bytes);
    return ptr;
}


void CudaAsyncAllocator::deallocate(void* ptr)
{
    GKO_EXIT_ON_CUDA_ERROR(cudaFreeAsync(ptr, stream_));
}


#else  // Fall back to regular allocation


CudaAsyncAllocator::CudaAsyncAllocator(cudaStream_t stream) : stream_{stream}
{
#if GKO_VERBOSE_LEVEL >= 1
    std::cerr << "This version of CUDA does not support cudaMallocAsync, "
                 "please use CudaAllocator instead of CudaAsyncAllocator.\n";
#endif
}


void* CudaAsyncAllocator::allocate(size_type num_bytes)
{
    void* ptr{};
    GKO_ASSERT_NO_CUDA_ALLOCATION_ERRORS(cudaMalloc(&ptr, num_bytes),
                                         num_bytes);
    return ptr;
}


void CudaAsyncAllocator::deallocate(void* ptr)
{
    GKO_EXIT_ON_CUDA_ERROR(cudaFree(ptr));
}


#endif


bool CudaAsyncAllocator::check_environment(int device_id,
                                           CUstream_st* stream) const
{
    return stream == stream_;
}


CudaUnifiedAllocator::CudaUnifiedAllocator(int device_id)
    : CudaUnifiedAllocator{device_id, cudaMemAttachGlobal}
{}


CudaUnifiedAllocator::CudaUnifiedAllocator(int device_id, unsigned int flags)
    : device_id_{device_id}, flags_{flags}
{}


void* CudaUnifiedAllocator::allocate(size_type num_bytes)
{
    // we need to set the device ID in case this gets used in a host executor
    detail::cuda_scoped_device_id_guard g(device_id_);
    void* ptr{};
    GKO_ASSERT_NO_CUDA_ALLOCATION_ERRORS(
        cudaMallocManaged(&ptr, num_bytes, flags_), num_bytes);
    return ptr;
}


void CudaUnifiedAllocator::deallocate(void* ptr)
{
    // we need to set the device ID in case this gets used in a host executor
    detail::cuda_scoped_device_id_guard g(device_id_);
    GKO_EXIT_ON_CUDA_ERROR(cudaFree(ptr));
}


bool CudaUnifiedAllocator::check_environment(int device_id,
                                             CUstream_st* stream) const
{
    return device_id == device_id_;
}


CudaHostAllocator::CudaHostAllocator(int device_id) : device_id_{device_id} {}


void* CudaHostAllocator::allocate(size_type num_bytes)
{
    // we need to set the device ID in case this gets used in a host executor
    detail::cuda_scoped_device_id_guard g(device_id_);
    void* ptr{};
    GKO_ASSERT_NO_CUDA_ALLOCATION_ERRORS(cudaMallocHost(&ptr, num_bytes),
                                         num_bytes);
    return ptr;
}


void CudaHostAllocator::deallocate(void* ptr)
{
    // we need to set the device ID in case this gets used in a host executor
    detail::cuda_scoped_device_id_guard g(device_id_);
    GKO_EXIT_ON_CUDA_ERROR(cudaFreeHost(ptr));
}


bool CudaHostAllocator::check_environment(int device_id,
                                          CUstream_st* stream) const
{
    return device_id == device_id_;
}


}  // namespace gko
