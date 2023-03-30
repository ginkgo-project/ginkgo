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

#ifndef GKO_PUBLIC_CORE_BASE_MEMORY_HPP_
#define GKO_PUBLIC_CORE_BASE_MEMORY_HPP_


#include <ginkgo/core/base/fwd_defs.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {


/**
 * Provides generic allocation and deallocation functionality to be used by an
 * Executor.
 */
class Allocator {
public:
    virtual ~Allocator() = default;

    virtual void* allocate(size_type num_bytes) const = 0;

    virtual void deallocate(void* ptr) const = 0;
};


/**
 * Implement this interface to provide an allocator for OmpExecutor or
 * ReferenceExecutor.
 */
class CpuAllocatorBase : public Allocator {};


/**
 * Implement this interface to provide an allocator for CudaExecutor.
 */
class CudaAllocatorBase : public Allocator {};


/**
 * Implement this interface to provide an allocator for HipExecutor.
 */
class HipAllocatorBase : public Allocator {};


/**
 * Implement this interface to provide an allocator for DpcppExecutor.
 */
class DpcppAllocatorBase : public Allocator {
public:
    DpcppAllocatorBase(sycl::queue* queue);

protected:
    virtual void* allocate_impl(sycl::queue* queue,
                                size_type num_bytes) const = 0;

    virtual void deallocate_impl(sycl::queue* queue, void* ptr) const = 0;

private:
    sycl::queue* queue_;
};


/**
 * Allocator using new/delete.
 */
class CpuAllocator : public CpuAllocatorBase {
public:
    void* allocate(size_type num_bytes) const override;

    void deallocate(void* ptr) const override;
};


/**
 * Allocator using cudaMalloc.
 */
class CudaAllocator : public CudaAllocatorBase {
public:
    void* allocate(size_type num_bytes) const override;

    void deallocate(void* ptr) const override;
};


/*
 * Allocator using cudaMallocAsync.
 */
class CudaAsyncAllocator : public CudaAllocatorBase {
public:
    void* allocate(size_type num_bytes) const override;

    void deallocate(void* ptr) const override;

    CudaAsyncAllocator(CUstream_st* stream);

private:
    CUstream_st* stream_;
};


/*
 * Allocator using cudaMallocManaged
 */
class CudaUnifiedAllocator : public CudaAllocatorBase, public CpuAllocatorBase {
public:
    void* allocate(size_type num_bytes) const override;

    void deallocate(void* ptr) const override;

    CudaUnifiedAllocator(int device_id);

    CudaUnifiedAllocator(int device_id, unsigned int flags);

private:
    int device_id_;
    unsigned int flags_;
};


/*
 * Allocator using cudaMallocHost.
 */
class CudaHostAllocator : public CudaAllocatorBase, public CpuAllocatorBase {
public:
    void* allocate(size_type num_bytes) const override;

    void deallocate(void* ptr) const override;

    CudaHostAllocator(int device_id);

private:
    int device_id_;
};


/*
 * Allocator using hipMalloc.
 */
class HipAllocator : public HipAllocatorBase {
public:
    void* allocate(size_type num_bytes) const override;

    void deallocate(void* ptr) const override;
};


/*
 * Allocator using sycl::malloc_device.
 */
class DpcppAllocator : public DpcppAllocatorBase {
public:
    using DpcppAllocatorBase::DpcppAllocatorBase;

protected:
    void* allocate_impl(sycl::queue* queue, size_type num_bytes) const override;

    void deallocate_impl(sycl::queue* queue, void* ptr) const override;
};


/*
 * Allocator using sycl::malloc_shared.
 */
class DpcppUnifiedAllocator : public DpcppAllocatorBase,
                              public CpuAllocatorBase {
public:
    using DpcppAllocatorBase::DpcppAllocatorBase;

protected:
    void* allocate_impl(sycl::queue* queue, size_type num_bytes) const override;

    void deallocate_impl(sycl::queue* queue, void* ptr) const override;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_MEMORY_HPP_
