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

    virtual void* allocate(size_type num_bytes) = 0;

    virtual void deallocate(void* ptr) = 0;
};


/**
 * Implement this interface to provide an allocator for OmpExecutor or
 * ReferenceExecutor.
 */
class CpuAllocatorBase : public Allocator {};


/**
 * Implement this interface to provide an allocator for CudaExecutor.
 */
class CudaAllocatorBase : public Allocator {
    friend class CudaExecutor;

protected:
    /**
     * Checks if the allocator can be used safely with the provided device ID
     * and stream. The check is necessary to ensure safe usage of stream-ordered
     * allocators and unified shared memory allocators.
     *
     * @param device_id the device ID the allocator will be used in.
     * @param stream the stream the allocator will be used with.
     *
     * @return true if and only if the allocator can be used by CudaExecutor in
     *         the given environment.
     */
    virtual bool check_environment(int device_id, CUstream_st* stream) const
    {
        return true;
    }
};


/**
 * Implement this interface to provide an allocator for HipExecutor.
 */
class HipAllocatorBase : public Allocator {
    friend class HipExecutor;

protected:
    /**
     * Checks if the allocator can be used safely with the provided device ID
     * and stream. The check is necessary to ensure safe usage of stream-ordered
     * allocators and unified shared memory allocators.
     *
     * @param device_id the device ID the allocator will be used in.
     * @param stream the stream the allocator will be used with.
     *
     * @return true if and only if the allocator can be used by HipExecutor in
     *         the given environment.
     */
    virtual bool check_environment(int device_id,
                                   GKO_HIP_STREAM_STRUCT* stream) const
    {
        return true;
    }
};


/**
 * Allocator using new/delete.
 */
class CpuAllocator : public CpuAllocatorBase {
public:
    void* allocate(size_type num_bytes) override;

    void deallocate(void* ptr) override;
};


/**
 * Allocator using cudaMalloc.
 */
class CudaAllocator : public CudaAllocatorBase {
public:
    void* allocate(size_type num_bytes) override;

    void deallocate(void* ptr) override;
};


/*
 * Allocator using cudaMallocAsync.
 */
class CudaAsyncAllocator : public CudaAllocatorBase {
public:
    void* allocate(size_type num_bytes) override;

    void deallocate(void* ptr) override;

    CudaAsyncAllocator(CUstream_st* stream);

    bool check_environment(int device_id, CUstream_st* stream) const override;

private:
    CUstream_st* stream_;
};


/*
 * Allocator using cudaMallocManaged
 */
class CudaUnifiedAllocator : public CudaAllocatorBase, public CpuAllocatorBase {
public:
    void* allocate(size_type num_bytes) override;

    void deallocate(void* ptr) override;

    CudaUnifiedAllocator(int device_id);

    CudaUnifiedAllocator(int device_id, unsigned int flags);

protected:
    bool check_environment(int device_id, CUstream_st* stream) const override;

private:
    int device_id_;
    unsigned int flags_;
};


/*
 * Allocator using cudaHostMalloc.
 */
class CudaHostAllocator : public CudaAllocatorBase, public CpuAllocatorBase {
public:
    void* allocate(size_type num_bytes) override;

    void deallocate(void* ptr) override;

    CudaHostAllocator(int device_id);

protected:
    bool check_environment(int device_id, CUstream_st* stream) const override;

private:
    int device_id_;
};


/*
 * Allocator using hipMalloc.
 */
class HipAllocator : public HipAllocatorBase {
public:
    void* allocate(size_type num_bytes) override;

    void deallocate(void* ptr) override;
};


/*
 * Allocator using hipMallocAsync.
 */
class HipAsyncAllocator : public HipAllocatorBase {
public:
    void* allocate(size_type num_bytes) override;

    void deallocate(void* ptr) override;

    HipAsyncAllocator(GKO_HIP_STREAM_STRUCT* stream);

protected:
    bool check_environment(int device_id,
                           GKO_HIP_STREAM_STRUCT* stream) const override;

private:
    GKO_HIP_STREAM_STRUCT* stream_;
};


/*
 * Allocator using hipMallocManaged
 */
class HipUnifiedAllocator : public HipAllocatorBase, public CpuAllocatorBase {
public:
    void* allocate(size_type num_bytes) override;

    void deallocate(void* ptr) override;

    HipUnifiedAllocator(int device_id);

    HipUnifiedAllocator(int device_id, unsigned int flags);

protected:
    bool check_environment(int device_id,
                           GKO_HIP_STREAM_STRUCT* stream) const override;

private:
    int device_id_;
    unsigned int flags_;
};


/*
 * Allocator using hipHostAlloc.
 */
class HipHostAllocator : public HipAllocatorBase, public CpuAllocatorBase {
public:
    void* allocate(size_type num_bytes) override;

    void deallocate(void* ptr) override;

    HipHostAllocator(int device_id);

protected:
    bool check_environment(int device_id,
                           GKO_HIP_STREAM_STRUCT* stream) const override;

private:
    int device_id_;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_MEMORY_HPP_
