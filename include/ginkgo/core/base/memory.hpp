// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_MEMORY_HPP_
#define GKO_PUBLIC_CORE_BASE_MEMORY_HPP_


#include <memory>
#include <string>

#include <ginkgo/core/base/fwd_decls.hpp>
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


/**
 * Common base of the Umpire-backed allocators, which forward to a named
 * allocator in [Umpire](https://github.com/LLNL/Umpire)'s ResourceManager.
 *
 * - Statistics cover everything allocated from the Umpire allocator.
 * - Alignment is not checked (application-built pools: 16 bytes by default).
 * - Not serialized by executors; share a pool across threads only if it is
 *   thread-safe (umpire_pool_config::thread_safe).
 * - Each process, hence each MPI rank, has its own ResourceManager and pools.
 * - The accessors are deliberately not noexcept: without Umpire they throw
 *   NotCompiled.
 *
 * @warning Pools are not stream-ordered: deallocate() frees immediately while
 *          Ginkgo's kernels may still use the memory. Call
 *          Executor::synchronize() before sharing a device pool across streams.
 */
class UmpireAllocatorBase {
public:
    /** Name of the underlying Umpire allocator. */
    const std::string& get_name() const;

    /**
     * Bytes currently allocated from the Umpire allocator and not yet
     * deallocated, by any of its users (not only through this object).
     */
    size_type get_current_size() const;

    /**
     * Bytes the Umpire allocator holds from its memory resource. For a pool,
     * this includes reserved memory that is not currently allocated, so it is
     * at least get_current_size().
     */
    size_type get_actual_size() const;

    /** Largest value get_current_size() has reached so far. */
    size_type get_high_watermark() const;

    /** Number of allocations from the Umpire allocator that are still live. */
    size_type get_allocation_count() const;

    /** Returns the pool's completely unused blocks to its memory resource. */
    void release();

protected:
    explicit UmpireAllocatorBase(std::string name);

    ~UmpireAllocatorBase();

    void* allocate_impl(size_type num_bytes);

    void deallocate_impl(void* ptr) noexcept;

    /** True if the Umpire resource is readable from the host. */
    bool is_host_accessible() const noexcept;

    /** True if the Umpire resource is readable from the device. */
    bool is_device_accessible() const noexcept;

    /** True if the resource is not device-bound, or is bound to device_id. */
    bool matches_device(int device_id) const noexcept;

private:
    struct data;
    std::unique_ptr<data> data_;
};


/**
 * Allocator using a named Umpire allocator over a host-accessible resource.
 *
 * @warning Umpire's ResourceManager is a function-local static and this class
 *          holds a non-owning handle into it, so an Executor using it must
 *          not outlive main() (e.g. in a global shared_ptr). The same holds
 *          for UmpireCudaAllocator and UmpireHipAllocator.
 */
class UmpireCpuAllocator : public CpuAllocatorBase, public UmpireAllocatorBase {
public:
    /**
     * @param name  name of an allocator registered with Umpire.
     *
     * @throw Error  if `name` is not registered, or its resource is not
     *               host-accessible.
     * @throw NotCompiled  if Ginkgo was built without Umpire.
     */
    explicit UmpireCpuAllocator(std::string name);

    void* allocate(size_type num_bytes) override;

    void deallocate(void* ptr) override;
};


/**
 * Allocator using a named Umpire allocator over a CUDA device resource.
 * The pool is not stream-ordered; see UmpireAllocatorBase.
 */
class UmpireCudaAllocator : public CudaAllocatorBase,
                            public UmpireAllocatorBase {
public:
    /**
     * @param name  name of an allocator registered with Umpire.
     *
     * @throw Error  if `name` is not registered, or its resource is not
     *               device-accessible.
     * @throw NotCompiled  if Ginkgo was built without Umpire.
     */
    explicit UmpireCudaAllocator(std::string name);

    void* allocate(size_type num_bytes) override;

    void deallocate(void* ptr) override;

protected:
    /** Rejects host-only pools and pools bound to a different device. */
    bool check_environment(int device_id, CUstream_st* stream) const override;
};


/**
 * Allocator using a named Umpire allocator over a HIP device resource.
 * The pool is not stream-ordered; see UmpireAllocatorBase.
 */
class UmpireHipAllocator : public HipAllocatorBase, public UmpireAllocatorBase {
public:
    /**
     * @param name  name of an allocator registered with Umpire.
     *
     * @throw Error  if `name` is not registered, or its resource is not
     *               device-accessible.
     * @throw NotCompiled  if Ginkgo was built without Umpire.
     */
    explicit UmpireHipAllocator(std::string name);

    void* allocate(size_type num_bytes) override;

    void deallocate(void* ptr) override;

protected:
    bool check_environment(int device_id,
                           GKO_HIP_STREAM_STRUCT* stream) const override;
};


/**
 * Configuration for a pool created by make_umpire_cpu_pool(),
 * make_umpire_cuda_pool() or make_umpire_hip_pool(). The pool stays registered
 * with Umpire, and keeps its reserved memory, until release() is called or
 * the program exits.
 */
struct umpire_pool_config {
    /**
     * Umpire memory resource to pool over; empty means "HOST" for CPU and
     * "DEVICE" for CUDA/HIP. May be device-qualified ("DEVICE::1").
     *
     * A plain "DEVICE" binds to whichever device is current when Umpire first
     * creates it, so a mismatched executor throws. Multi-GPU users should pass
     * "DEVICE::<id>".
     */
    std::string resource = "";

    /**
     * Size in bytes of the pool's first block, requested on first allocation.
     * Each MPI rank has its own pool, so divide the budget by the number of
     * ranks sharing a GPU or node.
     */
    size_type initial_block_size = size_type{512} << 20;

    /** Minimum bytes per subsequent growth. */
    size_type growth_block_size = size_type{64} << 20;

    /**
     * Allocation alignment, matching cudaMalloc/hipMalloc rather than
     * Umpire's 16-byte default. Must be a nonzero power of two. Applies only
     * to pools created by make_umpire_*_pool().
     */
    size_type alignment = 256;

    /**
     * Register each allocation in Umpire's tracking registry. If false, the
     * size/watermark/count getters report 0, and deallocating a foreign
     * pointer is undefined behaviour instead of a reported error.
     */
    bool tracked = true;

    /**
     * Wrap the pool in Umpire's ThreadSafeAllocator. Set this if the pool is
     * used from several threads. The unwrapped pool is registered as
     * `<name>_pool`, and `name` refers to the wrapper.
     */
    bool thread_safe = false;
};


/**
 * Creates a named Umpire QuickPool over a host resource and wraps it.
 *
 * If `name` is already registered, the existing allocator is reused and
 * `config` is ignored. Not thread-safe for concurrent calls with the same
 * `name`.
 *
 * @throw NotCompiled  if Ginkgo was built without Umpire.
 * @throw Error  if `name` is registered but not host-accessible; if
 *               `config.resource` (default "HOST") is not a registered,
 *               host-accessible memory resource; if `config.alignment` is not
 *               a nonzero power of two; or if Umpire fails to create the
 *               pool. The resource and alignment checks run before anything
 *               is registered, so a corrected retry can succeed.
 */
std::shared_ptr<UmpireCpuAllocator> make_umpire_cpu_pool(
    std::string name, umpire_pool_config config = {});


/**
 * Creates a named Umpire QuickPool over a CUDA device resource and wraps it.
 *
 * If `name` is already registered, the existing allocator is reused and
 * `config` is ignored. Not thread-safe for concurrent calls with the same
 * `name`.
 *
 * @throw NotCompiled  if Ginkgo was built without Umpire.
 * @throw Error  if `name` is registered but not device-accessible; if
 *               `config.resource` (default "DEVICE") is not a registered,
 *               device-accessible memory resource; if `config.alignment` is not
 *               a nonzero power of two; or if Umpire fails to create the
 *               pool. The resource and alignment checks run before anything
 *               is registered, so a corrected retry can succeed.
 */
std::shared_ptr<UmpireCudaAllocator> make_umpire_cuda_pool(
    std::string name, umpire_pool_config config = {});


/**
 * Creates a named Umpire QuickPool over a HIP device resource and wraps it.
 *
 * If `name` is already registered, the existing allocator is reused and
 * `config` is ignored. Not thread-safe for concurrent calls with the same
 * `name`.
 *
 * @throw NotCompiled  if Ginkgo was built without Umpire.
 * @throw Error  if `name` is registered but not device-accessible; if
 *               `config.resource` (default "DEVICE") is not a registered,
 *               device-accessible memory resource; if `config.alignment` is not
 *               a nonzero power of two; or if Umpire fails to create the
 *               pool. The resource and alignment checks run before anything
 *               is registered, so a corrected retry can succeed.
 */
std::shared_ptr<UmpireHipAllocator> make_umpire_hip_pool(
    std::string name, umpire_pool_config config = {});


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_MEMORY_HPP_
