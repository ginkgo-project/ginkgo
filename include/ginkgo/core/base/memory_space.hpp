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

#ifndef GKO_CORE_BASE_MEMORY_SPACE_HPP_
#define GKO_CORE_BASE_MEMORY_SPACE_HPP_


#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>


#include <ginkgo/core/base/async_handle.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>


inline namespace cl {
namespace sycl {

class queue;

}  // namespace sycl
}  // namespace cl


namespace gko {


#define GKO_FORWARD_DECLARE(_type, ...) class _type

GKO_ENABLE_FOR_ALL_MEMORY_SPACES(GKO_FORWARD_DECLARE);

#undef GKO_FORWARD_DECLARE


namespace detail {


template <typename>
class MemorySpaceBase;


}  // namespace detail

#define GKO_DECLARE_MEMSPACE_FRIEND(_type, ...) friend class _type

class MemorySpace : public log::EnableLogging<MemorySpace> {
    template <typename T>
    friend class detail::MemorySpaceBase;

    GKO_ENABLE_FOR_ALL_MEMORY_SPACES(GKO_DECLARE_MEMSPACE_FRIEND);

public:
    virtual ~MemorySpace() = default;

    MemorySpace() = default;
    MemorySpace(MemorySpace&) = delete;
    MemorySpace(MemorySpace&&) = default;
    MemorySpace& operator=(MemorySpace&) = delete;
    MemorySpace& operator=(MemorySpace&&) = default;

    /**
     * Allocates memory in this MemorySpace.
     *
     * @tparam T datatype to allocate
     *
     * @param num_elems number of elements of type T to allocate
     *
     * @throw AllocationError if the allocation failed *
     *
     * @return pointer to allocated memory
     */
    template <typename T>
    T* alloc(size_type num_elems) const
    {
        this->template log<log::Logger::allocation_started>(
            this, num_elems * sizeof(T));
        T* allocated = static_cast<T*>(this->raw_alloc(num_elems * sizeof(T)));
        this->template log<log::Logger::allocation_completed>(
            this, num_elems * sizeof(T), reinterpret_cast<uintptr>(allocated));
        return allocated;
    }

    /**
     * Frees memory previously allocated with MemorySpace::alloc().
     *
     * If `ptr` is a `nullptr`, the function has no effect.
     *
     * @param ptr  pointer to the allocated memory block
     */
    void free(void* ptr) const noexcept
    {
        this->template log<log::Logger::free_started>(
            this, reinterpret_cast<uintptr>(ptr));
        this->raw_free(ptr);
        this->template log<log::Logger::free_completed>(
            this, reinterpret_cast<uintptr>(ptr));
    }

    /**
     * Copies data from another MemorySpace.
     *
     * @tparam T  datatype to copy
     *
     * @param src_mem_space  MemorySpace from which the memory will be copied
     * @param num_elems  number of elements of type T to copy
     * @param src_ptr  pointer to a block of memory containing the data to be
     *                 copied
     * @param dest_ptr  pointer to an allocated block of memory
     *                  where the data will be copied to
     */
    template <typename T>
    std::shared_ptr<AsyncHandle> copy_from(const MemorySpace* src_mem_space,
                                           size_type num_elems,
                                           const T* src_ptr, T* dest_ptr) const
    {
        this->template log<log::Logger::copy_started>(
            src_mem_space, this, reinterpret_cast<uintptr>(src_ptr),
            reinterpret_cast<uintptr>(dest_ptr), num_elems * sizeof(T));
        // try {
        return this->raw_copy_from(src_mem_space, num_elems * sizeof(T),
                                   src_ptr, dest_ptr);
        // TODO Find a nice way to fix this
        //         } catch (NotSupported&) {
        // #if (GKO_VERBOSE_LEVEL >= 1) && !defined(NDEBUG)
        //             // Unoptimized copy. Try to go through the masters.
        //             // output to log when verbose >= 1 and debug build
        //             std::clog << "Not direct copy. Try to copy data from the
        //             masters."
        //                       << std::endl;
        // #endif
        //             auto src_host = HostMemorySpace::create();
        //             if (num_elems > 0 &&
        //                 (dynamic_cast<HostMemorySpace>(src_mem_space) ==
        //                 nullptr)) { auto* host_ptr =
        //                 src_host->alloc<T>(num_elems);
        //                 src_host->copy_from<T>(src_mem_space, num_elems,
        //                 src_ptr,
        //                                        host_ptr);
        //                 this->copy_from<T>(src_host, num_elems, host_ptr,
        //                 dest_ptr); src_host->free(host_ptr);
        //             }
        //         }
        // TODO Find a nice way to fix this
        // this->template log<log::Logger::copy_completed>(
        //     src_mem_space, this, reinterpret_cast<uintptr>(src_ptr),
        //     reinterpret_cast<uintptr>(dest_ptr), num_elems * sizeof(T));
    }

    /**
     * Verifies whether the memory spaces share the same memory.
     *
     * @param other  the other MemorySpace to compare against
     *
     * @return whether the memory spaces this and other share the same memory.
     */
    bool memory_accessible(
        const std::shared_ptr<const MemorySpace>& other) const
    {
        return this->verify_memory_from(other.get());
    }

    /**
     * Synchronize the operations launched on the mem_space with its master.
     */
    virtual void synchronize() const = 0;

    std::shared_ptr<AsyncHandle> get_default_input_stream() const
    {
        return this->default_input_stream_;
    }

    std::shared_ptr<AsyncHandle> get_default_output_stream() const
    {
        return this->default_output_stream_;
    }

protected:
    /**
     * Allocates raw memory in this MemorySpace.
     *
     * @param size  number of bytes to allocate
     *
     * @throw AllocationError  if the allocation failed
     *
     * @return raw pointer to allocated memory
     */
    virtual void* raw_alloc(size_type size) const = 0;

    /**
     * Frees memory previously allocated with MemorySpace::alloc().
     *
     * If `ptr` is a `nullptr`, the function has no effect.
     *
     * @param ptr  pointer to the allocated memory block
     */
    virtual void raw_free(void* ptr) const noexcept = 0;

    /**
     * Copies raw data from another MemorySpace.
     *
     * @param src_mem_space  MemorySpace from which the memory will be copied
     * @param n_bytes  number of bytes to copy
     * @param src_ptr  pointer to a block of memory containing the data to be
     *                 copied
     * @param dest_ptr  pointer to an allocated block of memory where the data
     *                  will be copied to
     */
    virtual std::shared_ptr<AsyncHandle> raw_copy_from(
        const MemorySpace* src_mem_space, size_type n_bytes,
        const void* src_ptr, void* dest_ptr) const = 0;

/**
 * @internal
 * Declares a raw_copy_to() overload for a specified MemorySpace subclass.
 *
 * This is the second stage of the double dispatch emulation required to
 * implement raw_copy_from().
 *
 * @param _mem_space_type  the MemorySpace subclass
 */
#define GKO_ENABLE_RAW_COPY_TO(_mem_space_type, ...)              \
    virtual std::shared_ptr<AsyncHandle> raw_copy_to(             \
        const _mem_space_type* dest_mem_space, size_type n_bytes, \
        const void* src_ptr, void* dest_ptr) const = 0

    GKO_ENABLE_FOR_ALL_MEMORY_SPACES(GKO_ENABLE_RAW_COPY_TO);

#undef GKO_ENABLE_RAW_COPY_TO


    /**
     * Verify the memory from another MemorySpace.
     *
     * @param src_mem_space  MemorySpace from which to verify the memory.
     *
     * @return whether this mem_space and src_mem_space share the same
     * memory.
     */
    virtual bool verify_memory_from(const MemorySpace* src_mem_space) const = 0;

/**
 * @internal
 * Declares a verify_memory_to() overload for a specified MemorySpace subclass.
 *
 * This is the second stage of the double dispatch emulation required to
 * implement verify_memory_from().
 *
 * @param _mem_space_type  the MemorySpace subclass
 */
#define GKO_ENABLE_VERIFY_MEMORY_TO(_mem_space_type, ...)                \
    virtual bool verify_memory_to(const _mem_space_type* dest_mem_space) \
        const = 0

    GKO_ENABLE_FOR_ALL_MEMORY_SPACES(GKO_ENABLE_VERIFY_MEMORY_TO);

#undef GKO_ENABLE_VERIFY_MEMORY_TO

    std::shared_ptr<AsyncHandle> default_input_stream_;
    std::shared_ptr<AsyncHandle> default_output_stream_;
};


/**
 * This is a deleter that uses an mem_space's `free` method to deallocate
 * the data.
 *
 * @tparam T  the type of object being deleted
 *
 * @ingroup MemorySpace
 */
template <typename T>
class memory_space_deleter {
public:
    using pointer = T*;

    /**
     * Creates a new deleter.
     *
     * @param mem_space  the mem_spaceutor used to free the data
     */
    explicit memory_space_deleter(std::shared_ptr<const MemorySpace> mem_space)
        : mem_space_{mem_space}
    {}

    /**
     * Deletes the object.
     *
     * @param ptr  pointer to the object being deleted
     */
    void operator()(pointer ptr) const
    {
        if (mem_space_) {
            mem_space_->free(ptr);
        }
    }

private:
    std::shared_ptr<const MemorySpace> mem_space_;
};


// a specialization for arrays
template <typename T>
class memory_space_deleter<T[]> {
public:
    using pointer = T[];

    explicit memory_space_deleter(std::shared_ptr<const MemorySpace> mem_space)
        : mem_space_{mem_space}
    {}

    void operator()(pointer ptr) const
    {
        if (mem_space_) {
            mem_space_->free(ptr);
        }
    }

private:
    std::shared_ptr<const MemorySpace> mem_space_;
};


namespace detail {


template <typename ConcreteMemorySpace>
class MemorySpaceBase : public MemorySpace {
    GKO_ENABLE_FOR_ALL_MEMORY_SPACES(GKO_DECLARE_MEMSPACE_FRIEND);

public:
    std::shared_ptr<AsyncHandle> raw_copy_from(const MemorySpace* src_mem_space,
                                               size_type n_bytes,
                                               const void* src_ptr,
                                               void* dest_ptr) const override
    {
        return src_mem_space->raw_copy_to(self(), n_bytes, src_ptr, dest_ptr);
    }

private:
    ConcreteMemorySpace* self() noexcept
    {
        return static_cast<ConcreteMemorySpace*>(this);
    }

    const ConcreteMemorySpace* self() const noexcept
    {
        return static_cast<const ConcreteMemorySpace*>(this);
    }
};

#undef GKO_DECLARE_MEMSPACE_FRIEND


}  // namespace detail


#define GKO_OVERRIDE_RAW_COPY_TO(_memory_space_type, ...)            \
    std::shared_ptr<AsyncHandle> raw_copy_to(                        \
        const _memory_space_type* dest_mem_space, size_type n_bytes, \
        const void* src_ptr, void* dest_ptr) const override


#define GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(dest_, bool_)                     \
    virtual bool verify_memory_to(const dest_* other) const override         \
    {                                                                        \
        return bool_;                                                        \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


class HostMemorySpace : public detail::MemorySpaceBase<HostMemorySpace> {
    friend class detail::MemorySpaceBase<HostMemorySpace>;

public:
    /**
     * Creates a new HostMemorySpace.
     */
    static std::shared_ptr<HostMemorySpace> create()
    {
        return std::shared_ptr<HostMemorySpace>(new HostMemorySpace());
    }

    void synchronize() const override;

protected:
    HostMemorySpace()
    {
        this->default_input_stream_ = {};
        this->default_output_stream_ = {};
    }

    void* raw_alloc(size_type size) const override;

    void raw_free(void* ptr) const noexcept override;

    bool verify_memory_from(const MemorySpace* src_mem_space) const override
    {
        return src_mem_space->verify_memory_to(this);
    }

    GKO_ENABLE_FOR_ALL_MEMORY_SPACES(GKO_OVERRIDE_RAW_COPY_TO);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(ReferenceMemorySpace, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(HostMemorySpace, true);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(CudaMemorySpace, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(CudaUVMSpace, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(HipMemorySpace, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(DpcppMemorySpace, false);
};


class ReferenceMemorySpace
    : public detail::MemorySpaceBase<ReferenceMemorySpace> {
    friend class detail::MemorySpaceBase<ReferenceMemorySpace>;

public:
    /**
     * Creates a new ReferenceMemorySpace.
     */
    static std::shared_ptr<ReferenceMemorySpace> create()
    {
        return std::shared_ptr<ReferenceMemorySpace>(
            new ReferenceMemorySpace());
    }

    void synchronize() const override;

protected:
    ReferenceMemorySpace()
    {
        this->default_input_stream_ = {};
        this->default_output_stream_ = {};
    }

    void* raw_alloc(size_type size) const override;

    void raw_free(void* ptr) const noexcept override;

    bool verify_memory_from(const MemorySpace* src_mem_space) const override
    {
        return src_mem_space->verify_memory_to(this);
    }

    GKO_ENABLE_FOR_ALL_MEMORY_SPACES(GKO_OVERRIDE_RAW_COPY_TO);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(HostMemorySpace, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(ReferenceMemorySpace, true);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(CudaMemorySpace, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(CudaUVMSpace, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(HipMemorySpace, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(DpcppMemorySpace, false);
};


class CudaMemorySpace : public detail::MemorySpaceBase<CudaMemorySpace> {
    friend class detail::MemorySpaceBase<CudaMemorySpace>;

public:
    /**
     * Creates a new CudaMemorySpace.
     *
     * @param device_id  the CUDA device id of this device
     */
    static std::shared_ptr<CudaMemorySpace> create(int device_id)
    {
        return std::shared_ptr<CudaMemorySpace>(new CudaMemorySpace(device_id));
    }

    /**
     * Get the CUDA device id of the device associated to this memory_space.
     */
    int get_device_id() const noexcept { return this->device_id_; }

    /**
     * Get the number of devices present on the system.
     */
    static int get_num_devices();

    void synchronize() const override;

protected:
    CudaMemorySpace()
    {
        this->default_input_stream_ = {};
        this->default_output_stream_ = {};
    }

    CudaMemorySpace(int device_id);

    void* raw_alloc(size_type size) const override;

    void raw_free(void* ptr) const noexcept override;

    GKO_ENABLE_FOR_ALL_MEMORY_SPACES(GKO_OVERRIDE_RAW_COPY_TO);

    bool verify_memory_from(const MemorySpace* src_mem_space) const override
    {
        return src_mem_space->verify_memory_to(this);
    }

    bool verify_memory_to(const HipMemorySpace* dest_mem_space) const override;

    bool verify_memory_to(const CudaMemorySpace* dest_mem_space) const override;

    bool verify_memory_to(const CudaUVMSpace* dest_mem_space) const override;

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(ReferenceMemorySpace, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(HostMemorySpace, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(DpcppMemorySpace, false);

private:
    int device_id_;
    static constexpr int max_devices = 64;
};


class CudaUVMSpace : public detail::MemorySpaceBase<CudaUVMSpace> {
    friend class detail::MemorySpaceBase<CudaUVMSpace>;

public:
    /**
     * Creates a new CudaUVMSpace.
     *
     * @param device_id  the CUDA device id of this device
     */
    static std::shared_ptr<CudaUVMSpace> create(int device_id)
    {
        return std::shared_ptr<CudaUVMSpace>(new CudaUVMSpace(device_id));
    }

    /**
     * Get the CUDA device id of the device associated to this memory_space.
     */
    int get_device_id() const noexcept { return this->device_id_; }

    /**
     * Get the number of devices present on the system.
     */
    static int get_num_devices();

    void synchronize() const override;

protected:
    CudaUVMSpace()
    {
        this->default_input_stream_ = {};
        this->default_output_stream_ = {};
    }

    CudaUVMSpace(int device_id);

    void* raw_alloc(size_type size) const override;

    void raw_free(void* ptr) const noexcept override;

    GKO_ENABLE_FOR_ALL_MEMORY_SPACES(GKO_OVERRIDE_RAW_COPY_TO);

    bool verify_memory_from(const MemorySpace* src_mem_space) const override
    {
        return src_mem_space->verify_memory_to(this);
    }

    bool verify_memory_to(const HipMemorySpace* dest_mem_space) const override;

    bool verify_memory_to(const CudaMemorySpace* dest_mem_space) const override;

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(ReferenceMemorySpace, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(CudaUVMSpace, true);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(HostMemorySpace, true);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(DpcppMemorySpace, false);

private:
    int device_id_;
    static constexpr int max_devices = 64;
};


class HipMemorySpace : public detail::MemorySpaceBase<HipMemorySpace> {
    friend class detail::MemorySpaceBase<HipMemorySpace>;

public:
    /**
     * Creates a new HipMemorySpace.
     *
     * @param device_id  the HIP device id of this device
     */
    static std::shared_ptr<HipMemorySpace> create(int device_id)
    {
        return std::shared_ptr<HipMemorySpace>(new HipMemorySpace(device_id));
    }

    /**
     * Get the HIP device id of the device associated to this memory_space.
     */
    int get_device_id() const noexcept { return this->device_id_; }

    /**
     * Get the number of devices present on the system.
     */
    static int get_num_devices();

    void synchronize() const override;

protected:
    HipMemorySpace()
    {
        this->default_input_stream_ = {};
        this->default_output_stream_ = {};
    }

    HipMemorySpace(int device_id);

    void* raw_alloc(size_type size) const override;

    void raw_free(void* ptr) const noexcept override;

    GKO_ENABLE_FOR_ALL_MEMORY_SPACES(GKO_OVERRIDE_RAW_COPY_TO);

    bool verify_memory_from(const MemorySpace* src_mem_space) const override
    {
        return src_mem_space->verify_memory_to(this);
    }

    bool verify_memory_to(const CudaMemorySpace* dest_mem_space) const override;

    bool verify_memory_to(const CudaUVMSpace* dest_mem_space) const override;

    bool verify_memory_to(const HipMemorySpace* dest_mem_space) const override;

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(ReferenceMemorySpace, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(HostMemorySpace, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(DpcppMemorySpace, false);

private:
    int device_id_;
    static constexpr int max_devices = 64;
};


class DpcppMemorySpace : public detail::MemorySpaceBase<DpcppMemorySpace> {
    friend class detail::MemorySpaceBase<DpcppMemorySpace>;

public:
    /**
     * Creates a new DpcppMemorySpace.
     *
     * @param device_id  the DPCPP device id of this device
     */
    static std::shared_ptr<DpcppMemorySpace> create(
        int device_id, std::string device_type = "all")
    {
        return std::shared_ptr<DpcppMemorySpace>(
            new DpcppMemorySpace(device_id, device_type));
    }

    ::cl::sycl::queue* get_queue() const { return queue_.get(); }

    /**
     * Get the DPCPP device id of the device associated to this memory_space.
     */
    int get_device_id() const noexcept { return this->device_id_; }

    /**
     * Get a string representing the device type.
     *
     * @return a string representing the device type
     */
    std::string get_device_type() const noexcept { return this->device_type_; }

    /**
     * Get the number of devices present on the system.
     *
     * @param device_type  a string representing the device type
     *
     * @return the number of devices present on the system
     */
    static int get_num_devices(std::string device_type);

    void synchronize() const override;

protected:
    DpcppMemorySpace() = default;

    DpcppMemorySpace(int device_id, std::string device_type);

    void* raw_alloc(size_type size) const override;

    void raw_free(void* ptr) const noexcept override;

    GKO_ENABLE_FOR_ALL_MEMORY_SPACES(GKO_OVERRIDE_RAW_COPY_TO);

    bool verify_memory_from(const MemorySpace* src_mem_space) const override
    {
        return src_mem_space->verify_memory_to(this);
    }

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(ReferenceMemorySpace, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(CudaMemorySpace, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(CudaUVMSpace, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(HipMemorySpace, false);

    bool verify_memory_to(const HostMemorySpace* dest_mem_space) const override;

    bool verify_memory_to(
        const DpcppMemorySpace* dest_mem_space) const override;

private:
    int device_id_;
    std::string device_type_;
    static constexpr int max_devices = 64;

    template <typename T>
    using queue_manager = std::unique_ptr<T, std::function<void(T*)>>;
    queue_manager<::cl::sycl::queue> queue_;
};

#undef GKO_OVERRIDE_RAW_COPY_TO


}  // namespace gko


#endif  // GKO_CORE_MEMORY_SPACE_HPP_
