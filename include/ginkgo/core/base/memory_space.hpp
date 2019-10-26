/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_CORE_MEMORY_SPACE_HPP_
#define GKO_CORE_MEMORY_SPACE_HPP_


#include <memory>
#include <mutex>
#include <sstream>
#include <tuple>
#include <type_traits>


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>


namespace gko {


class HostMemorySpace;
class CudaMemorySpace;


namespace detail {


template <typename>
class MemorySpaceBase;


}  // namespace detail


class MemorySpace : public log::EnableLogging<MemorySpace> {
    template <typename T>
    friend class detail::MemorySpaceBase;

public:
    virtual ~MemorySpace() = default;

    MemorySpace() = default;
    MemorySpace(MemorySpace &) = delete;
    MemorySpace(MemorySpace &&) = default;
    MemorySpace &operator=(MemorySpace &) = delete;
    MemorySpace &operator=(MemorySpace &&) = default;

    /**
     * Allocates memory in this MemorySpace.
     *
     * @tparam T  datatype to allocate
     *
     * @param num_elems  number of elements of type T to allocate
     *
     * @throw AllocationError  if the allocation failed
     *
     * @return pointer to allocated memory
     */
    template <typename T>
    T *alloc(size_type num_elems) const
    {
        this->template log<log::Logger::allocation_started>(
            this, num_elems * sizeof(T));
        T *allocated = static_cast<T *>(this->raw_alloc(num_elems * sizeof(T)));
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
    void free(void *ptr) const noexcept
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
    void copy_from(const MemorySpace *src_mem_space, size_type num_elems,
                   const T *src_ptr, T *dest_ptr) const
    {
        this->template log<log::Logger::copy_started>(
            src_mem_space, this, reinterpret_cast<uintptr>(src_ptr),
            reinterpret_cast<uintptr>(dest_ptr), num_elems * sizeof(T));
        this->raw_copy_from(src_mem_space, num_elems * sizeof(T), src_ptr,
                            dest_ptr);
        this->template log<log::Logger::copy_completed>(
            src_mem_space, this, reinterpret_cast<uintptr>(src_ptr),
            reinterpret_cast<uintptr>(dest_ptr), num_elems * sizeof(T));
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
    virtual void *raw_alloc(size_type size) const;

    /**
     * Frees memory previously allocated with MemorySpace::alloc().
     *
     * If `ptr` is a `nullptr`, the function has no effect.
     *
     * @param ptr  pointer to the allocated memory block
     */
    virtual void raw_free(void *ptr) const noexcept;

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
    virtual void raw_copy_from(
        const MemorySpace *src_mem_space, size_type n_bytes,
        const void *src_ptr,
        void *dest_ptr) const;  // Change to Pure virtual ? TODO

/**
 * @internal
 * Declares a raw_copy_to() overload for a specified MemorySpace subclass.
 *
 * This is the second stage of the double dispatch emulation required to
 * implement raw_copy_from().
 *
 * @param _mem_space_type  the MemorySpace subclass
 */
#define GKO_ENABLE_RAW_COPY_TO(_mem_space_type, ...)                 \
    virtual void raw_copy_to(const _mem_space_type *dest_mem_space,  \
                             size_type n_bytes, const void *src_ptr, \
                             void *dest_ptr) const

    GKO_ENABLE_FOR_ALL_MEMORY_SPACES(GKO_ENABLE_RAW_COPY_TO);

#undef GKO_ENABLE_RAW_COPY_TO
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
    using pointer = T *;

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
public:
    void raw_copy_from(const MemorySpace *src_mem_space, size_type n_bytes,
                       const void *src_ptr, void *dest_ptr) const override
    {
        src_mem_space->raw_copy_to(self(), n_bytes, src_ptr, dest_ptr);
    }

private:
    ConcreteMemorySpace *self() noexcept
    {
        return static_cast<ConcreteMemorySpace *>(this);
    }

    const ConcreteMemorySpace *self() const noexcept
    {
        return static_cast<const ConcreteMemorySpace *>(this);
    }
};


}  // namespace detail


#define GKO_OVERRIDE_RAW_COPY_TO(_memory_space_type, ...)                    \
    void raw_copy_to(const _memory_space_type *dest_mem_space,               \
                     size_type n_bytes, const void *src_ptr, void *dest_ptr) \
        const override


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

protected:
    HostMemorySpace() = default;

    void *raw_alloc(size_type size) const override;

    void raw_free(void *ptr) const noexcept override;

    GKO_ENABLE_FOR_ALL_MEMORY_SPACES(GKO_OVERRIDE_RAW_COPY_TO);
};


class CudaMemorySpace : public detail::MemorySpaceBase<CudaMemorySpace> {
    friend class detail::MemorySpaceBase<CudaMemorySpace>;

public:
    /**
     * Creates a new CudaMemorySpace.
     *
     * @param device_id  the CUDA device id of this device
     * @param master  an executor on the host that is used to invoke the device
     * kernels
     */
    static std::shared_ptr<CudaMemorySpace> create(int device_id)
    {
        return std::shared_ptr<CudaMemorySpace>(new CudaMemorySpace(device_id));
    }
    /**
     * Get the CUDA device id of the device associated to this memory_space.
     */
    int get_device_id() const noexcept { return device_id_; }

protected:
    CudaMemorySpace() = default;

    CudaMemorySpace(int device_id) : device_id_(device_id) {}

    void *raw_alloc(size_type size) const override;

    void raw_free(void *ptr) const noexcept override;

    GKO_ENABLE_FOR_ALL_MEMORY_SPACES(GKO_OVERRIDE_RAW_COPY_TO);

private:
    int device_id_;
};

#undef GKO_OVERRIDE_RAW_COPY_TO


}  // namespace gko


#endif  // GKO_CORE_MEMORY_SPACE_HPP_
