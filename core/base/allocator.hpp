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

#ifndef GKO_CORE_BASE_ALLOCATOR_HPP_
#define GKO_CORE_BASE_ALLOCATOR_HPP_


#include <ginkgo/core/base/executor.hpp>
#include <memory>
#include <type_traits>
#include <vector>


namespace gko {

/**
 * @internal
 *
 * C++ standard library-compatible allocator that uses an executor for
 * allocations.
 *
 * @tparam T  the type of the allocated elements.
 */
template <typename T>
class ExecutorAllocator {
public:
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    /**
     * Constructs an allocator from a given executor.
     *
     * This function works with both const and non-const ExecType,
     * as long as it is derived from gko::Executor.
     * @param exec  the executor
     * @tparam ExecType  the static type of the executor
     */
    template <typename ExecType>
    ExecutorAllocator(std::shared_ptr<ExecType> exec) : exec_{std::move(exec)}
    {}

    /**
     * Constructs an allocator for another element type from a given executor.
     *
     * This is related to `std::allocator_traits::template rebind<U>` and its
     * use in more advanced data structures.
     *
     * @param other  the other executor
     * @tparam U  the element type of the allocator to be constructed.
     */
    template <typename U>
    explicit ExecutorAllocator(const ExecutorAllocator<U> &other)
        : exec_{other.get_executor()}
    {}

    /** Returns the executor used by this allocator.  */
    std::shared_ptr<const Executor> get_executor() const { return exec_; }

    /**
     * Allocates a memory area of the given size.
     *
     * @param n  the number of elements to allocate
     * @return  the pointer to a newly allocated memory area of `n` elements.
     */
    T *allocate(std::size_t n) const { return exec_->alloc<T>(n); }

    /**
     * Frees a memory area that was allocated by this allocator.
     *
     * @param ptr  The memory area to free, previously returned by `allocate`.
     *
     * @note  The second parameter is unused.
     */
    void deallocate(T *ptr, std::size_t) const { exec_->free(ptr); }

    /**
     * Compares two ExecutorAllocators for equality
     *
     * @param l  the first allocator
     * @param r  the second allocator
     * @return true iff the two allocators use the same executor
     */
    template <typename T2>
    friend bool operator==(const ExecutorAllocator<T> &l,
                           const ExecutorAllocator<T2> &r)
    {
        return l.get_executor() == r.get_executor();
    }

    /**
     * Compares two ExecutorAllocators for inequality
     *
     * @param l  the first allocator
     * @param r  the second allocator
     * @return true iff the two allocators use different executors
     */
    template <typename T2>
    friend bool operator!=(const ExecutorAllocator<T> &l,
                           const ExecutorAllocator<T2> &r)
    {
        return !(l == r);
    }

private:
    std::shared_ptr<const Executor> exec_;
};


// Convenience type alias
template <typename T>
using vector = std::vector<T, ExecutorAllocator<T>>;


}  // namespace gko

#endif  // GKO_CORE_BASE_ALLOCATOR_HPP_