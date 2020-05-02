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


#include <map>
#include <memory>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>


#include <ginkgo/core/base/memory_space.hpp>


namespace gko {

/**
 * @internal
 *
 * C++ standard library-compatible allocator that uses an memory space for
 * allocations.
 *
 * @tparam T  the type of the allocated elements.
 */
template <typename T>
class MemorySpaceAllocator {
public:
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    /**
     * Constructs an allocator from a given memory space.
     *
     * This function works with both const and non-const MemspaceType,
     * as long as it is derived from gko::MemorySpace.
     * @param mem_space  the memory space
     * @tparam MemspaceType  the static type of the memory space
     */
    template <typename MemspaceType>
    MemorySpaceAllocator(std::shared_ptr<MemspaceType> mem_space)
        : mem_space_{std::move(mem_space)}
    {}

    /**
     * Constructs an allocator for another element type from a given memory
     * space.
     *
     * This is related to `std::allocator_traits::template rebind<U>` and its
     * use in more advanced data structures.
     *
     * @param other  the other memory space
     * @tparam U  the element type of the allocator to be constructed.
     */
    template <typename U>
    explicit MemorySpaceAllocator(const MemorySpaceAllocator<U> &other)
        : mem_space_{other.get_mem_space()}
    {}

    /** Returns the memory space used by this allocator.  */
    std::shared_ptr<const MemorySpace> get_mem_space() const
    {
        return mem_space_;
    }

    /**
     * Allocates a memory area of the given size.
     *
     * @param n  the number of elements to allocate
     * @return  the pointer to a newly allocated memory area of `n` elements.
     */
    T *allocate(std::size_t n) const { return mem_space_->alloc<T>(n); }

    /**
     * Frees a memory area that was allocated by this allocator.
     *
     * @param ptr  The memory area to free, previously returned by `allocate`.
     *
     * @note  The second parameter is unused.
     */
    void deallocate(T *ptr, std::size_t) const { mem_space_->free(ptr); }

    /**
     * Compares two MemorySpaceAllocators for equality
     *
     * @param l  the first allocator
     * @param r  the second allocator
     * @return true iff the two allocators use the same memory space
     */
    template <typename T2>
    friend bool operator==(const MemorySpaceAllocator<T> &l,
                           const MemorySpaceAllocator<T2> &r)
    {
        return l.get_mem_space() == r.get_mem_space();
    }

    /**
     * Compares two MemorySpaceAllocators for inequality
     *
     * @param l  the first allocator
     * @param r  the second allocator
     * @return true iff the two allocators use different memory spaces
     */
    template <typename T2>
    friend bool operator!=(const MemorySpaceAllocator<T> &l,
                           const MemorySpaceAllocator<T2> &r)
    {
        return !(l == r);
    }

private:
    std::shared_ptr<const MemorySpace> mem_space_;
};


// Convenience type aliases
/** std::vector using an MemorySpaceAllocator. */
template <typename T>
using vector = std::vector<T, MemorySpaceAllocator<T>>;

/** std::set using an MemorySpaceAllocator. */
template <typename Key>
using set = std::set<Key, std::less<Key>, gko::MemorySpaceAllocator<Key>>;

/** std::map using an MemorySpaceAllocator. */
template <typename Key, typename Value>
using map = std::map<Key, Value, std::less<Key>,
                     gko::MemorySpaceAllocator<std::pair<const Key, Value>>>;

/** std::unordered_set using an MemorySpaceAllocator. */
template <typename Key>
using unordered_set =
    std::unordered_set<Key, std::hash<Key>, std::equal_to<Key>,
                       gko::MemorySpaceAllocator<Key>>;

/** std::unordered_map using an MemorySpaceAllocator. */
template <typename Key, typename Value>
using unordered_map =
    std::unordered_map<Key, Value, std::hash<Key>, std::equal_to<Key>,
                       gko::MemorySpaceAllocator<std::pair<const Key, Value>>>;


}  // namespace gko

#endif  // GKO_CORE_BASE_ALLOCATOR_HPP_
