// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_ALLOCATOR_HPP_
#define GKO_CORE_BASE_ALLOCATOR_HPP_


#include <deque>
#include <map>
#include <memory>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>


#include <ginkgo/core/base/executor.hpp>


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
    ExecutorAllocator(std::shared_ptr<ExecType> exec) noexcept
        : exec_{std::move(exec)}
    {}

    /**
     * Copy-constructs an allocator.
     *
     * @param other  the other allocator
     */
    ExecutorAllocator(const ExecutorAllocator& other) noexcept
        : exec_{other.get_executor()}
    {}

    /**
     * Copy-assigns an allocator.
     *
     * @param other  the other allocator
     */
    ExecutorAllocator& operator=(const ExecutorAllocator& other) noexcept
    {
        exec_ = other.get_executor();
        return *this;
    }

    /**
     * Copy-assigns an allocator.
     *
     * This is related to `std::allocator_traits::template rebind<U>` and its
     * use in more advanced data structures.
     *
     * @param other  the other allocator
     * @tparam U  the element type of the allocator to be assigned.
     */
    template <typename U>
    ExecutorAllocator& operator=(const ExecutorAllocator<U>& other) noexcept
    {
        exec_ = other.get_executor();
        return *this;
    }

    /**
     * Constructs an allocator for another element type from a given executor.
     *
     * This is related to `std::allocator_traits::template rebind<U>` and its
     * use in more advanced data structures.
     *
     * @param other  the other allocator
     * @tparam U  the element type of the allocator to be constructed.
     */
    template <typename U>
    ExecutorAllocator(const ExecutorAllocator<U>& other) noexcept
        : exec_{other.get_executor()}
    {}

    /** Returns the executor used by this allocator.  */
    std::shared_ptr<const Executor> get_executor() const noexcept
    {
        return exec_;
    }

    /**
     * Allocates a memory area of the given size.
     *
     * @param n  the number of elements to allocate
     * @return  the pointer to a newly allocated memory area of `n` elements.
     */
    T* allocate(std::size_t n) const { return exec_->alloc<T>(n); }

    /**
     * Frees a memory area that was allocated by this allocator.
     *
     * @param ptr  The memory area to free, previously returned by `allocate`.
     *
     * @note  The second parameter is unused.
     */
    void deallocate(T* ptr, std::size_t) const { exec_->free(ptr); }

    /**
     * Compares two ExecutorAllocators for equality
     *
     * @param l  the first allocator
     * @param r  the second allocator
     * @return true iff the two allocators use the same executor
     */
    template <typename T2>
    friend bool operator==(const ExecutorAllocator<T>& l,
                           const ExecutorAllocator<T2>& r) noexcept
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
    friend bool operator!=(const ExecutorAllocator<T>& l,
                           const ExecutorAllocator<T2>& r) noexcept
    {
        return !(l == r);
    }

private:
    std::shared_ptr<const Executor> exec_;
};


// Convenience type aliases
/** std::vector using an ExecutorAllocator. */
template <typename T>
using vector = std::vector<T, ExecutorAllocator<T>>;

/** std::deque using an ExecutorAllocator. */
template <typename T>
using deque = std::deque<T, ExecutorAllocator<T>>;

/** std::set using an ExecutorAllocator. */
template <typename Key>
using set = std::set<Key, std::less<Key>, gko::ExecutorAllocator<Key>>;

/** std::map using an ExecutorAllocator. */
template <typename Key, typename Value>
using map = std::map<Key, Value, std::less<Key>,
                     gko::ExecutorAllocator<std::pair<const Key, Value>>>;

/** std::unordered_set using an ExecutorAllocator. */
template <typename Key>
using unordered_set =
    std::unordered_set<Key, std::hash<Key>, std::equal_to<Key>,
                       gko::ExecutorAllocator<Key>>;

/** std::unordered_map using an ExecutorAllocator. */
template <typename Key, typename Value>
using unordered_map =
    std::unordered_map<Key, Value, std::hash<Key>, std::equal_to<Key>,
                       gko::ExecutorAllocator<std::pair<const Key, Value>>>;


}  // namespace gko

#endif  // GKO_CORE_BASE_ALLOCATOR_HPP_
