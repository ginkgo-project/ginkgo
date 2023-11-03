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

#ifndef GKO_CORE_COMPONENTS_ADDRESSABLE_PQ_HPP_
#define GKO_CORE_COMPONENTS_ADDRESSABLE_PQ_HPP_


#include <algorithm>
#include <vector>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/allocator.hpp"


namespace gko {


/**
 * An addressable priority queue based on a k-ary heap.
 *
 * It allows inserting key-value pairs, modifying their key as well as accessing
 * and removing the key-value pair with the minimum key.
 *
 * @tparam KeyType    The type of the keys
 * @tparam ValueType  The type of the values, it needs to be an integer type.
 * @tparam deg_log2  The binary logarithm of the node degree k
 */
template <typename KeyType, typename ValueType, int deg_log2 = 4>
struct addressable_priority_queue {
    constexpr static int degree = 1 << deg_log2;

    /**
     * Constructs an addressable PQ from its host executor and an array for
     * storing the binary heap positions for each of the values.
     */
    addressable_priority_queue(std::shared_ptr<const Executor> exec,
                               size_type num_values)
        : keys_{exec},
          values_{exec},
          heap_pos_{num_values, unused_handle(), exec}
    {}

    /**
     * Inserts the given key-value pair into the PQ.
     * Duplicate keys are allowed, they may be returned in an arbitrary order.
     *
     * @param key  the key by which the queue is ordered
     * @param value  the value associated with the key. No two keys may have the
     *               same value!
     */
    void insert(KeyType key, ValueType value)
    {
        GKO_ASSERT(value < static_cast<ValueType>(heap_pos_.size()));
        GKO_ASSERT(value >= 0);
        GKO_ASSERT(heap_pos_[value] == unused_handle());
        keys_.push_back(key);
        values_.push_back(value);
        const auto new_pos = size() - 1;
        heap_pos_[value] = new_pos;
        sift_up(new_pos);
    }

    /**
     * Updates the key of the pair with the given new key.
     */
    void update_key(KeyType new_key, ValueType value)
    {
        GKO_ASSERT(value < static_cast<ValueType>(heap_pos_.size()));
        GKO_ASSERT(value >= 0);
        auto pos = heap_pos_[value];
        GKO_ASSERT(pos < size());
        GKO_ASSERT(pos != unused_handle());
        GKO_ASSERT(values_[pos] == value);
        auto old_key = keys_[pos];
        keys_[pos] = new_key;
        if (old_key < new_key) {
            sift_down(pos);
        } else {
            sift_up(pos);
        }
    }

    /**
     * Returns the minimum key from the queue.
     *
     * @return the minimum key from the queue
     */
    KeyType min_key() const { return keys_[0]; }

    /**
     * Returns the value belonging to the minimum key from the queue.
     *
     * @return the value corresponding to the minimum key
     */
    ValueType min_val() const { return values_[0]; }

    /**
     * Returns the key-value pair with the minimum key from the queue.
     *
     * @return the key-value pair corresponding to the minimum key
     */
    std::pair<KeyType, ValueType> min() const { return {min_key(), min_val()}; }

    /**
     * Removes the key-value pair with the minimum key from the queue.
     */
    void pop_min()
    {
        swap(0, size() - 1);
        heap_pos_[values_.back()] = unused_handle();
        keys_.pop_back();
        values_.pop_back();
        sift_down(0);
    }

    /**
     * Returns the number of key-value pairs in the queue.
     *
     * @return  the number of key-value pairs in the queue
     */
    std::size_t size() const { return keys_.size(); }

    /**
     * Returns true if and only if the queue has size 0.
     *
     * @return if queue has size 0
     */
    bool empty() const { return size() == 0; }

    /** Clears the queue, removing all entries. */
    void reset()
    {
        for (auto value : values_) {
            heap_pos_[value] = unused_handle();
        }
        keys_.clear();
        values_.clear();
    }

private:
    std::size_t parent(std::size_t i) const { return (i - 1) / degree; }

    std::size_t first_child(std::size_t i) const { return degree * i + 1; }

    // This is a function instead of a member because otherwise we'd need to
    // explicitly export the symbol. C++17 fixes this with inline variables
    constexpr static size_type unused_handle() { return ~size_type{}; }

    void swap(std::size_t i, std::size_t j)
    {
        std::swap(keys_[i], keys_[j]);
        std::swap(values_[i], values_[j]);
        std::swap(heap_pos_[values_[i]], heap_pos_[values_[j]]);
    }

    /**
     * Restores the heap invariant downwards, i.e. the
     * Moves the key-value pair at position i down (toward the leaves)
     * until its key is smaller or equal to the one of all its children.
     */
    void sift_down(std::size_t i)
    {
        auto cur = i;
        while (first_child(cur) < size()) {
            auto it = keys_.cbegin();
            if (first_child(cur + 1) < size()) {
                // fast path: known loop trip count
                it = std::min_element(keys_.cbegin() + first_child(cur),
                                      keys_.cbegin() + first_child(cur + 1));
            } else {
                // slow path: unknown loop trip count
                it = std::min_element(keys_.cbegin() + first_child(cur),
                                      keys_.cbegin() + size());
            }
            if (keys_[cur] <= *it) {
                break;
            }
            auto min_child = std::distance(keys_.cbegin(), it);
            swap(cur, min_child);
            cur = min_child;
        }
    }

    /**
     * Moves the key-value pair at position i up (toward the root)
     * until its key is larger or equal to the one of its parent.
     */
    void sift_up(std::size_t i)
    {
        auto cur = i;
        while (cur > 0) {
            if (keys_[cur] >= keys_[parent(cur)]) {
                break;
            }
            swap(cur, parent(cur));
            cur = parent(cur);
        }
    }

    vector<KeyType> keys_;
    vector<ValueType> values_;
    // for each value, heap_pos_[value] stores the position of this value inside
    // the heap, or unused_handle() if it's not in the heap.
    vector<size_type> heap_pos_;
};


}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_ADDRESSABLE_PQ_HPP_
