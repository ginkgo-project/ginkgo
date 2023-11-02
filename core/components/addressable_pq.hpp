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
 * @tparam ValueType  The type of the values
 * @tparam deg_log2  The binary logarithm of the node degree k
 */
template <typename KeyType, typename ValueType, int deg_log2 = 4>
struct addressable_priority_queue {
    constexpr static int degree = 1 << deg_log2;

    addressable_priority_queue(std::shared_ptr<const Executor> exec)
        : keys_{exec}, values_{exec}, handles_{exec}, handle_pos_{exec}
    {}

    /**
     * Inserts the given key-value pair into the PQ.
     * Duplicate keys are allowed, they may be returned in an arbitrary order.
     *
     * @param key  the key by which the queue is ordered
     * @param value  the value associated with the key
     *
     * @returns a handle for the pair to be used when modifying the key.
     */
    std::size_t insert(KeyType key, ValueType value)
    {
        keys_.push_back(key);
        values_.push_back(value);
        auto handle = next_handle();
        handles_.push_back(handle);
        if (handle == handle_pos_.size()) {
            handle_pos_.push_back(size() - 1);
        } else {
            handle_pos_[handle] = size() - 1;
        }
        sift_up(size() - 1);
        return handle;
    }

    /**
     * Updates the key of the pair with the given handle.
     */
    void update_key(std::size_t handle, KeyType new_key)
    {
        auto pos = handle_pos_[handle];
        GKO_ASSERT(pos < size());
        GKO_ASSERT(handles_[pos] == handle);
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
        keys_.pop_back();
        values_.pop_back();
        handles_.pop_back();
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

    void reset()
    {
        keys_.clear();
        values_.clear();
        handles_.clear();
        handle_pos_.clear();
    }

private:
    std::size_t parent(std::size_t i) const { return (i - 1) / degree; }

    std::size_t first_child(std::size_t i) const { return degree * i + 1; }

    void swap(std::size_t i, std::size_t j)
    {
        std::swap(keys_[i], keys_[j]);
        std::swap(values_[i], values_[j]);
        std::swap(handles_[i], handles_[j]);
        std::swap(handle_pos_[handles_[i]], handle_pos_[handles_[j]]);
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
            const auto begin = keys_.begin() + first_child(cur);
            const auto end =
                keys_.begin() + std::min(first_child(cur + 1), size());
            const auto it = std::min_element(begin, end);
            if (keys_[cur] <= *it) {
                break;
            }
            auto min_child = std::distance(keys_.begin(), it);
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

    // FIXME use free-list
    std::size_t next_handle() const { return handle_pos_.size(); }

    vector<KeyType> keys_;
    vector<ValueType> values_;
    vector<std::size_t> handles_;
    vector<std::size_t> handle_pos_;
};


}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_ADDRESSABLE_PQ_HPP_
