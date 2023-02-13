/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


#include <ginkgo/core/base/types.hpp>


namespace gko {


/**
 * An addressable priority queue based on a k-ary heap.
 *
 * It allows inserting key-value pairs, modifying their key as well as accessing
 * and removing the key-value pair with the minimum key.
 *
 * @tparam KeyType    The type of the keys
 * @tparam ValueType  The type of the values
 */
template <typename KeyType, typename ValueType>
struct addressable_priority_queue {
    explicit addressable_priority_queue(int deg_log2) : degree{1 << deg_log2} {}

    /**
     * Inserts the given key-value pair into the PQ.
     * Duplicate keys are allowed, they may be returned in an arbitrary order.
     *
     * @returns a handle for the pair to be used when modifying the key.
     */
    std::size_t insert(KeyType key, ValueType value)
    {
        m_keys.push_back(key);
        m_values.push_back(value);
        auto handle = next_handle();
        m_handles.push_back(handle);
        if (handle == m_handle_pos.size())
            m_handle_pos.push_back(size() - 1);
        else
            m_handle_pos[handle] = size() - 1;
        sift_up(size() - 1);
        return handle;
    }

    /**
     * Updates the key of the pair with the given handle.
     */
    void update_key(std::size_t handle, KeyType new_key)
    {
        auto pos = m_handle_pos[handle];
        GKO_ASSERT(pos < size());
        GKO_ASSERT(m_handles[pos] == handle);
        auto old_key = m_keys[pos];
        m_keys[pos] = new_key;
        if (old_key < new_key) {
            sift_down(pos);
        } else {
            sift_up(pos);
        }
    }

    /**
     * Returns the minimum key from the queue.
     *
     * @return the minimun key from the queue
     */
    KeyType min_key() const { return m_keys[0]; }

    /**
     * Returns the value belonging to the minimum key from the queue.
     *
     * @return the value corresponding to the minimun key
     */
    ValueType min_val() const { return m_values[0]; }

    /**
     * Returns the key-value pair with the minimum key from the queue.
     *
     * @return the key-value pair corresponding to the minimun key
     */
    std::pair<KeyType, ValueType> min() const { return {min_key(), min_val()}; }

    /**
     * Removes the key-value pair with the minimum key from the queue.
     */
    void pop_min()
    {
        swap(0, size() - 1);
        m_keys.pop_back();
        m_values.pop_back();
        auto old_handle = m_handles.back();
        m_handles.pop_back();
        sift_down(0);
    }

    /**
     * Returns the number of key-value pairs in the queue.
     *
     * @return  the number of key-value pairs in the queue
     */
    std::size_t size() const { return m_keys.size(); }

    /**
     * Returns true if and only if the queue has size 0.
     *
     * @return if queue has size 0
     */
    bool empty() const { return size() == 0; }

    void reset()
    {
        m_keys.clear();
        m_values.clear();
        m_handles.clear();
        m_handle_pos.clear();
    }

private:
    std::size_t parent(std::size_t i) const { return (i - 1) / degree; }

    std::size_t first_child(std::size_t i) const { return degree * i + 1; }

    void swap(std::size_t i, std::size_t j)
    {
        std::swap(m_keys[i], m_keys[j]);
        std::swap(m_values[i], m_values[j]);
        std::swap(m_handles[i], m_handles[j]);
        std::swap(m_handle_pos[m_handles[i]], m_handle_pos[m_handles[j]]);
    }

    /**
     * Moves the key-value pair at position i down (toward the leaves)
     * until its key is smaller or equal to the one of all its children.
     */
    void sift_down(std::size_t i)
    {
        auto cur = i;
        while (first_child(cur) < size()) {
            const auto begin = m_keys.begin() + first_child(cur);
            const auto end =
                m_keys.begin() + std::min(first_child(cur + 1), size());
            const auto it = std::min_element(begin, end);
            if (m_keys[cur] <= *it) {
                break;
            }
            auto min_child = std::distance(m_keys.begin(), it);
            swap(cur, min_child);
            cur = min_child;
        }
    }

    /**
     * Moves the key-value pair at position i up (toward the root)
     * until its key is larger or equal to the one of its parent.
     * */
    void sift_up(std::size_t i)
    {
        auto cur = i;
        while (cur > 0) {
            if (m_keys[cur] >= m_keys[parent(cur)]) {
                break;
            }
            swap(cur, parent(cur));
            cur = parent(cur);
        }
    }

    std::size_t next_handle() const { return m_handle_pos.size(); }

    const int degree;
    std::vector<KeyType> m_keys;
    std::vector<ValueType> m_values;
    std::vector<std::size_t> m_handles;
    std::vector<std::size_t> m_handle_pos;
};


}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_ADDRESSABLE_PQ_HPP_
