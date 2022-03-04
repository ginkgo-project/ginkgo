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

#ifndef GKO_CORE_COMPONENTS_ADRESSABLE_PQ_HPP_
#define GKO_CORE_COMPONENTS_ADRESSABLE_PQ_HPP_


#include <deque>
#include <vector>


namespace gko {


/**
 * An addressable priority queue based on a k-ary heap.
 *
 * It allows inserting key-value pairs, modifying their key as well as accessing
 * and removing the key-value pair with the minimum key.
 *
 * @tparam Degree_Log2 the binary logarithm of the heap arity, i.e.,
 *         `k = 1 << Degree_Log2`
 */
template <typename KeyType, typename ValueType, int Degree_Log2>
struct addressable_priority_queue {
    /**
     * Inserts the given key-value pair into the PQ.
     * Duplicate keys are allowed, they may be returned in an arbitrary order.
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

    /** Updates the key of the pair with the given handle. */
    void update_key(std::size_t handle, KeyType new_key)
    {
        auto pos = m_handle_pos[handle];
        assert(pos < size());
        assert(m_handles[pos] == handle);
        auto old_key = m_keys[pos];
        m_keys[pos] = new_key;
        if (old_key < new_key) {
            sift_down(pos);
        } else {
            sift_up(pos);
        }
    }

    /** Returns the minimum key from the queue. */
    KeyType min_key() const { return m_keys[0]; }

    /** Returns the value belonging to the minimum key from the queue. */
    ValueType min_val() const { return m_values[0]; }

    /** Returns the key-value pair with the minimum key from the queue. */
    std::pair<KeyType, ValueType> min() const { return {min_key(), min_val()}; }

    /** Removes the key-value pair with the minimum key from the queue. */
    void pop_min()
    {
        swap(0, size() - 1);
        m_keys.pop_back();
        auto val = m_values.back();
        m_values.pop_back();
        auto old_handle = m_handles.back();
        m_handles.pop_back();
        m_free_handles.push_front(old_handle);
        m_handle_pos[old_handle] = invalid_handle;
        sift_down(0);
    }

    /** Returns the number of key-value pairs in the queue. */
    std::size_t size() const { return m_keys.size(); }

    /** Returns true if and only if the queue has size 0. */
    bool empty() const { return size() == 0; }

    void reset()
    {
        m_keys.clear();
        m_values.clear();
        m_handles.clear();
        m_handle_pos.clear();
        m_free_handles.clear();
    }

private:
    constexpr static int degree = 1 << Degree_Log2;
    constexpr static auto invalid_handle = -1;  //((std::size_t)-1);

    std::size_t parent(std::size_t i) const { return (i - 1) / degree; }

    std::size_t first_child(std::size_t i) const { return degree * i + 1; }

    void swap(std::size_t i, std::size_t j)
    {
        std::swap(m_keys[i], m_keys[j]);
        std::swap(m_values[i], m_values[j]);
        std::swap(m_handles[i], m_handles[j]);
        std::swap(m_handle_pos[m_handles[i]], m_handle_pos[m_handles[j]]);
    }

    void sift_down(std::size_t i)
    {
        auto cur = i;
        while (first_child(cur) < size()) {
            auto begin = m_keys.begin() + first_child(cur);
            auto end = m_keys.begin() + std::min(first_child(cur + 1), size());
            auto it = std::min_element(begin, end);
            if (m_keys[cur] <= *it) {
                break;
            }
            auto min_child = std::distance(m_keys.begin(), it);
            swap(cur, min_child);
            cur = min_child;
        }
    }

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

    std::size_t next_handle()
    {
        if (m_free_handles.empty()) {
            return m_handle_pos.size();
        } else {
            auto next = m_free_handles.back();
            m_free_handles.pop_back();
            return next;
        }
    }

    std::vector<KeyType> m_keys;
    std::vector<ValueType> m_values;
    std::vector<std::size_t> m_handles;
    std::vector<std::size_t> m_handle_pos;
    std::deque<std::size_t> m_free_handles;
};


}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_ADDRESSABLE_PQ_HPP_
