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
#include <numeric>
#include <vector>


#include <ginkgo/core/base/types.hpp>


#include "core/base/allocator.hpp"


namespace gko {


/**
 * An addressable priority queue based on a k-ary heap storing key-index pairs.
 * It gets initialized with given key values and indices 0, ..., size() - 1.
 * It allows modifying key values as well as accessing and removing the
 * key-value pair with the minimum key.
 *
 * @tparam KeyType    The type of the keys
 * @tparam IndexType  The type of the indices
 */
template <int deg_log2, typename KeyType, typename IndexType>
class static_addressable_pq {
public:
    static_addressable_pq(vector<KeyType> keys)
        : m_keys(std::move(keys)),
          m_positions(m_keys.size(), m_keys.allocator()),
          m_indices(m_keys.size(), m_keys.allocator())
    {
        std::iota(m_positions.begin(), m_positions.end(), 0);
        std::iota(m_indices.begin(), m_indices.end(), 0);
        // heapify
        for (auto i = parent(size() - 1); i >= 0; i--) {
            sift_down(i);
        }
    }

    /**
     * Updates the key of the pair with the given index.
     */
    void update_key(IndexType idx, KeyType new_key)
    {
        auto pos = m_positions[idx];
        GKO_ASSERT(pos < size());
        GKO_ASSERT(m_indices[pos] == idx);
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
     * Returns the index belonging to the minimum key from the queue.
     *
     * @return the index corresponding to the minimun key
     */
    IndexType min_idx() const { return m_indices[0]; }

    /**
     * Returns the key-index pair with the minimum key from the queue.
     *
     * @return the key-index pair corresponding to the minimun key
     */
    std::pair<KeyType, IndexType> min() const { return {min_key(), min_idx()}; }

    /**
     * Removes the key-value pair with the minimum key from the queue.
     */
    void pop_min()
    {
        swap(0, size() - 1);
        m_keys.pop_back();
        m_indices.pop_back();
        sift_down(0);
    }

    /**
     * Returns the number of key-value pairs in the queue.
     *
     * @return  the number of key-value pairs in the queue
     */
    IndexType size() const { return m_keys.size(); }

    /**
     * Returns true if and only if the queue has size 0.
     *
     * @return if queue has size 0
     */
    bool empty() const { return size() == 0; }

private:
    constexpr static int degree{1 << deg_log2};

    IndexType parent(IndexType i) const { return (i - 1) / degree; }

    IndexType first_child(IndexType i) const { return degree * i + 1; }

    void swap(IndexType i, IndexType j)
    {
        std::swap(m_keys[i], m_keys[j]);
        std::swap(m_indices[i], m_indices[j]);
        std::swap(m_positions[m_keys[i]], m_positions[m_keys[j]]);
    }

    /**
     * Moves the key-value pair at position i down (toward the leaves)
     * until its key is smaller or equal to the one of all its children.
     */
    void sift_down(IndexType i)
    {
        auto cur = i;
        while (first_child(cur) < size()) {
            const auto it = [&] {
                if (first_child(cur + 1) <= size()) {
                    // fast path: fixed loop count
                    const auto begin = m_keys.begin() + first_child(cur);
                    const auto end = begin + degree;
                    return std::min_element(begin, end);
                } else {
                    // slow path: variable loop count
                    const auto begin = m_keys.begin() + first_child(cur);
                    const auto end = m_keys.begin() + size();
                    return std::min_element(begin, end);
                }
            }();
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
    void sift_up(IndexType i)
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

    vector<KeyType> m_keys;
    vector<IndexType> m_positions;
    vector<IndexType> m_indices;
};


}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_ADDRESSABLE_PQ_HPP_
