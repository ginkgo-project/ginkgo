// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
 * It allows inserting key-node pairs, modifying their key as well as accessing
 * and removing the key-node pair with the minimum key.
 *
 * @tparam KeyType    The type of the keys
 * @tparam IndexType  The type of the nodes, it needs to be an integer type.
 * @tparam degree     The node degree k
 */
template <typename KeyType, typename IndexType, int degree = 4>
struct addressable_priority_queue {
    /**
     * Constructs an addressable PQ from a host executor.
     *
     * @param host_exec  the host executor for allocating the data
     * @param num_nodes  the number of nodes that may be inserted into this
     *                   queue. Every node ID inserted must be below num_nodes.
     */
    addressable_priority_queue(std::shared_ptr<const Executor> host_exec,
                               size_type num_nodes)
        : keys_{host_exec},
          nodes_{host_exec},
          heap_pos_{num_nodes, invalid_index<IndexType>(), host_exec}
    {}

    /**
     * Inserts the given key-node pair into the PQ.
     * Duplicate keys are allowed, they may be returned in an arbitrary order.
     *
     * @param key  the key by which the queue is ordered
     * @param node  the node associated with the key. Every node may only be
     *              inserted once!
     */
    void insert(KeyType key, IndexType node)
    {
        GKO_ASSERT(node < static_cast<IndexType>(heap_pos_.size()));
        GKO_ASSERT(node >= 0);
        GKO_ASSERT(heap_pos_[node] == invalid_index<IndexType>());
        keys_.push_back(key);
        nodes_.push_back(node);
        const auto new_pos = size() - 1;
        heap_pos_[node] = new_pos;
        sift_up(new_pos);
    }

    /**
     * Updates the key of a node with the given new key.
     * Duplicate keys are allowed, they may be returned in an arbitrary order.
     *
     * @param new_key  the key by which the queue is ordered
     * @param node  the node associated with the key. It must have been inserted
     *              beforehand.
     */
    void update_key(KeyType new_key, IndexType node)
    {
        GKO_ASSERT(node < static_cast<IndexType>(heap_pos_.size()));
        GKO_ASSERT(node >= 0);
        auto pos = heap_pos_[node];
        GKO_ASSERT(pos < size());
        GKO_ASSERT(pos != invalid_index<IndexType>());
        GKO_ASSERT(nodes_[pos] == node);
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
     * Returns the node belonging to the minimum key from the queue.
     *
     * @return the node corresponding to the minimum key
     */
    IndexType min_node() const { return nodes_[0]; }

    /**
     * Returns the key-node pair with the minimum key from the queue.
     *
     * @return the key-node pair corresponding to the minimum key
     */
    std::pair<KeyType, IndexType> min() const
    {
        return {min_key(), min_node()};
    }

    /**
     * Removes the key-node pair with the minimum key from the queue.
     */
    void pop_min()
    {
        GKO_ASSERT(!empty());
        swap(0, size() - 1);
        heap_pos_[nodes_.back()] = invalid_index<IndexType>();
        keys_.pop_back();
        nodes_.pop_back();
        sift_down(0);
    }

    /**
     * Returns the number of key-node pairs in the queue.
     *
     * @return  the number of key-node pairs in the queue
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
        for (auto node : nodes_) {
            heap_pos_[node] = invalid_index<IndexType>();
        }
        keys_.clear();
        nodes_.clear();
    }

private:
    std::size_t parent(std::size_t i) const { return (i - 1) / degree; }

    std::size_t first_child(std::size_t i) const { return degree * i + 1; }

    void swap(std::size_t i, std::size_t j)
    {
        std::swap(keys_[i], keys_[j]);
        std::swap(nodes_[i], nodes_[j]);
        std::swap(heap_pos_[nodes_[i]], heap_pos_[nodes_[j]]);
    }

    /**
     * Restores the heap invariant downwards, i.e. the
     * Moves the key-node pair at position i down (toward the leaves)
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
     * Moves the key-node pair at position i up (toward the root)
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
    vector<IndexType> nodes_;
    // for each node, heap_pos_[node] stores the position of this node inside
    // the heap, or invalid_index<IndexType>() if it's not in the heap.
    vector<IndexType> heap_pos_;
};


}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_ADDRESSABLE_PQ_HPP_
