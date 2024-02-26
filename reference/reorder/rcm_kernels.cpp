// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/reorder/rcm_kernels.hpp"


#include <algorithm>
#include <iterator>
#include <memory>
#include <queue>
#include <utility>
#include <vector>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/allocator.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The reordering namespace.
 *
 * @ingroup reorder
 */
namespace rcm {


/**
 * Computes a level structure rooted at `root`, returning a node of minimal
 * degree in its last level, along with the height of the structure.
 */
template <typename IndexType>
std::pair<IndexType, size_type> rls_contender_and_height(
    std::shared_ptr<const ReferenceExecutor> exec, const IndexType num_vertices,
    const IndexType root, const IndexType* const row_ptrs,
    const IndexType* const col_idxs, const IndexType* const degrees)
{
    // This could actually be allocated in the calling scope, then reused.
    vector<bool> visited_local(num_vertices, false, exec);

    // This stores a reordering in bfs order, starting with the root node.
    array<IndexType> rls(exec, num_vertices);
    auto rls_p = rls.get_data();
    rls_p[0] = root;
    IndexType rls_offset = 1;
    visited_local[root] = true;

    IndexType rls_index = 0;

    // Used to compute the height.
    // Always count up the count of the next level,
    // while counting down the current levels count.
    IndexType height = 0;
    IndexType current_level_countdown = 1;
    IndexType next_level_countup = 0;

    // The last levels size is required to compute the contender.
    IndexType last_level_size = 0;

    // While there are still nodes whose neighbors haven't been inspected.
    while (rls_index < rls_offset) {
        auto parent = rls_p[rls_index];
        --current_level_countdown;

        // Iterate through parents neighbors.
        auto row_start = row_ptrs[parent];
        auto row_end = row_ptrs[parent + 1];
        for (auto neighbor_idx = row_start; neighbor_idx < row_end;
             ++neighbor_idx) {
            auto neighbor = col_idxs[neighbor_idx];

            // If this is a new node, add it to the rls and mark it as visited.
            if (!visited_local[neighbor]) {
                visited_local[neighbor] = true;
                ++next_level_countup;
                rls_p[rls_offset] = neighbor;
                ++rls_offset;
            }
        }

        // Machinery for computing the last levels length.
        if (current_level_countdown == 0) {
            if (next_level_countup > 0) {
                last_level_size = next_level_countup;
            }

            current_level_countdown = next_level_countup;
            next_level_countup = 0;
            ++height;
        }

        ++rls_index;
    }

    // Choose the contender.
    // It's the node of minimum degree in the last level.
    auto rls_size = rls_offset;
    auto contender = *std::min_element(
        rls_p + rls_size - last_level_size, rls_p + rls_size,
        [&](IndexType u, IndexType v) { return degrees[u] < degrees[v]; });

    return std::pair<IndexType, IndexType>(contender, height);
}

template <typename IndexType>
IndexType find_starting_node(std::shared_ptr<const ReferenceExecutor> exec,
                             const IndexType num_vertices,
                             const IndexType* const row_ptrs,
                             const IndexType* const col_idxs,
                             const IndexType* const degrees,
                             const vector<bool>& visited,
                             const gko::reorder::starting_strategy strategy)
{
    using strategies = gko::reorder::starting_strategy;

    // Only those strategies are supported here yet, assert this.
    GKO_ASSERT(strategy == strategies::minimum_degree ||
               strategy == strategies::pseudo_peripheral);

    // There must always be at least one unvisited node left.
    IndexType index_min_node = 0;
    IndexType min_node_degree = std::numeric_limits<IndexType>::max();
    for (IndexType i = 0; i < num_vertices; ++i) {
        if (!visited[i]) {
            if (degrees[i] < min_node_degree) {
                index_min_node = i;
                min_node_degree = degrees[i];
            }
        }
    }

    // If that is all that is required, return.
    if (strategy == gko::reorder::starting_strategy::minimum_degree) {
        return index_min_node;
    }

    // Find a pseudo-peripheral node, starting from an node of minimum degree.
    auto current = index_min_node;

    // Isolated nodes are by definition peripheral.
    if (min_node_degree == 0) {
        return index_min_node;
    }

    // This algorithm is e. g. presented in
    // http://heath.cs.illinois.edu/courses/cs598mh/george_liu.pdf, p. 70
    // onwards.
    auto contender_and_height = rls_contender_and_height<IndexType>(
        exec, num_vertices, current, row_ptrs, col_idxs, degrees);
    while (true) {
        auto contender_and_height_contender =
            rls_contender_and_height<IndexType>(exec, num_vertices,
                                                contender_and_height.first,
                                                row_ptrs, col_idxs, degrees);

        if (contender_and_height_contender.second >
            contender_and_height.second) {
            contender_and_height.second = contender_and_height_contender.second;
            current = contender_and_height.first;
            contender_and_height.first = contender_and_height_contender.first;
        } else {
            return current;
        }
    }
}

/**
 * Computes a RCM reordering using a naive sequential algorithm.
 */
template <typename IndexType>
void compute_permutation(std::shared_ptr<const ReferenceExecutor> exec,
                         const IndexType num_vertices,
                         const IndexType* const row_ptrs,
                         const IndexType* const col_idxs,
                         IndexType* const permutation,
                         IndexType* const inv_permutation,
                         const gko::reorder::starting_strategy strategy)
{
    // compute node degrees
    array<IndexType> degree_array{exec, static_cast<size_type>(num_vertices)};
    const auto degrees = degree_array.get_data();
    for (IndexType i = 0; i < num_vertices; ++i) {
        degrees[i] = row_ptrs[i + 1] - row_ptrs[i];
    }
    // Storing vertices left to proceess.
    array<IndexType> linear_queue(exec, num_vertices);
    auto linear_queue_p = linear_queue.get_data();
    IndexType head_offset = 0;
    IndexType tail_offset = 0;

    // Storing which vertices have already been visited.
    vector<bool> visited(num_vertices, false, exec);

    for (IndexType perm_index = 0; perm_index < num_vertices; ++perm_index) {
        // Get the next vertex to process.
        IndexType next_vertex = 0;
        if (head_offset <= tail_offset) {
            // Choose a new starting vertex, new component needs to be
            // discovered.
            next_vertex = find_starting_node<IndexType>(
                exec, num_vertices, row_ptrs, col_idxs, degrees, visited,
                strategy);
            visited[next_vertex] = true;
        } else {
            next_vertex = linear_queue_p[tail_offset];
            ++tail_offset;
        }

        // Get the neighbors of the next vertex,
        // check if they have already been visited,
        // if no, insert them to sort.
        auto prev_head_offset = head_offset;

        // Get the next vertex neighbors.
        auto row_start = row_ptrs[next_vertex];
        auto row_end = row_ptrs[next_vertex + 1];
        for (auto neighbor_idx = row_start; neighbor_idx < row_end;
             ++neighbor_idx) {
            auto neighbor = col_idxs[neighbor_idx];

            // If the neighbour hasn't been visited yet, add it to the queue and
            // mark it.
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                linear_queue_p[head_offset] = neighbor;
                ++head_offset;
            }
        }

        // Sort all just-added neighbors by degree.
        std::stable_sort(
            linear_queue_p + prev_head_offset, linear_queue_p + head_offset,
            [&](IndexType i, IndexType j) { return degrees[i] < degrees[j]; });

        // Write out the processed vertex.
        permutation[num_vertices - perm_index - 1] = next_vertex;
    }

    if (inv_permutation) {
        for (IndexType i = 0; i < num_vertices; ++i) {
            inv_permutation[permutation[i]] = i;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RCM_COMPUTE_PERMUTATION_KERNEL);


}  // namespace rcm
}  // namespace reference
}  // namespace kernels
}  // namespace gko
