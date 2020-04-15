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


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The reordering namespace.
 *
 * @ingroup reorder
 */
namespace rcm {


template <typename ValueType, typename IndexType>
void get_degree_of_nodes(
    std::shared_ptr<const ReferenceExecutor> exec,
    std::shared_ptr<matrix::SparsityCsr<ValueType, IndexType>> adjacency_matrix,
    std::shared_ptr<gko::Array<IndexType>> node_degrees)
{
    auto num_rows = adjacency_matrix->get_size()[0];
    auto adj_ptrs = adjacency_matrix->get_row_ptrs();
    auto node_deg = node_degrees->get_data();

    for (auto i = 0; i < num_rows; ++i) {
        node_deg[i] = adj_ptrs[i + 1] - adj_ptrs[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_RCM_GET_DEGREE_OF_NODES_KERNEL);


/**
 * Computes a level structure rooted at `root`, returning the height of the
 * structure along with the index of the node of minimal degree in its last
 * level.
 */
template <typename IndexType, typename ValueType>
std::pair<IndexType, size_type> rls_contender_and_height(
    std::shared_ptr<const ReferenceExecutor> exec, IndexType root,
    std::shared_ptr<matrix::SparsityCsr<ValueType, IndexType>> adjacency_matrix,
    std::shared_ptr<Array<IndexType>> node_degrees, size_type num_vertices)
{
    auto adj_ptrs = adjacency_matrix->get_row_ptrs();
    auto adj_idxs = adjacency_matrix->get_col_idxs();
    auto node_degrees_p = node_degrees->get_data();

    // This could actually be allocated in the calling scope, then reused.
    auto visited_local =
        std::unique_ptr<Array<bool>>(new Array<bool>(exec, num_vertices));
    auto visited_local_p = visited_local->get_data();
    for (auto i = 0; i < num_vertices; ++i) {
        visited_local_p[i] = false;
    }

    // This stores the level structure, starting with the root node.
    std::vector<IndexType> rls;
    rls.reserve(num_vertices);
    rls.push_back(root);
    visited_local_p[root] = true;

    auto rls_index = 0;

    // Used to compute the height.
    auto height = 0;
    auto current_level_countdown = 1;
    auto next_level_countup = 0;

    // The last levels size is required to compute the contender.
    auto last_level_size = 0;

    while (rls_index < rls.size()) {
        auto parent = rls[rls_index];
        --current_level_countdown;


        // Iterate through the parents neighbors.
        auto row_start = adj_ptrs[parent];
        auto row_end = adj_ptrs[parent + 1];
        for (auto neighbor_idx = row_start; neighbor_idx < row_end;
             ++neighbor_idx) {
            auto neighbor = adj_idxs[neighbor_idx];

            if (!visited_local_p[neighbor]) {
                visited_local_p[neighbor] = true;
                ++next_level_countup;
                rls.push_back(neighbor);
            }
        }

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
    auto rls_size = rls.size();
    auto min_degree = std::numeric_limits<IndexType>::max();
    auto contender = rls_size - last_level_size;
    for (auto i = rls_size - last_level_size; i < rls_size; ++i) {
        if (node_degrees_p[rls[i]] < min_degree) {
            contender = i;
            min_degree = node_degrees_p[rls[i]];
        }
    }

    return std::pair<IndexType, size_type>(rls[contender], height);
}

template <typename IndexType, typename ValueType>
IndexType find_starting_node(
    std::shared_ptr<const ReferenceExecutor> exec,
    std::shared_ptr<matrix::SparsityCsr<ValueType, IndexType>> adjacency_matrix,
    std::shared_ptr<Array<IndexType>> node_degrees,
    std::shared_ptr<Array<bool>> visited, size_type num_vertices,
    gko::reorder::starting_strategy strategy)
{
    using strategies = gko::reorder::starting_strategy;

    // Only those strategies are supported here yet, assert this.
    GKO_ASSERT(strategy == strategies::minimum_degree ||
               startegy == strategies::pseudo_peripheral);

    auto node_degrees_p = node_degrees->get_data();
    auto visited_p = visited->get_data();

    // There must always be at least one unvisited node left.
    auto index_min_node = 0;
    auto min_node_degree = std::numeric_limits<IndexType>::max();
    for (auto i = 0; i < num_vertices; ++i) {
        if (!visited_p[i]) {
            if (node_degrees_p[i] < min_node_degree) {
                index_min_node = i;
                min_node_degree = node_degrees_p[i];
            }
        }
    }

    // If that is all that is required, return.
    if (strategy == gko::reorder::starting_strategy::minimum_degree) {
        return index_min_node;
    }

    // Find a pseudo-peripheral node, starting from an node of minimum degree.
    auto current = index_min_node;

    // Isolated nodes are by definition perioheral.
    if (min_node_degree == 0) {
        return index_min_node;
    }

    // This algorithm is e. g. presented in
    // http://heath.cs.illinois.edu/courses/cs598mh/george_liu.pdf, p. 70
    // onwards.
    auto contender_and_height = rls_contender_and_height<IndexType, ValueType>(
        exec, current, adjacency_matrix, node_degrees, num_vertices);
    while (true) {
        auto contender_and_height_contender =
            rls_contender_and_height<IndexType, ValueType>(
                exec, contender_and_height.first, adjacency_matrix,
                node_degrees, num_vertices);

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
 * Comutes a rcm reording using a naive sequential algorithm.
 */
template <typename ValueType, typename IndexType>
void get_permutation(
    std::shared_ptr<const ReferenceExecutor> exec, size_type num_vertices,
    std::shared_ptr<matrix::SparsityCsr<ValueType, IndexType>> adjacency_matrix,
    std::shared_ptr<Array<IndexType>> node_degrees,
    std::shared_ptr<matrix::Permutation<IndexType>> permutation_mat,
    std::shared_ptr<matrix::Permutation<IndexType>> inv_permutation_mat,
    const gko::reorder::starting_strategy strategy)
{
    IndexType num_vtxs = static_cast<IndexType>(num_vertices);
    auto adj_ptrs = adjacency_matrix->get_row_ptrs();
    auto adj_idxs = adjacency_matrix->get_col_idxs();

    auto permutation = permutation_mat->get_permutation();
    auto permutation_inv = inv_permutation_mat.get()
                               ? inv_permutation_mat->get_permutation()
                               : nullptr;

    IndexType *degrees = node_degrees->get_data();

    // Storing vertices left to proceess.
    auto linear_queue = std::unique_ptr<Array<IndexType>>(
        new Array<IndexType>(exec, num_vertices));
    auto linear_queue_p = linear_queue->get_data();
    auto head_offset = 0;
    auto tail_offset = 0;

    // Storing which vertices have already been visited.
    auto visited = std::make_shared<Array<bool>>(exec, num_vertices);
    auto visited_p = visited->get_data();
    for (auto i = 0; i < num_vertices; ++i) {
        visited_p[i] = false;
    }

    for (auto perm_index = 0; perm_index < num_vertices; ++perm_index) {
        // Get the next vertex to process.
        auto next_vertex = 0;
        if (head_offset <= tail_offset) {
            // Choose a new starting vertex, new componenet needs to be
            // discovered.
            next_vertex = find_starting_node<IndexType, ValueType>(
                exec, adjacency_matrix, node_degrees, visited, num_vertices,
                strategy);
            visited_p[next_vertex] = true;
        } else {
            next_vertex = linear_queue_p[tail_offset];
            ++tail_offset;
        }

        // Get the neigbours of the next vertex,
        // check if they have already been visited,
        // if no, insert them to sort.
        auto prev_head_offset = head_offset;

        // Is this code correct for getting next_vertexs neighbors?
        auto row_start = adj_ptrs[next_vertex];
        auto row_end = adj_ptrs[next_vertex + 1];
        for (auto neighbor_idx = row_start; neighbor_idx < row_end;
             ++neighbor_idx) {
            auto neighbor = adj_idxs[neighbor_idx];

            // If the neighbour hasn't been visited yet, add it to the queue and
            // mark it.
            if (!visited_p[neighbor]) {
                visited_p[neighbor] = true;
                linear_queue_p[head_offset] = neighbor;
                ++head_offset;
            }
        }

        auto count_added = head_offset - prev_head_offset;
        std::sort(linear_queue_p + prev_head_offset,
                  linear_queue_p + head_offset,
                  [&degrees](IndexType i, IndexType j) {
                      return degrees[i] < degrees[j];
                  });

        permutation[perm_index] = next_vertex;
    }

    if (permutation_inv) {
        for (auto i = 0; i < num_vertices; ++i) {
            permutation_inv[permutation[i]] = i;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_RCM_GET_PERMUTATION_KERNEL);


}  // namespace rcm
}  // namespace reference
}  // namespace kernels
}  // namespace gko