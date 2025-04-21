// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/elimination_forest.hpp"

#include <ginkgo/core/base/types.hpp>

#include "core/factorization/elimination_forest_kernels.hpp"


namespace gko {
namespace factorization {
namespace {


template <typename IndexType>
void compute_elimination_forest_parent_impl(
    std::shared_ptr<const Executor> host_exec, const IndexType* row_ptrs,
    const IndexType* cols, IndexType num_rows, IndexType* parent)
{
    disjoint_sets<IndexType> subtrees{host_exec, num_rows};
    array<IndexType> subtree_root_array{host_exec,
                                        static_cast<size_type>(num_rows)};
    // pseudo-root one past the last row to deal with disconnected matrices
    const auto unattached = num_rows;
    auto subtree_root = subtree_root_array.get_data();
    for (IndexType row = 0; row < num_rows; row++) {
        // so far the row is an unattached singleton subtree
        subtree_root[row] = row;
        parent[row] = unattached;
        auto row_rep = row;
        for (auto nz = row_ptrs[row]; nz < row_ptrs[row + 1]; nz++) {
            const auto col = cols[nz];
            // for each lower triangular entry
            if (col < row) {
                // find the subtree it is contained in
                const auto col_rep = subtrees.find(col);
                const auto col_root = subtree_root[col_rep];
                // if it is not yet attached, put it below row
                // and make row its new root
                if (parent[col_root] == unattached && col_root != row) {
                    parent[col_root] = row;
                    row_rep = subtrees.join(row_rep, col_rep);
                    subtree_root[row_rep] = row;
                }
            }
        }
    }
}


template <typename IndexType>
void compute_elimination_forest_postorder_impl(
    std::shared_ptr<const Executor> host_exec, const IndexType* parent,
    const IndexType* child_ptr, const IndexType* child, IndexType size,
    IndexType* postorder, IndexType* inv_postorder)
{
    array<IndexType> current_child_array{host_exec,
                                         static_cast<size_type>(size + 1)};
    current_child_array.fill(0);
    auto current_child = current_child_array.get_data();
    IndexType postorder_idx{};
    // for each tree in the elimination forest
    for (auto tree = child_ptr[size]; tree < child_ptr[size + 1]; tree++) {
        // start from the root
        const auto root = child[tree];
        auto cur_node = root;
        // traverse until we moved to the pseudo-root
        while (cur_node < size) {
            const auto first_child = child_ptr[cur_node];
            const auto num_children = child_ptr[cur_node + 1] - first_child;
            if (current_child[cur_node] >= num_children) {
                // if this node is completed, output it
                postorder[postorder_idx] = cur_node;
                inv_postorder[cur_node] = postorder_idx;
                cur_node = parent[cur_node];
                postorder_idx++;
            } else {
                // otherwise go to the next child node
                const auto old_node = cur_node;
                cur_node = child[first_child + current_child[old_node]];
                current_child[old_node]++;
            }
        }
    }
}


}  // namespace


template <typename IndexType>
void elimination_forest<IndexType>::set_executor(
    std::shared_ptr<const Executor> exec)
{
    parents.set_executor(exec);
    child_ptrs.set_executor(exec);
    children.set_executor(exec);
    postorder.set_executor(exec);
    inv_postorder.set_executor(exec);
    postorder_parents.set_executor(exec);
}


}  // namespace factorization
}  // namespace gko
