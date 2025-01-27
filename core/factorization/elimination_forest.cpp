// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/elimination_forest.hpp"

#include <ginkgo/core/base/types.hpp>

#include "core/factorization/elimination_forest_kernels.hpp"


namespace gko {
namespace factorization {
namespace {


GKO_REGISTER_OPERATION(compute_children, elimination_forest::compute_children);
GKO_REGISTER_OPERATION(compute_subtree_sizes,
                       elimination_forest::compute_subtree_sizes);
GKO_REGISTER_OPERATION(compute_postorder,
                       elimination_forest::compute_postorder);
GKO_REGISTER_OPERATION(map_postorder, elimination_forest::map_postorder);


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


template <typename ValueType, typename IndexType>
void compute_elimination_forest(
    const matrix::Csr<ValueType, IndexType>* mtx,
    std::unique_ptr<elimination_forest<IndexType>>& forest,
    elimination_forest_algorithm algorithm)
{
    const auto exec = mtx->get_executor();
    const auto host_exec = exec->get_master();
    const auto host_mtx = make_temporary_clone(host_exec, mtx);
    const auto num_rows = static_cast<IndexType>(host_mtx->get_size()[0]);
    forest = std::make_unique<elimination_forest<IndexType>>(exec, num_rows,
                                                             algorithm);
    compute_elimination_forest_parent_impl(
        host_exec, host_mtx->get_const_row_ptrs(),
        host_mtx->get_const_col_idxs(), num_rows, forest->parents.get_data());
    if (algorithm == elimination_forest_algorithm::device_children) {
        // from now on we do everything on the device
        forest->set_executor(exec);
        exec->run(make_compute_children(forest->parents.get_const_data(),
                                        num_rows, forest->child_ptrs.get_data(),
                                        forest->children.get_data()));
    } else {
        host_exec->run(make_compute_children(
            forest->parents.get_const_data(), num_rows,
            forest->child_ptrs.get_data(), forest->children.get_data()));
    }
    if (algorithm == elimination_forest_algorithm::device_children ||
        algorithm == elimination_forest_algorithm::device_postorder) {
        // from now on we do everything on the device
        forest->set_executor(exec);
        array<IndexType> subtree_sizes{exec, static_cast<size_type>(num_rows)};
        exec->run(
            make_compute_subtree_sizes(forest->child_ptrs.get_const_data(),
                                       forest->children.get_const_data(),
                                       num_rows, subtree_sizes.get_data()));
        exec->run(make_compute_postorder(
            forest->child_ptrs.get_const_data(),
            forest->children.get_const_data(), num_rows,
            subtree_sizes.get_const_data(), forest->postorder.get_data(),
            forest->inv_postorder.get_data()));
    } else {
        host_exec->run(make_compute_postorder(
            forest->child_ptrs.get_const_data(),
            forest->children.get_const_data(), num_rows,
            static_cast<const IndexType*>(nullptr),
            forest->postorder.get_data(), forest->inv_postorder.get_data()));
    }
    if (algorithm != elimination_forest_algorithm::host) {
        // from now on we do everything on the device
        forest->set_executor(exec);
        exec->run(make_map_postorder(forest->parents.get_const_data(),
                                     forest->child_ptrs.get_const_data(),
                                     forest->children.get_const_data(),
                                     num_rows,
                                     forest->inv_postorder.get_const_data(),
                                     forest->postorder_parents.get_data(),
                                     forest->postorder_child_ptrs.get_data(),
                                     forest->postorder_children.get_data()));
    } else {
        host_exec->run(
            make_map_postorder(forest->parents.get_const_data(),
                               forest->child_ptrs.get_const_data(),
                               forest->children.get_const_data(), num_rows,
                               forest->inv_postorder.get_const_data(),
                               forest->postorder_parents.get_data(),
                               forest->postorder_child_ptrs.get_data(),
                               forest->postorder_children.get_data()));
        forest->set_executor(exec);
    }
}

#define GKO_DECLARE_COMPUTE_ELIMINATION_FOREST(ValueType, IndexType) \
    void compute_elimination_forest(                                 \
        const matrix::Csr<ValueType, IndexType>* mtx,                \
        std::unique_ptr<elimination_forest<IndexType>>& forest,      \
        elimination_forest_algorithm algorithm)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COMPUTE_ELIMINATION_FOREST);


}  // namespace factorization
}  // namespace gko
