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

#include "core/factorization/elimination_forest.hpp"


#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace factorization {
namespace {


template <typename IndexType>
void compute_elim_forest_parent_impl(std::shared_ptr<const Executor> host_exec,
                                     const IndexType* row_ptrs,
                                     const IndexType* cols, IndexType num_rows,
                                     IndexType* parent)
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
void compute_elim_forest_children_impl(const IndexType* parent, IndexType size,
                                       IndexType* child_ptr, IndexType* child)
{
    // count how many times each parent occurs, excluding pseudo-root at
    // parent == size
    std::fill_n(child_ptr, size + 2, IndexType{});
    for (IndexType i = 0; i < size; i++) {
        const auto p = parent[i];
        if (p < size) {
            child_ptr[p + 2]++;
        }
    }
    // shift by 2 leads to exclusive prefix sum with 0 padding
    std::partial_sum(child_ptr, child_ptr + size + 2, child_ptr);
    // we count the same again, this time shifted by 1 => exclusive prefix sum
    for (IndexType i = 0; i < size; i++) {
        const auto p = parent[i];
        child[child_ptr[p + 1]] = i;
        child_ptr[p + 1]++;
    }
}


template <typename IndexType>
void compute_elim_forest_postorder_impl(
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


template <typename IndexType>
void compute_elim_forest_postorder_parent_impl(const IndexType* parent,
                                               const IndexType* inv_postorder,
                                               IndexType size,
                                               IndexType* postorder_parent)
{
    for (IndexType row = 0; row < size; row++) {
        postorder_parent[inv_postorder[row]] =
            parent[row] == size ? size : inv_postorder[parent[row]];
    }
}


}  // namespace


template <typename IndexType>
elimination_forest<IndexType>::elimination_forest(
    std::shared_ptr<const Executor> host_exec, IndexType size)
    : parents{host_exec, static_cast<size_type>(size)},
      child_ptrs{host_exec, static_cast<size_type>(size + 2)},
      children{host_exec, static_cast<size_type>(size)},
      postorder{host_exec, static_cast<size_type>(size)},
      inv_postorder{host_exec, static_cast<size_type>(size)},
      postorder_parents{host_exec, static_cast<size_type>(size)}
{}


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
elimination_forest<IndexType> compute_elim_forest(
    const matrix::Csr<ValueType, IndexType>* mtx)
{
    const auto host_exec = mtx->get_executor()->get_master();
    const auto host_mtx = make_temporary_clone(host_exec, mtx);
    const auto num_rows = static_cast<IndexType>(host_mtx->get_size()[0]);
    elimination_forest<IndexType> forest{host_exec, num_rows};
    compute_elim_forest_parent_impl(host_exec, host_mtx->get_const_row_ptrs(),
                                    host_mtx->get_const_col_idxs(), num_rows,
                                    forest.parents.get_data());
    compute_elim_forest_children_impl(forest.parents.get_const_data(), num_rows,
                                      forest.child_ptrs.get_data(),
                                      forest.children.get_data());
    compute_elim_forest_postorder_impl(
        host_exec, forest.parents.get_const_data(),
        forest.child_ptrs.get_const_data(), forest.children.get_const_data(),
        num_rows, forest.postorder.get_data(), forest.inv_postorder.get_data());
    compute_elim_forest_postorder_parent_impl(
        forest.parents.get_const_data(), forest.inv_postorder.get_const_data(),
        num_rows, forest.postorder_parents.get_data());

    forest.set_executor(mtx->get_executor());
    return forest;
}


#define GKO_DECLARE_COMPUTE_ELIM_FOREST(ValueType, IndexType) \
    elimination_forest<IndexType> compute_elim_forest(        \
        const matrix::Csr<ValueType, IndexType>* mtx)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COMPUTE_ELIM_FOREST);


}  // namespace factorization
}  // namespace gko
