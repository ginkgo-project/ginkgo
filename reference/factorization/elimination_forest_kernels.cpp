// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/elimination_forest.hpp"

#include <algorithm>
#include <memory>
#include <numeric>

#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/allocator.hpp"
#include "core/base/index_range.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/elimination_forest_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace elimination_forest {


template <typename IndexType>
void compute_skeleton_tree(std::shared_ptr<const DefaultExecutor> exec,
                           const IndexType* row_ptrs, const IndexType* cols,
                           size_type size, IndexType* out_row_ptrs,
                           IndexType* out_cols)
{
    disjoint_sets<IndexType> sets(exec, size);
    const auto nnz = static_cast<size_type>(row_ptrs[size]);
    vector<std::pair<IndexType, IndexType>> edges(exec);
    edges.reserve(nnz / 2);
    // collect edge list
    for (auto row : irange(static_cast<IndexType>(size))) {
        for (auto nz : irange(row_ptrs[row], row_ptrs[row + 1])) {
            const auto col = cols[nz];
            if (col >= row) {
                continue;
            }
            // edge contains (max, min) pair
            edges.emplace_back(row, col);
        }
    }
    // the edge list is now sorted by row, which also matches the edge weight
    // we don't need to do any additional sorting operations
    assert(std::is_sorted(edges.begin(), edges.end(),
                          [](auto a, auto b) { return a.first < b.first; }));
    // output helper array: Store row indices for output rows
    // since the input is sorted by edge.first == row, this will be sorted
    vector<IndexType> out_rows(size, exec);
    IndexType output_count{};
    // Kruskal algorithm: Connect unconnected components using edges with
    // ascending weight
    for (const auto edge : edges) {
        const auto first_rep = sets.find(edge.first);
        const auto second_rep = sets.find(edge.second);
        if (first_rep != second_rep) {
            // we are only interested in the lower triangle, so we add an edge
            // max -> min
            out_rows[output_count] = edge.first;
            out_cols[output_count] = edge.second;
            output_count++;
            sets.join(first_rep, second_rep);
        }
    }
    assert(std::is_sorted(out_rows.begin(), out_rows.begin() + output_count));
    components::convert_idxs_to_ptrs(exec, out_rows.data(), output_count, size,
                                     out_row_ptrs);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_SKELETON_TREE);


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
void compute_elimination_forest_children_impl(const IndexType* parent,
                                              IndexType size,
                                              IndexType* child_ptr,
                                              IndexType* child)
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
void compute_elimination_forest_traversal_impl(
    std::shared_ptr<const Executor> host_exec, const IndexType* parent,
    const IndexType* child_ptr, const IndexType* child, IndexType size,
    IndexType* euler_walk, IndexType* euler_levels, IndexType* euler_first,
    IndexType* levels, IndexType* postorder, IndexType* inv_postorder)
{
    array<IndexType> current_child_array{host_exec,
                                         static_cast<size_type>(size + 1)};
    auto current_child = current_child_array.get_data();
    std::fill_n(current_child, size + 1, 0);
    IndexType postorder_idx{};
    IndexType euler_idx{};
    // for each tree in the elimination forest
    for (auto tree = child_ptr[size]; tree < child_ptr[size + 1]; tree++) {
        // start from the root
        const auto root = child[tree];
        auto cur_node = root;
        IndexType level{};
        euler_first[cur_node] = euler_idx;
        // traverse until we moved to the pseudo-root
        while (cur_node < size) {
            const auto first_child = child_ptr[cur_node];
            const auto num_children = child_ptr[cur_node + 1] - first_child;
            // output a single node
            euler_walk[euler_idx] = cur_node;
            euler_levels[euler_idx] = level;
            euler_idx++;
            if (current_child[cur_node] >= num_children) {
                // if this node is completed, output it
                postorder[postorder_idx] = cur_node;
                inv_postorder[cur_node] = postorder_idx;
                // this ensures we write each level only once
                levels[cur_node] = level;
                postorder_idx++;
                cur_node = parent[cur_node];
                level--;
            } else {
                // otherwise go to the next child node
                const auto old_node = cur_node;
                cur_node = child[first_child + current_child[old_node]];
                current_child[old_node]++;
                level++;
                euler_first[cur_node] = euler_idx;
            }
        }
    }
    // fill the remaining elements with sentinels
    std::fill(euler_walk + euler_idx, euler_walk + 2 * size - 1, -1);
}


template <typename IndexType>
void compute_elimination_forest_postorder_parent_impl(
    const IndexType* parent, const IndexType* inv_postorder, IndexType size,
    IndexType* postorder_parent)
{
    for (IndexType row = 0; row < size; row++) {
        postorder_parent[inv_postorder[row]] =
            parent[row] == size ? size : inv_postorder[parent[row]];
    }
}


template <typename IndexType>
void compute(std::shared_ptr<const DefaultExecutor> exec,
             const IndexType* row_ptrs, const IndexType* cols, size_type size,
             gko::factorization::elimination_forest<IndexType>& forest)
{
    const auto ssize = static_cast<IndexType>(size);
    compute_elimination_forest_parent_impl(exec, row_ptrs, cols, ssize,
                                           forest.parents.get_data());
    compute_elimination_forest_children_impl(
        forest.parents.get_const_data(), ssize, forest.child_ptrs.get_data(),
        forest.children.get_data());
    compute_elimination_forest_traversal_impl(
        exec, forest.parents.get_const_data(),
        forest.child_ptrs.get_const_data(), forest.children.get_const_data(),
        ssize, forest.euler_walk.get_data(), forest.euler_levels.get_data(),
        forest.euler_first.get_data(), forest.levels.get_data(),
        forest.postorder.get_data(), forest.inv_postorder.get_data());
    compute_elimination_forest_postorder_parent_impl(
        forest.parents.get_const_data(), forest.inv_postorder.get_const_data(),
        ssize, forest.postorder_parents.get_data());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ELIMINATION_FOREST_COMPUTE);


template <typename ValueType, typename IndexType>
void from_factor(std::shared_ptr<const DefaultExecutor> exec,
                 const matrix::Csr<ValueType, IndexType>* factors,
                 IndexType* parents)
{
    const auto row_ptrs = factors->get_const_row_ptrs();
    const auto col_idxs = factors->get_const_col_idxs();
    const auto num_rows = static_cast<IndexType>(factors->get_size()[0]);
    // filled with sentinel for unattached nodes
    std::fill_n(parents, num_rows, num_rows);
    for (IndexType row = 0; row < num_rows; row++) {
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        for (auto nz = row_begin; nz < row_end; nz++) {
            const auto col = col_idxs[nz];
            // use the lower triangle, min row from column
            if (col < row) {
                parents[col] = std::min(parents[col], row);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_FROM_FACTOR);


}  // namespace elimination_forest
}  // namespace reference
}  // namespace kernels
}  // namespace gko
