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
void compute_children(std::shared_ptr<const DefaultExecutor> exec,
                      const IndexType* parents, IndexType size,
                      IndexType* child_ptrs, IndexType* children)
{
    // count how many times each parent occurs, excluding pseudo-root at
    // parent == size
    std::fill_n(child_ptrs, size + 2, IndexType{});
    for (IndexType i = 0; i < size; i++) {
        const auto p = parents[i];
        if (p < size) {
            child_ptrs[p + 2]++;
        }
    }
    // shift by 2 leads to exclusive prefix sum with 0 padding
    std::partial_sum(child_ptrs, child_ptrs + size + 2, child_ptrs);
    // we count the same again, this time shifted by 1 => exclusive prefix sum
    for (IndexType i = 0; i < size; i++) {
        const auto p = parents[i];
        children[child_ptrs[p + 1]] = i;
        child_ptrs[p + 1]++;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_CHILDREN);


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
    const auto parent = forest.parents.get_data();
    const auto ssize = static_cast<IndexType>(size);
    disjoint_sets<IndexType> subtrees{exec, ssize};
    array<IndexType> subtree_root_array{exec, size};
    // pseudo-root one past the last row to deal with disconnected matrices
    const auto unattached = size;
    auto subtree_root = subtree_root_array.get_data();
    for (const auto row : irange{ssize}) {
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


template <typename IndexType>
void compute_subtree_sizes(std::shared_ptr<const DefaultExecutor> exec,
                           const IndexType* child_ptrs,
                           const IndexType* children, IndexType size,
                           IndexType* subtree_sizes)
{
    vector<bool> finished(size, exec);
    for (const auto node : irange{size}) {
        IndexType local_size{1};
        const auto child_begin = child_ptrs[node];
        const auto child_end = child_ptrs[node + 1];
        assert(child_begin <= child_end);
        for (const auto child_idx : irange{child_begin, child_end}) {
            const auto child = children[child_idx];
            assert(finished[child]);
            local_size += subtree_sizes[child];
        }
        subtree_sizes[node] = local_size;
        finished[node] = true;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_SUBTREE_SIZES);


template <typename IndexType>
void compute_subtree_euler_path_sizes(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* child_ptrs,
    const IndexType* children, IndexType size,
    IndexType* subtree_euler_path_sizes)
{
    vector<bool> finished(size, exec);
    for (const auto node : irange{size}) {
        IndexType local_size{};
        const auto child_begin = child_ptrs[node];
        const auto child_end = child_ptrs[node + 1];
        assert(child_begin <= child_end);
        for (const auto child_idx : irange{child_begin, child_end}) {
            const auto child = children[child_idx];
            assert(finished[child]);
            // euler path: follow the edge into the subtree, traverse, follow
            // edge back
            local_size += subtree_euler_path_sizes[child] + 2;
        }
        subtree_euler_path_sizes[node] = local_size;
        finished[node] = true;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_SUBTREE_EULER_PATH_SIZES);


template <typename IndexType>
void compute_levels(std::shared_ptr<const DefaultExecutor> exec,
                    const IndexType* parents, IndexType size, IndexType* levels)
{
    constexpr IndexType sentinel = -2;
    std::fill_n(levels, size, sentinel);
    for (const auto i : irange{size}) {
        auto cur = i;
        auto parent = parents[cur];
        auto parent_level = parent == size ? IndexType{-1} : levels[parent];
        IndexType delta{1};
        // walk up until we find a node we know the level of
        while (parent_level == sentinel) {
            cur = parent;
            parent = parents[cur];
            parent_level = parent == size ? IndexType{-1} : levels[parent];
            delta++;
        }
        // walk up from the original node to set the level for all nodes
        cur = i;
        while (cur != parent) {
            levels[cur] = parent_level + delta;
            cur = parents[cur];
            delta--;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_LEVELS);


template <typename IndexType>
IndexType traverse_postorder(const IndexType* child_ptrs,
                             const IndexType* children, IndexType node,
                             IndexType index, IndexType* postorder,
                             IndexType* inv_postorder)
{
    const auto child_begin = child_ptrs[node];
    const auto child_end = child_ptrs[node + 1];
    for (const auto child_idx : irange{child_begin, child_end}) {
        const auto child = children[child_idx];
        index = traverse_postorder(child_ptrs, children, child, index,
                                   postorder, inv_postorder);
    }
    postorder[index] = node;
    inv_postorder[node] = index;
    return index + 1;
}


template <typename IndexType>
void compute_postorder(std::shared_ptr<const DefaultExecutor> exec,
                       const IndexType* child_ptrs, const IndexType* children,
                       IndexType size, const IndexType* subtree_size,
                       IndexType* postorder, IndexType* inv_postorder)
{
    const auto root_begin = child_ptrs[size];
    const auto root_end = child_ptrs[size + 1];
    IndexType index{};
    for (const auto root_idx : irange{root_begin, root_end}) {
        const auto root = children[root_idx];
        index = traverse_postorder(child_ptrs, children, root, index, postorder,
                                   inv_postorder);
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_POSTORDER);


template <typename IndexType>
void map_postorder(std::shared_ptr<const DefaultExecutor> exec,
                   const IndexType* parents, const IndexType* child_ptrs,
                   const IndexType* children, IndexType size,
                   const IndexType* inv_postorder, IndexType* postorder_parents,
                   IndexType* postorder_child_ptrs,
                   IndexType* postorder_children)
{
    // map parents and child counts
    for (const auto i : irange{size}) {
        const auto postorder_i = inv_postorder[i];
        const auto parent = parents[i];
        postorder_parents[postorder_i] =
            parent == size ? size : inv_postorder[parent];
        postorder_child_ptrs[postorder_i] = child_ptrs[i + 1] - child_ptrs[i];
    }
    // we don't store a parent for the pseudo-root, but child ptrs
    postorder_child_ptrs[size] = child_ptrs[size + 1] - child_ptrs[size];
    // build postorder_child_ptrs from sizes
    components::prefix_sum_nonnegative(exec, postorder_child_ptrs,
                                       static_cast<size_type>(size + 2));
    // now map children for all nodes (including pseudo-root, thus + 1)
    for (const auto i : irange{size + 1}) {
        const auto postorder_i = i == size ? size : inv_postorder[i];
        const auto in_begin = child_ptrs[i];
        const auto in_end = child_ptrs[i + 1];
        auto out_idx = postorder_child_ptrs[postorder_i];
        for (const auto child : irange{in_begin, in_end}) {
            postorder_children[out_idx] = inv_postorder[children[child]];
            out_idx++;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_MAP_POSTORDER);


template <typename IndexType>
IndexType traverse_euler_path(const IndexType* child_ptrs,
                              const IndexType* children, IndexType node,
                              IndexType index, IndexType size, IndexType level,
                              IndexType* euler_path, IndexType* euler_first,
                              IndexType* euler_level)
{
    const auto child_begin = child_ptrs[node];
    const auto child_end = child_ptrs[node + 1];
    euler_path[index] = node;
    if (node < size) {
        euler_first[node] = index;
    }
    euler_level[index] = level;
    index++;
    for (const auto child_idx : irange{child_begin, child_end}) {
        const auto child = children[child_idx];
        index = traverse_euler_path(child_ptrs, children, child, index, size,
                                    level + 1, euler_path, euler_first,
                                    euler_level);
        euler_path[index] = node;
        euler_level[index] = level;
        index++;
    }
    return index;
}


template <typename IndexType>
void compute_euler_path(std::shared_ptr<const DefaultExecutor> exec,
                        const IndexType* child_ptrs, const IndexType* children,
                        IndexType size,
                        const IndexType* subtree_euler_tree_size,
                        const IndexType* levels, IndexType* euler_path,
                        IndexType* first_visit, IndexType* euler_levels)
{
    const auto pseudo_root = size;
    traverse_euler_path(child_ptrs, children, pseudo_root, IndexType{}, size,
                        IndexType{-1}, euler_path, first_visit, euler_levels);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_EULER_PATH);


template <typename IndexType>
void pointer_double(std::shared_ptr<const DefaultExecutor> exec,
                    const IndexType* input, IndexType size, IndexType* output)
{
    for (const auto i : irange{size}) {
        const auto target = input[i];
        output[i] = target == size ? size : input[target];
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_POINTER_DOUBLE);


}  // namespace elimination_forest
}  // namespace reference
}  // namespace kernels
}  // namespace gko
