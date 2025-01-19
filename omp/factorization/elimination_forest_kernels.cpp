// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/elimination_forest.hpp"

#include <algorithm>
#include <memory>

#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/allocator.hpp"
#include "core/base/index_range.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/factorization/elimination_forest_kernels.hpp"


namespace gko {
namespace kernels {
namespace omp {
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


template <typename ValueType, typename IndexType>
void from_factor(std::shared_ptr<const DefaultExecutor> exec,
                 const matrix::Csr<ValueType, IndexType>* factors,
                 gko::factorization::elimination_forest<IndexType>& forest)
{
    const auto row_ptrs = factors->get_const_row_ptrs();
    const auto col_idxs = factors->get_const_col_idxs();
    const auto parents = forest.parents.get_data();
    const auto children = forest.children.get_data();
    const auto child_ptrs = forest.child_ptrs.get_data();
    const auto num_rows = static_cast<IndexType>(factors->get_size()[0]);
    components::fill_array(exec, parents, num_rows, num_rows);
#pragma omp parallel for
    for (IndexType l_col = 0; l_col < num_rows; l_col++) {
        const auto llt_row_begin = row_ptrs[l_col];
        const auto llt_row_end = row_ptrs[l_col + 1];
        for (auto nz = llt_row_begin; nz < llt_row_end; nz++) {
            const auto l_row = col_idxs[nz];
            // parent[j] = min(i | i > j and l_ij =/= 0)
            // we read from L^T stored above the diagonal in factors
            // assuming a sorted order of the columns
            if (l_row > l_col) {
                parents[l_col] = l_row;
                break;
            }
        }
    }
    // group by parent
    array<IndexType> parents_copy{exec, static_cast<size_type>(num_rows)};
    exec->copy(num_rows, parents, parents_copy.get_data());
    components::fill_seq_array(exec, children, num_rows);
    const auto it =
        detail::make_zip_iterator(parents_copy.get_data(), children);
    std::stable_sort(it, it + num_rows);
    components::convert_idxs_to_ptrs(exec, parents_copy.get_const_data(),
                                     num_rows, num_rows + 1, child_ptrs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_FROM_FACTOR);


template <typename IndexType>
void compute_subtree_sizes(
    std::shared_ptr<const DefaultExecutor> exec,
    const gko::factorization::elimination_forest<IndexType>& forest,
    IndexType* subtree_sizes)
{
    const auto size = static_cast<IndexType>(forest.parents.get_size());
    const auto child_ptrs = forest.child_ptrs.get_const_data();
    const auto children = forest.children.get_const_data();
    for (const auto node : irange{size}) {
        IndexType local_size{1};
        const auto child_begin = child_ptrs[node];
        const auto child_end = child_ptrs[node + 1];
        for (const auto child_idx : irange{child_begin, child_end}) {
            const auto child = children[child_idx];
            local_size += subtree_sizes[child];
        }
        subtree_sizes[node] = local_size;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_SUBTREE_SIZES);


template <typename IndexType>
void compute_subtree_euler_path_sizes(
    std::shared_ptr<const DefaultExecutor> exec,
    const gko::factorization::elimination_forest<IndexType>& forest,
    IndexType* subtree_euler_path_sizes)
{
    const auto size = static_cast<IndexType>(forest.parents.get_size());
    const auto child_ptrs = forest.child_ptrs.get_const_data();
    const auto children = forest.children.get_const_data();
    for (const auto node : irange{size}) {
        IndexType local_size{};
        const auto child_begin = child_ptrs[node];
        const auto child_end = child_ptrs[node + 1];
        for (const auto child_idx : irange{child_begin, child_end}) {
            const auto child = children[child_idx];
            // euler path: follow the edge into the subtree, traverse, follow
            // edge back
            local_size += subtree_euler_path_sizes[child] + 2;
        }
        subtree_euler_path_sizes[node] = local_size;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_SUBTREE_EULER_PATH_SIZES);


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
void compute_postorder(
    std::shared_ptr<const DefaultExecutor> exec,
    const gko::factorization::elimination_forest<IndexType>& forest,
    const IndexType* subtree_size, IndexType* postorder,
    IndexType* inv_postorder)
{
    const auto child_ptrs = forest.child_ptrs.get_const_data();
    const auto children = forest.children.get_const_data();
    const auto size = static_cast<IndexType>(forest.parents.get_size());
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
void compute_euler_path(
    std::shared_ptr<const DefaultExecutor> exec,
    const gko::factorization::elimination_forest<IndexType>& forest,
    const IndexType* subtree_euler_tree_size, const IndexType* levels,
    IndexType* euler_path, IndexType* first_visit, IndexType* euler_levels)
{
    const auto child_ptrs = forest.child_ptrs.get_const_data();
    const auto children = forest.children.get_const_data();
    const auto size = static_cast<IndexType>(forest.parents.get_size());
    const auto pseudo_root = size;
    traverse_euler_path(child_ptrs, children, pseudo_root, IndexType{}, size,
                        IndexType{-1}, euler_path, first_visit, euler_levels);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_EULER_PATH);


template <typename IndexType>
void compute_levels(
    std::shared_ptr<const DefaultExecutor> exec,
    const gko::factorization::elimination_forest<IndexType>& forest,
    IndexType* levels)
{
    const auto size = static_cast<IndexType>(forest.parents.get_size());
    const auto parents = forest.parents.get_const_data();
    for (auto node = size - 1; node >= 0; node--) {
        const auto parent = parents[node];
        // root nodes are attached to pseudo-root at index ssize
        levels[node] = parent == size ? IndexType{} : levels[parent] + 1;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_LEVELS);


}  // namespace elimination_forest
}  // namespace omp
}  // namespace kernels
}  // namespace gko
