// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/elimination_forest.hpp"

#include <algorithm>
#include <memory>

#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/allocator.hpp"
#include "core/base/index_range.hpp"
#include "core/base/intrinsics.hpp"
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


template <typename IndexType>
void compute(std::shared_ptr<const DefaultExecutor> exec,
             const IndexType* row_ptrs, const IndexType* cols, size_type size,
             gko::factorization::elimination_forest<IndexType>& forest)
{
    if (size == 0) {
        return;
    }
    using unsigned_type = std::make_unsigned_t<IndexType>;
    const auto ssize = static_cast<IndexType>(size);
    std::vector<std::pair<IndexType, IndexType>> edges;
    for (const auto row : irange{ssize}) {
        for (const auto nz : irange{row_ptrs[row], row_ptrs[row + 1]}) {
            const auto col = cols[nz];
            if (col < row) {
                edges.emplace_back(col, row);
            }
        }
    }
    // round up size to the next power of two
    const auto rounded_up_size =
        IndexType{1}
        << (detail::find_highest_bit(static_cast<unsigned_type>(size - 1)) + 1);
    // insert fill-in edges top-down
    for (auto block_size = rounded_up_size; block_size > 1; block_size /= 2) {
        const auto half_block_size = block_size / 2;
        const auto is_inner_edge = [&](auto e) {
            assert(e.first < e.second);
            return e.first / half_block_size == e.second / half_block_size;
        };
        const auto is_cut_edge = [&](auto e) {
            assert(e.first < e.second);
            return e.first / block_size == e.second / block_size &&
                   e.first / half_block_size < e.second / half_block_size;
        };
        disjoint_sets<IndexType> cc{exec, ssize};
        for (auto edge : edges) {
            // join edges inside blocks of size half_block_size
            if (is_inner_edge(edge)) {
                cc.join(edge.first, edge.second);
            }
        }
        // now find the smallest upper node adjacent to a cc in a lower block
        std::vector<IndexType> mins(size, ssize);
        for (auto edge : edges) {
            if (is_cut_edge(edge)) {
                const auto first_rep = cc.find(edge.first);
                mins[first_rep] = std::min(mins[first_rep], edge.second);
            }
        }
        std::vector<std::pair<IndexType, IndexType>> new_edges;
        // now add new edges for every one of those cut edges
        for (auto edge : edges) {
            if (is_cut_edge(edge)) {
                const auto first_rep = cc.find(edge.first);
                const auto min_neighbor = mins[first_rep];
                if (min_neighbor != edge.second) {
                    new_edges.emplace_back(min_neighbor, edge.second);
                }
            }
        }
        edges.insert(edges.end(), new_edges.begin(), new_edges.end());
    }
    // compute elimination forest bottom-up
    disjoint_sets<IndexType> cc{exec, ssize};
    std::vector<IndexType> subtree_roots(size);
    std::vector<std::pair<IndexType, IndexType>> tree_edges;
    std::iota(subtree_roots.begin(), subtree_roots.end(), IndexType{});
    for (IndexType block_size = 2; block_size <= rounded_up_size;
         block_size *= 2) {
        std::vector<IndexType> mins(size, ssize);
        const auto half_block_size = block_size / 2;
        const auto is_inner_edge = [&](auto e) {
            assert(e.first < e.second);
            return e.first / half_block_size == e.second / half_block_size;
        };
        const auto is_cut_edge = [&](auto e) {
            assert(e.first < e.second);
            return e.first / block_size == e.second / block_size &&
                   e.first / half_block_size < e.second / half_block_size;
        };
        // reproduce CC again, this time with subtree roots
        for (auto edge : edges) {
            if (is_inner_edge(edge)) {
                const auto first_rep = cc.find(edge.first);
                const auto second_rep = cc.find(edge.second);
                const auto combined_rep = cc.join(first_rep, second_rep);
                subtree_roots[combined_rep] = std::max(
                    subtree_roots[first_rep], subtree_roots[second_rep]);
            }
        }
        for (auto edge : edges) {
            if (is_cut_edge(edge)) {
                const auto first_rep = cc.find(edge.first);
                mins[first_rep] = std::min(mins[first_rep], edge.second);
            }
        }
        for (auto node : irange{ssize}) {
            // for every connected component: insert an edge from its root to
            // the minimal adjacent node
            if ((node / half_block_size) % 2 == 0 &&
                cc.is_representative(node)) {
                tree_edges.emplace_back(subtree_roots[node], mins[node]);
            }
        }
    }
    // translate to parents
    const auto parents = forest.parents.get_data();
    std::fill_n(parents, ssize, ssize);
    for (auto tree_edge : tree_edges) {
        assert(parents[tree_edge.first] == ssize);
        parents[tree_edge.first] = tree_edge.second;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ELIMINATION_FOREST_COMPUTE);


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


}  // namespace elimination_forest
}  // namespace omp
}  // namespace kernels
}  // namespace gko
