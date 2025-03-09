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
#include "omp/components/atomic.hpp"
#include "omp/components/disjoint_sets.hpp"


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
    // This is a minimum spanning tree algorithm implementation based on
    // A. Fallin, A. Gonzalez, J. Seo, and M. Burtscher,
    // "A High-Performance MST Implementation for GPUs,”
    // doi: 10.1145/3581784.3607093
    // we don't filter heavy edges since the heaviest edges are necessary to
    // reach the last node and we don't need to sort since the COO format
    // already sorts by row index.
    const auto nnz = row_ptrs[size];
    const auto ssize = static_cast<IndexType>(size);
    // convert edges to COO representation
    // the edge list is sorted, since we only consider edges where row > col,
    // and the row array (= weights) is sorted coming from row_ptrs
    array<IndexType> row_array{exec, static_cast<size_type>(nnz)};
    const auto rows = row_array.get_data();
    components::convert_ptrs_to_idxs(exec, row_ptrs, size, rows);
    // we assume the matrix is symmetric, so we can remove every second edge
    const auto worklist_size = ceildiv(nnz, 2);
    // create 2 worklists consisting of (start, end, edge_id)
    array<IndexType> worklist{exec, static_cast<size_type>(worklist_size * 6)};
    auto wl1_source = worklist.get_data();
    auto wl1_target = wl1_source + worklist_size;
    auto wl1_edge_id = wl1_target + worklist_size;
    auto wl2_source = wl1_source + 3 * worklist_size;
    auto wl2_target = wl1_target + 3 * worklist_size;
    auto wl2_edge_id = wl1_edge_id + 3 * worklist_size;
    // atomic counters for worklists and output edge list
    IndexType wl1_counter{};
    IndexType wl2_counter{};
    IndexType output_counter{};
    // helpers for interacting with worklists
    const auto swap_wl1_wl2 = [&] {
        std::swap(wl1_source, wl2_source);
        std::swap(wl1_target, wl2_target);
        std::swap(wl1_edge_id, wl2_edge_id);
        std::swap(wl1_counter, wl2_counter);
    };
    // initialize every node to a singleton set
    array<IndexType> parent_array{exec, size};
    const auto parents = parent_array.get_data();
    components::fill_seq_array(exec, parents, size);
    // array storing the minimal edge adjacent to each node
    array<IndexType> min_edge_array{exec, size};
    const auto min_edges = min_edge_array.get_data();
    constexpr auto min_edge_sentinel = std::numeric_limits<IndexType>::max();
    components::fill_array(exec, min_edges, size, min_edge_sentinel);
    // output row array, to be used in conjunction with out_cols in COO storage
    array<IndexType> out_row_array{exec, size};
    const auto out_rows = out_row_array.get_data();
#pragma omp parallel for shared(wl1_counter)
    for (IndexType i = 0; i < nnz; i++) {
        // initialize worklist1 with forward edges
        const auto row = rows[i];
        const auto col = cols[i];
        if (col < row) {
            const auto output_idx = atomic_inc(wl1_counter);
            wl1_source[output_idx] = row;
            wl1_target[output_idx] = col;
            wl1_edge_id[output_idx] = i;
        }
    }
    while (wl1_counter > 0) {
        wl2_counter = 0;
        device_disjoint_sets<IndexType> sets{parents, ssize};
#pragma omp parallel for shared(wl2_counter)
        for (IndexType i = 0; i < wl1_counter; i++) {
            // attach each node to its smallest adjacent non-cycle edge
            const auto source = wl1_source[i];
            const auto target = wl1_target[i];
            const auto edge_id = wl1_edge_id[i];
            const auto source_rep = sets.find_weak(source);
            const auto target_rep = sets.find_weak(target);
            if (source_rep != target_rep) {
                const auto output_idx = atomic_inc(wl2_counter);
                wl2_source[output_idx] = source_rep;
                wl2_target[output_idx] = target_rep;
                wl2_edge_id[output_idx] = edge_id;
                atomic_min(min_edges + source_rep, edge_id);
                atomic_min(min_edges + target_rep, edge_id);
            }
        }
        wl1_counter = 0;
        swap_wl1_wl2();
        if (wl1_counter > 0) {
#pragma omp parallel for shared(output_counter)
            for (IndexType i = 0; i < wl1_counter; i++) {
                // join minimal edges
                const auto source = wl1_source[i];
                const auto target = wl1_target[i];
                const auto edge_id = wl1_edge_id[i];
                if (min_edges[source] == edge_id ||
                    min_edges[target] == edge_id) {
                    // join source and sink
                    const auto source_rep = sets.find_relaxed(source);
                    const auto target_rep = sets.find_relaxed(target);
                    assert(source_rep != target_rep);
                    sets.join(source_rep, target_rep);
                    const auto out_i = atomic_inc(output_counter);
                    out_rows[out_i] = rows[edge_id];
                    out_cols[out_i] = cols[edge_id];
                }
            }
#pragma omp parallel for
            for (IndexType i = 0; i < wl1_counter; i++) {
                // join minimal edges
                const auto source = wl1_source[i];
                const auto target = wl1_target[i];
#pragma omp atomic write
                min_edges[source] = min_edge_sentinel;
#pragma omp atomic write
                min_edges[target] = min_edge_sentinel;
            }
        }
    }
    const auto it = detail::make_zip_iterator(out_rows, out_cols);
    std::sort(it, it + output_counter);
    components::convert_idxs_to_ptrs(exec, out_rows, output_counter, size,
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
