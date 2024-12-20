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
#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/elimination_forest_kernels.hpp"
#include "omp/components/atomic.hpp"
#include "omp/components/disjoint_sets.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace elimination_forest {


template <typename IndexType>
void sort_edges(std::shared_ptr<const DefaultExecutor> exec, IndexType* sources,
                IndexType* targets, IndexType num_edges)
{
    const auto it = detail::make_zip_iterator(sources, targets);
    std::sort(it, it + num_edges);
}


// This is a minimum spanning tree algorithm implementation based on
// A. Fallin, A. Gonzalez, J. Seo, and M. Burtscher,
// "A High-Performance MST Implementation for GPUs,”
// doi: 10.1145/3581784.3607093
template <typename IndexType>
struct mst_state {
    static size_type storage_requirement(IndexType num_nodes,
                                         IndexType num_edges)
    {
        return 6 * static_cast<size_type>(num_edges) +
               2 * static_cast<size_type>(num_nodes);
    }

    mst_state(std::shared_ptr<const DefaultExecutor> exec, IndexType num_nodes,
              IndexType num_edges, IndexType* input_sources,
              IndexType* input_targets, IndexType* tree_sources,
              IndexType* tree_targets)
        : exec_{exec},
          num_nodes_{num_nodes},
          num_edges_{num_edges},
          tree_counter_{},
          counter1_{},
          counter2_{},
          input_sources_{input_sources},
          input_targets_{input_targets},
          work_array_{exec, storage_requirement(num_nodes, num_edges)},
          tree_sources_{tree_sources},
          tree_targets_{tree_targets},
          flip_{}
    {
        reset(input_sources, input_targets_, num_edges_);
    }

    const IndexType* input_worklist()
    {
        return work_array_.get_const_data() + (flip_ ? 3 * num_edges_ : 0);
    }

    IndexType* output_worklist()
    {
        return work_array_.get_data() + (flip_ ? 0 : 3 * num_edges_);
    }

    const IndexType* input_wl_sources() { return input_worklist(); }

    const IndexType* input_wl_targets()
    {
        return input_wl_sources() + num_edges_;
    }

    const IndexType* input_wl_edge_ids()
    {
        return input_wl_targets() + num_edges_;
    }

    IndexType* output_wl_sources() { return output_worklist(); }

    IndexType* output_wl_targets() { return output_wl_sources() + num_edges_; }

    IndexType* output_wl_edge_ids() { return output_wl_targets() + num_edges_; }

    IndexType* parents() { return work_array_.get_data() + 6 * num_edges_; }

    IndexType* min_edges() { return parents() + num_nodes_; }

    IndexType& tree_counter() { return tree_counter_; }

    const IndexType& input_counter() { return (flip_ ? counter1_ : counter2_); }

    IndexType& output_counter() { return (flip_ ? counter2_ : counter1_); }

    void output_to_input()
    {
        flip_ = !flip_;
        output_counter() = 0;
    }

    IndexType input_size() { return input_counter(); }

    IndexType tree_size() { return tree_counter(); }

    void reset(IndexType* new_input_sources, IndexType* new_input_targets,
               IndexType new_num_edges)
    {
        if (new_num_edges > num_edges_) {
            num_edges_ = new_num_edges;
            work_array_.resize_and_reset(
                storage_requirement(num_nodes_, num_edges_));
        }
        input_sources_ = new_input_sources;
        input_targets_ = new_input_targets;
        tree_counter() = 0;
        output_counter() = 0;
        components::fill_array(exec_, min_edges(),
                               static_cast<size_type>(num_nodes_),
                               min_edge_sentinel);
        components::fill_seq_array(exec_, parents(),
                                   static_cast<size_type>(num_nodes_));
        exec_->copy(num_edges_, input_sources_, output_wl_sources());
        exec_->copy(num_edges_, input_targets_, output_wl_targets());
        components::fill_seq_array(exec_, output_wl_edge_ids(),
                                   static_cast<size_type>(num_edges_));
        output_counter() = num_edges_;
        output_to_input();
    }

    void find_min_edges()
    {
        device_disjoint_sets<IndexType> sets{parents(), num_nodes_};
#pragma omp parallel for
        for (IndexType i = 0; i < input_size(); i++) {
            // attach each node to its smallest adjacent non-cycle edge
            const auto source = input_wl_sources()[i];
            const auto target = input_wl_targets()[i];
            const auto edge_id = input_wl_edge_ids()[i];
            const auto source_rep = sets.find_weak(source);
            const auto target_rep = sets.find_weak(target);
            if (source_rep != target_rep) {
                const auto output_idx = atomic_inc(output_counter());
                output_wl_sources()[output_idx] = source_rep;
                output_wl_targets()[output_idx] = target_rep;
                output_wl_edge_ids()[output_idx] = edge_id;
                atomic_min(min_edges() + source_rep, edge_id);
                atomic_min(min_edges() + target_rep, edge_id);
            }
        }
    }

    void join_min_edges()
    {
        device_disjoint_sets<IndexType> sets{parents(), num_nodes_};
#pragma omp parallel for
        for (IndexType i = 0; i < input_size(); i++) {
            // join minimal edges
            const auto source = input_wl_sources()[i];
            const auto target = input_wl_targets()[i];
            const auto edge_id = input_wl_edge_ids()[i];
            if (min_edges()[source] == edge_id ||
                min_edges()[target] == edge_id) {
                // join source and sink
                const auto source_rep = sets.find_relaxed(source);
                const auto target_rep = sets.find_relaxed(target);
                assert(source_rep != target_rep);
                sets.join(source_rep, target_rep);
                const auto out_i = atomic_inc(tree_counter());
                tree_sources_[out_i] = input_sources_[edge_id];
                tree_targets_[out_i] = input_targets_[edge_id];
            }
        }
    }

    void reset_min_edges()
    {
        if (input_size() < num_nodes_ / 8) {
            // if there are only a handful of min_edges values to reset:
            // do it individually
#pragma omp parallel for
            for (IndexType i = 0; i < input_size(); i++) {
                // join minimal edges
                const auto source = input_wl_sources()[i];
                const auto target = input_wl_targets()[i];
#pragma omp atomic write
                min_edges()[source] = min_edge_sentinel;
#pragma omp atomic write
                min_edges()[target] = min_edge_sentinel;
            }
        } else {
            // otherwise reset the entire array
            components::fill_array(exec_, min_edges(),
                                   static_cast<size_type>(num_nodes_),
                                   min_edge_sentinel);
        }
    }

    void run()
    {
        while (input_size() > 0) {
            find_min_edges();
            output_to_input();
            if (input_size() > 0) {
                join_min_edges();
                reset_min_edges();
            }
        }
    }

    void sort_input_edges()
    {
        sort_edges(exec_, input_sources_, input_targets_, num_edges_);
    }

    void sort_tree_edges()
    {
        sort_edges(exec_, tree_sources_, tree_targets_, tree_size());
    }

    constexpr static auto min_edge_sentinel =
        std::numeric_limits<IndexType>::max();
    std::shared_ptr<const DefaultExecutor> exec_;
    IndexType tree_counter_;
    IndexType counter1_;
    IndexType counter2_;
    IndexType num_nodes_;
    IndexType num_edges_;
    IndexType* input_sources_;
    IndexType* input_targets_;
    IndexType* tree_sources_;
    IndexType* tree_targets_;
    array<IndexType> work_array_;
    bool flip_;
};


template <typename IndexType>
std::pair<array<IndexType>, array<IndexType>> extract_lower_triangular(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* row_ptrs,
    const IndexType* cols, size_type size)
{
    array<IndexType> source_array{exec};
    array<IndexType> target_array{exec};

    array<IndexType> out_ptr_array{exec, size + 1};
    const auto out_ptrs = out_ptr_array.get_data();
    const auto ssize = static_cast<IndexType>(size);
#pragma omp parallel for
    for (IndexType row = 0; row < ssize; row++) {
        const auto begin = row_ptrs[row];
        const auto end = row_ptrs[row + 1];
        IndexType count{};
        for (auto nz : irange{begin, end}) {
            const auto col = cols[nz];
            count += col < row ? 1 : 0;
        }
        out_ptrs[row] = count;
    }
    components::prefix_sum_nonnegative(exec, out_ptrs, size + 1);
    const auto num_edges = out_ptrs[ssize];
    source_array.resize_and_reset(num_edges);
    target_array.resize_and_reset(num_edges);
    const auto sources = source_array.get_data();
    const auto targets = target_array.get_data();
#pragma omp parallel for
    for (IndexType row = 0; row < ssize; row++) {
        const auto begin = row_ptrs[row];
        const auto end = row_ptrs[row + 1];
        auto out_idx = out_ptrs[row];
        for (auto nz : irange{begin, end}) {
            const auto col = cols[nz];
            if (col < row) {
                sources[out_idx] = row;
                targets[out_idx] = col;
                out_idx++;
            }
        }
    }
    return std::make_pair(std::move(source_array), std::move(target_array));
}


template <typename IndexType>
void compute_skeleton_tree(std::shared_ptr<const DefaultExecutor> exec,
                           const IndexType* row_ptrs, const IndexType* cols,
                           size_type size, IndexType* out_row_ptrs,
                           IndexType* out_cols)
{
    const auto ssize = static_cast<IndexType>(size);
    // we don't filter heavy edges since the heaviest edges are necessary to
    // reach the last node and we don't need to sort since the COO format
    // already sorts by row index.
    auto [sources, targets] =
        extract_lower_triangular(exec, row_ptrs, cols, size);
    array<IndexType> out_row_array{exec, size - 1};
    mst_state<IndexType> state{exec,
                               ssize,
                               static_cast<IndexType>(sources.get_size()),
                               sources.get_data(),
                               targets.get_data(),
                               out_row_array.get_data(),
                               out_cols};
    state.run();
    state.sort_tree_edges();
    components::convert_idxs_to_ptrs(exec, out_row_array.get_data(),
                                     state.tree_size(), size, out_row_ptrs);
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
            assert(finished[child]);
            local_size += subtree_sizes[child];
        }
        subtree_sizes[node] = local_size;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_SUBTREE_SIZES);


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
