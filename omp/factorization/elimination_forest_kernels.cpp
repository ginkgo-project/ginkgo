// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/elimination_forest_kernels.hpp"

#include <algorithm>
#include <memory>

#include <omp.h>

#include <ginkgo/core/log/profiler_hook.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/allocator.hpp"
#include "core/base/index_range.hpp"
#include "core/base/intrinsics.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/combined_workspace.hpp"
#include "core/components/double_buffer.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/elimination_forest_kernels.hpp"
#include "omp/components/atomic.hpp"
#include "omp/components/disjoint_sets.hpp"
#include "omp/components/sorting.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace elimination_forest {


struct OperationScopeGuard : Operation {
    void run(std::shared_ptr<const OmpExecutor> exec) const override
        GKO_NOT_IMPLEMENTED;
    void run(std::shared_ptr<const ReferenceExecutor> exec) const override
        GKO_NOT_IMPLEMENTED;
    void run(std::shared_ptr<const CudaExecutor> exec) const override
        GKO_NOT_IMPLEMENTED;
    void run(std::shared_ptr<const HipExecutor> exec) const override
        GKO_NOT_IMPLEMENTED;
    void run(std::shared_ptr<const DpcppExecutor> exec) const override
        GKO_NOT_IMPLEMENTED;
    const char* get_name() const noexcept override { return name.c_str(); }

    OperationScopeGuard(std::string name, std::shared_ptr<const Executor> exec)
        : name{std::move(name)}, exec{exec}
    {
        for (auto& log : exec->get_loggers()) {
            if (auto profiler =
                    std::dynamic_pointer_cast<const log::ProfilerHook>(log)) {
                profiler->on_operation_launched(exec.get(), this);
            }
        }
    }

    OperationScopeGuard(const OperationScopeGuard&) = delete;
    OperationScopeGuard(OperationScopeGuard&&) = delete;
    OperationScopeGuard& operator=(const OperationScopeGuard&) = delete;
    OperationScopeGuard& operator=(OperationScopeGuard&&) = delete;

    ~OperationScopeGuard()
    {
        for (auto& log : exec->get_loggers()) {
            if (auto profiler =
                    std::dynamic_pointer_cast<const log::ProfilerHook>(log)) {
                profiler->on_operation_completed(exec.get(), this);
            }
        }
    }

    std::string name;
    std::shared_ptr<const Executor> exec;
};

#define GKO_FUNCTION_SCOPEGUARD(_name) \
    OperationScopeGuard guard { #_name, exec }


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
    static std::vector<size_type> storage_sizes(size_type num_nodes,
                                                size_type num_edges)
    {
        return {
            num_edges,  // in/out_sources
            num_edges,  // in/out_targets
            num_edges,  // in/out_edge_ids
            num_edges,  // out/in_sources
            num_edges,  // out/in_targets
            num_edges,  // out/in_edge_ids
            num_nodes,  // min_edges
            num_nodes   // parents
        };
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
          workspace_{exec, storage_sizes(num_nodes, num_edges)},
          tree_sources_{tree_sources},
          tree_targets_{tree_targets},
          worklists_sources{workspace_.get_pointer(0),
                            workspace_.get_pointer(3),
                            static_cast<size_type>(num_edges)},
          worklists_targets{workspace_.get_pointer(1),
                            workspace_.get_pointer(4),
                            static_cast<size_type>(num_edges)},
          worklists_edge_ids{workspace_.get_pointer(2),
                             workspace_.get_pointer(5),
                             static_cast<size_type>(num_edges)},
          counters{&counter1_, &counter2_, 1}
    {
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

    const IndexType* input_wl_sources() { return worklists_sources.get(); }

    const IndexType* input_wl_targets() { return worklists_targets.get(); }

    const IndexType* input_wl_edge_ids() { return worklists_edge_ids.get(); }

    IndexType* output_wl_sources() { return worklists_sources.get_other(); }

    IndexType* output_wl_targets() { return worklists_targets.get_other(); }

    IndexType* output_wl_edge_ids() { return worklists_edge_ids.get_other(); }

    IndexType* parents() { return workspace_.get_pointer(6); }

    IndexType* min_edges() { return workspace_.get_pointer(7); }

    IndexType& tree_counter() { return tree_counter_; }

    const IndexType& input_counter() { return *counters.get(); }

    IndexType& output_counter() { return *counters.get_other(); }

    void output_to_input()
    {
        worklists_sources.swap();
        worklists_targets.swap();
        worklists_edge_ids.swap();
        counters.swap();
        output_counter() = 0;
    }

    IndexType input_size() { return input_counter(); }

    IndexType tree_size() { return tree_counter(); }

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
    combined_workspace<IndexType> workspace_;
    double_buffer<IndexType> worklists_sources;
    double_buffer<IndexType> worklists_targets;
    double_buffer<IndexType> worklists_edge_ids;
    double_buffer<IndexType> counters;
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
struct elimination_forest_algorithm_state {
    constexpr static int num_buckets = CHAR_BIT * sizeof(IndexType) - 1;
    elimination_forest_algorithm_state(
        std::shared_ptr<const DefaultExecutor> exec, IndexType num_nodes,
        IndexType num_edges, IndexType* tree_levels)
        : exec{exec},
          num_nodes{num_nodes},
          edge_capacity{2 * num_edges},
          num_levels{
              gko::detail::find_highest_bit(
                  static_cast<std::make_unsigned_t<IndexType>>(num_nodes - 1)) +
              1},
          num_threads{static_cast<int>(omp_get_max_threads())},
          ceil_num_nodes{IndexType{1} << num_levels},
          workspace{exec, workspace_sizes(num_nodes, num_edges)},
          worklists_sources{workspace.get_pointer(0), workspace.get_pointer(2),
                            static_cast<size_type>(edge_capacity)},
          worklists_targets{workspace.get_pointer(1), workspace.get_pointer(3),
                            static_cast<size_type>(edge_capacity)},
          tree_sources{workspace.get_pointer(4), workspace.get_pointer(6),
                       static_cast<size_type>(num_nodes - 1)},
          tree_targets{workspace.get_pointer(5), workspace.get_pointer(7),
                       static_cast<size_type>(num_nodes - 1)},
          euler_walk{workspace.get_pointer(8), workspace.get_pointer(9),
                     static_cast<size_type>(2 * num_nodes - 1)},
          euler_sizes{workspace.get_pointer(10), workspace.get_pointer(11),
                      static_cast<size_type>(num_nodes)},
          euler_first{workspace.get_pointer(12), workspace.get_pointer(13),
                      static_cast<size_type>(num_nodes)},
          cc_sizes{workspace.get_pointer(17), workspace.get_pointer(18),
                   static_cast<size_type>(num_nodes)},
          bucket_ranges{},
          tree_ranges{},
          tree_levels{tree_levels}
    {
        bucket_ranges.back() = num_edges;
    }

    static int get_edge_level(IndexType src, IndexType tgt)
    {
        using unsigned_type = std::make_unsigned_t<IndexType>;
        assert(src != tgt);
        return gko::detail::find_highest_bit(static_cast<unsigned_type>(src) ^
                                             static_cast<unsigned_type>(tgt));
    }

    static std::vector<size_type> workspace_sizes(size_type num_nodes,
                                                  size_type num_edges)
    {
        const auto edge_capacity = 2 * num_edges;
        return {
            edge_capacity,      // 0: buf1_sources
            edge_capacity,      // 1: buf1_targets
            edge_capacity,      // 2: buf2_sources
            edge_capacity,      // 3: buf2_targets
            num_nodes - 1,      // 4: tree_sources1
            num_nodes - 1,      // 5: tree_targets1
            num_nodes - 1,      // 6: tree_sources2
            num_nodes - 1,      // 7: tree_targets2
            2 * num_nodes - 1,  // 8: euler_walk1
            2 * num_nodes - 1,  // 9: euler_walk2
            num_nodes,          // 10: euler_sizes1
            num_nodes,          // 11: euler_sizes2
            num_nodes,          // 12: euler_first1
            num_nodes,          // 13: euler_first2
            num_nodes,          // 14: euler_last
            num_nodes,          // 15: cc_parents
            num_nodes,          // 16: cc_mins
            num_nodes,          // 17: cc_sizes1
            num_nodes,          // 18: cc_sizes2
            bucket_sort_workspace_size<num_buckets>(
                edge_capacity),  // 19: bucketsort_workspace
        };
    }

    static size_type storage_requirement(size_type num_nodes,
                                         size_type num_edges)
    {
        return combined_workspace<IndexType>::get_total_size(
            workspace_sizes(num_nodes, num_edges));
    }

    IndexType* euler_last() { return workspace.get_pointer(14); }

    IndexType* cc_parents() { return workspace.get_pointer(15); }

    IndexType* cc_mins() { return workspace.get_pointer(16); }

    IndexType* bucketsort_workspace() { return workspace.get_pointer(19); }

    array<IndexType> bucketsort_workspace_view()
    {
        return workspace.get_view(19);
    }

    IndexType num_edges() const { return bucket_ranges.back(); }

    const IndexType* in_edge_sources() { return worklists_sources.get(); }

    const IndexType* in_edge_targets() { return worklists_targets.get(); }

    IndexType* out_edge_sources() { return worklists_sources.get_other(); }

    IndexType* out_edge_targets() { return worklists_targets.get_other(); }

    IndexType* fill_edge_sources()
    {
        return worklists_sources.get() + num_edges();
    }

    IndexType* fill_edge_targets()
    {
        return worklists_targets.get() + num_edges();
    }

    void output_to_input()
    {
        worklists_sources.swap();
        worklists_targets.swap();
    }

    void init(const array<IndexType>& sources, const array<IndexType>& ends)
    {
        GKO_FUNCTION_SCOPEGUARD(init);
        assert(sources.get_size() == ends.get_size());
        assert(sources.get_size() <= edge_capacity);
        std::copy_n(sources.get_const_data(), sources.get_size(),
                    out_edge_sources());
        std::copy_n(ends.get_const_data(), ends.get_size(), out_edge_targets());
        tree_counter = 0;
        bucket_ranges.back() = static_cast<IndexType>(sources.get_size());
        output_to_input();
        components::fill_array(exec, tree_levels,
                               static_cast<size_type>(num_nodes), IndexType{});
        components::fill_array(exec, cc_sizes.get(),
                               static_cast<size_type>(num_nodes), IndexType{1});
    }

    void bucket_sort_input()
    {
        GKO_FUNCTION_SCOPEGUARD(bucket_sort_input);
        auto it =
            detail::make_zip_iterator(in_edge_sources(), in_edge_targets());
        auto out_it =
            detail::make_zip_iterator(out_edge_sources(), out_edge_targets());
        auto ws = bucketsort_workspace_view();
        bucket_ranges = bucket_sort<num_buckets>(
            it, it + num_edges(), out_it,
            [](auto edge) {
                using std::get;
                return get_edge_level(edge.template get<0>(),
                                      edge.template get<1>());
            },
            ws);
        output_to_input();
    }

    irange<IndexType> get_tree_edge_range(int level) const
    {
        assert(level >= 0);
        assert(level < num_levels);
        const auto begin = tree_ranges[level + 1];
        const auto end = tree_ranges[level];
        assert(end >= 0);
        assert(begin <= end);
        assert(end <= num_nodes - 1);
        return irange<IndexType>{begin, end};
    }

    irange<IndexType> get_inner_edge_range(int level) const
    {
        assert(level >= 0);
        assert(level < num_levels);
        const auto end = bucket_ranges[level];
        assert(end >= 0);
        assert(end <= edge_capacity);
        return irange<IndexType>{end};
    }

    irange<IndexType> get_cut_edge_range(int level) const
    {
        assert(level >= 0);
        assert(level < num_levels);
        const auto begin = bucket_ranges[level];
        const auto end = bucket_ranges[level + 1];
        assert(begin >= 0);
        assert(begin <= end);
        assert(end <= edge_capacity);
        return irange<IndexType>{begin, end};
    }

    template <typename Op>
    void foreach_edge_in_range_enumerated(irange<IndexType> range, Op op)
    {
        const auto srcs = in_edge_sources();
        const auto tgts = in_edge_targets();
#pragma omp parallel for
        for (auto i = range.begin_index(); i < range.end_index(); i++) {
            op(i - range.begin_index(), srcs[i], tgts[i]);
        }
    }

    template <typename Op>
    void foreach_edge_in_range(irange<IndexType> range, Op op)
    {
        foreach_edge_in_range_enumerated(
            range, [op](auto, auto src, auto tgt) { op(src, tgt); });
    }

    template <typename Op>
    void foreach_tree_edge_in_range(irange<IndexType> range, Op op)
    {
        const auto srcs = tree_sources.get();
        const auto tgts = tree_targets.get();
#pragma omp parallel for
        for (auto i = range.begin_index(); i < range.end_index(); i++) {
            op(srcs[i], tgts[i]);
        }
    }

    template <typename Op>
    void foreach_lower_node(int level, Op op)
    {
#pragma omp parallel for
        for (IndexType idx = 0; idx < ceil_num_nodes / 2; idx++) {
            // we only need to consider nodes where the level'th bit is 0
            const auto lower_part = idx & ((IndexType{1} << level) - 1);
            const auto upper_part = idx & ~((IndexType{1} << level) - 1);
            const auto i = (upper_part << 1) | lower_part;
            if (i < num_nodes) {
                op(i);
            }
        };
    }

    void reset_connected_components()
    {
        components::fill_seq_array(exec, cc_parents(),
                                   static_cast<size_type>(num_nodes));
    }

    void find_connected_components(int level)
    {
        GKO_FUNCTION_SCOPEGUARD(find_connected_components);
        reset_connected_components();
        device_disjoint_sets<IndexType> sets{cc_parents(), num_nodes};
        const auto inner_edges = get_inner_edge_range(level);
        foreach_edge_in_range(inner_edges, [&](auto src, auto tgt) {
            sets.join(sets.find_relaxed_compressing(src),
                      sets.find_relaxed_compressing(tgt));
        });
    }

    void find_min_cut_neighbors(int level)
    {
        GKO_FUNCTION_SCOPEGUARD(find_min_cut_neighbors);
        const auto min_sentinel = num_nodes;
        const auto cut_edges = get_cut_edge_range(level);
        const auto mins = cc_mins();
        components::fill_array(exec, cc_mins(),
                               static_cast<size_type>(num_nodes), min_sentinel);
        device_disjoint_sets<IndexType> sets{cc_parents(), num_nodes};
        foreach_edge_in_range(cut_edges, [sets, mins](auto src, auto tgt) {
            const auto src_rep = sets.find_relaxed_compressing(src);
            atomic_min(mins + src_rep, tgt);
        });
    }

    void add_fill_and_tree_edges(int level)
    {
        GKO_FUNCTION_SCOPEGUARD(add_fill_and_tree_edges);
        const auto cut_edges = get_cut_edge_range(level);
        device_disjoint_sets<const IndexType> sets{cc_parents(), num_nodes};
        const auto fill_srcs = fill_edge_sources();
        const auto fill_tgts = fill_edge_targets();
        const auto mins = cc_mins();
        foreach_edge_in_range_enumerated(
            cut_edges, [&](auto i, auto src, auto tgt) {
                const auto src_rep = sets.find_weak(src);
                const auto min_node = mins[src_rep];
                // we may have min_node == tgt, but that will get filtered out
                // in the sorting step
                fill_srcs[i] = min_node;
                fill_tgts[i] = tgt;
            });
        const auto out_srcs = tree_sources.get();
        const auto out_tgts = tree_targets.get();
        foreach_lower_node(level, [&](auto i) {
            if (sets.is_representative_weak(i)) {
                const auto min = mins[i];
                if (min < num_nodes) {
                    const auto out_idx = atomic_inc(tree_counter);
                    out_srcs[out_idx] = i;
                    out_tgts[out_idx] = min;
                }
            }
        });
        tree_ranges[level] = tree_counter;
    }

    void sort_new_tree_edges(int level)
    {
        const auto it =
            detail::make_zip_iterator(tree_targets.get(), tree_sources.get());
        const auto tree_range = get_tree_edge_range(level);
        std::sort(it + tree_range.begin_index(), it + tree_range.end_index());
    }

    void find_tree_connected_components(int level)
    {
        GKO_FUNCTION_SCOPEGUARD(find_tree_connected_components);
        const auto tree_edges = get_tree_edge_range(level);
        const auto srcs = tree_sources.get();
        const auto tgts = tree_targets.get();
        const auto parents = cc_parents();
        foreach_tree_edge_in_range(tree_edges, [&](auto src, auto tgt) {
            assert((src & (IndexType{1} << level)) == 0);
            assert((tgt & (IndexType{1} << level)) != 0);
            // src must be the CC representative
            assert(parents[src] == src);
            // parents should be path-compressed
            assert(parents[parents[tgt]] == parents[tgt]);
            // add the CC of src to the CC of tgt
            parents[src] = parents[tgt];
        });
        foreach_lower_node(level, [&](auto i) {
            assert((i & (IndexType{1} << level)) == 0);
            parents[i] = parents[parents[i]];
            // the path should now be fully compressed
            assert(parents[parents[i]] == parents[i]);
        });
    }

    void update_tree_node_levels(int level)
    {
        GKO_FUNCTION_SCOPEGUARD(update_tree_node_levels);
        const auto levels = tree_levels;
        const auto* parents = cc_parents();
        const auto* mins = cc_mins();
        foreach_lower_node(level, [&](IndexType i) {
            const auto min_sentinel = num_nodes;
            // for every node in a CC that gets connected to an upper node,
            // update the level using that upper node's level
            assert(parents[parents[i]] == parents[i]);
            const auto rep = parents[i];
            const auto min = mins[rep];
            if (min < min_sentinel) {
                levels[i] += levels[min] + 1;
            }
        });
    }

    void update_cc_sizes(int level)
    {
        GKO_FUNCTION_SCOPEGUARD(update_cc_sizes);
        const auto sizes = cc_sizes.get();
        const auto* parents = cc_parents();
        const auto* mins = cc_mins();
        const auto tree_edges = get_tree_edge_range(level);
        foreach_tree_edge_in_range(tree_edges, [&](auto lower, auto upper) {
            assert(parents[parents[upper]] == parents[upper]);
            auto upper_rep = parents[upper];
            atomic_add(sizes[upper_rep], sizes[lower]);
        });
    }

    void find_tree_min_cut_neighbors(int level)
    {
        GKO_FUNCTION_SCOPEGUARD(find_tree_min_cut_neighbors);
        const auto min_sentinel = num_nodes;
        components::fill_array(exec, cc_mins(),
                               static_cast<size_type>(num_nodes), min_sentinel);
        const auto tree_edges = get_tree_edge_range(level);
        const auto mins = cc_mins();
        foreach_tree_edge_in_range(tree_edges, [&](auto src, auto tgt) {
            // set mins[source] = target for every tree edge in that level
            // because we set source = i and target = mins[i] before
            mins[src] = tgt;
        });
    }

    void bucket_sort_fill_edges(int level)
    {
        GKO_FUNCTION_SCOPEGUARD(bucket_sort_fill_edges);
        const auto cut_edges = get_cut_edge_range(level);
        const auto inner_edges = get_inner_edge_range(level);
        auto sort_workspace = bucketsort_workspace_view();
        std::fill_n(sort_workspace.get_data(), sort_workspace.get_size(), 0);
        std::array<IndexType, num_buckets + 1> fill_bucket_ranges{};
        const auto in_srcs = in_edge_sources();
        const auto in_tgts = in_edge_targets();
        const auto fill_srcs = fill_edge_sources();
        const auto fill_tgts = fill_edge_targets();
        const auto out_srcs = out_edge_sources();
        const auto out_tgts = out_edge_targets();
        const auto counts = sort_workspace.get_data();
#pragma omp parallel
        {
            const auto tid = static_cast<IndexType>(omp_get_thread_num());
            const auto num_threads = omp_get_num_threads();
            const auto fill_work_per_thread =
                static_cast<IndexType>(ceildiv(cut_edges.size(), num_threads));
            const auto local_fill_begin =
                std::min(tid * fill_work_per_thread, cut_edges.size());
            const auto local_fill_end = std::min(
                local_fill_begin + fill_work_per_thread, cut_edges.size());
            const auto merge_work_per_thread = static_cast<IndexType>(
                ceildiv(inner_edges.size(), num_threads));
            const auto local_merge_begin =
                std::min(tid * merge_work_per_thread, inner_edges.size()) +
                inner_edges.begin_index();
            const auto local_merge_end =
                std::min(local_merge_begin + merge_work_per_thread,
                         inner_edges.size()) +
                inner_edges.begin_index();
            const auto local_counts = counts + tid * num_buckets;
            for (auto i : irange{local_fill_begin, local_fill_end}) {
                const auto src = fill_srcs[i];
                const auto tgt = fill_tgts[i];
                // fill edges may contain self-loops (which allow us to avoid
                // atomics)
                if (src != tgt) {
                    const auto bucket = get_edge_level(src, tgt);
                    assert(bucket >= 0);
                    assert(bucket < level);
                    local_counts[bucket]++;
                }
            }
#pragma omp barrier
#pragma omp single
            {
                std::array<IndexType, num_buckets> offsets{};
                for (int tid = 0; tid < num_threads; tid++) {
                    for (int i = 0; i < num_buckets; i++) {
                        const auto value = counts[tid * num_buckets + i];
                        counts[tid * num_buckets + i] = offsets[i];
                        offsets[i] += value;
                    }
                }
                std::copy_n(offsets.begin(), num_buckets,
                            fill_bucket_ranges.begin());
                std::exclusive_scan(fill_bucket_ranges.begin(),
                                    fill_bucket_ranges.end(),
                                    fill_bucket_ranges.begin(), IndexType{});
            }
            for (auto i : irange{local_merge_begin, local_merge_end}) {
                const auto src = in_srcs[i];
                const auto tgt = in_tgts[i];
                const auto lvl = get_edge_level(src, tgt);
                assert(i >= bucket_ranges[lvl]);
                assert(i < bucket_ranges[lvl + 1]);
                const auto rel_pos = i - bucket_ranges[lvl];
                const auto out_pos =
                    bucket_ranges[lvl] + fill_bucket_ranges[lvl] + rel_pos;
                out_srcs[out_pos] = src;
                out_tgts[out_pos] = tgt;
            }
            for (auto i : irange{local_fill_begin, local_fill_end}) {
                const auto src = fill_srcs[i];
                const auto tgt = fill_tgts[i];
                if (src != tgt) {
                    const auto bucket = get_edge_level(src, tgt);
                    assert(bucket >= 0);
                    assert(bucket < level);
                    // place fill edges after original edges
                    const auto out_pos = local_counts[bucket]++ +
                                         fill_bucket_ranges[bucket] +
                                         bucket_ranges[bucket + 1];
                    out_srcs[out_pos] = src;
                    out_tgts[out_pos] = tgt;
                }
            }
        }
        for (int i = 0; i <= num_buckets; i++) {
            bucket_ranges[i] += fill_bucket_ranges[i];
        }
        // all edges >= level were filtered out
        for (int i = level + 1; i <= num_buckets; i++) {
            bucket_ranges[i] = bucket_ranges[level];
        }
    }

    void run()
    {
        bucket_sort_input();
        // now the input is sorted by the level at which they become cut edges
        for (int level = num_levels - 1; level >= 0; level--) {
            // this level considers block of size 2^level
            // every even block is lower, every odd block is upper (0-based)
            // we consider connected components inside the blocks and
            // cut edges between an even block and its following odd block
            OperationScopeGuard guard{"level" + std::to_string(level), exec};
            // first compute connected components of all edges inside the blocks
            // these edges have at most level - 1
            find_connected_components(level);
            // then we find the smallest node from an odd block adjacent to each
            // even block's connected component with cut edges at level
            find_min_cut_neighbors(level);
            // then we add fill edges and tree edges with this information
            add_fill_and_tree_edges(level);
            // and insert the new fill edges sorted according to their level
            // additionally, we throw away all edges at level or higher
            bucket_sort_fill_edges(level);
            // and sort all new tree edges by (target, source)
            sort_new_tree_edges(level);
            // finally we swap the double buffer
            output_to_input();
        }
        // bottom-up add edges to assemble the tree
        reset_connected_components();
        for (int level = 0; level < num_levels; level++) {
            OperationScopeGuard guard{"tree_level" + std::to_string(level),
                                      exec};
            // first reconstruct the cut edge minima from the tree edges
            find_tree_min_cut_neighbors(level);
            // for every CC in this level, update using cc_min's level
            update_tree_node_levels(level);
            // before actually connecting the CCs, add the size of each lower CC
            // to its upper CC if they are getting connected
            update_cc_sizes(level);
            // build connected components with the tree edges from this level
            // to advance to the next level
            find_tree_connected_components(level);
        }
    }

    std::shared_ptr<const DefaultExecutor> exec;
    IndexType num_nodes;
    IndexType edge_capacity;
    int num_levels;
    int num_threads;
    IndexType ceil_num_nodes;
    IndexType tree_counter;
    combined_workspace<IndexType> workspace;
    double_buffer<IndexType> worklists_sources;
    double_buffer<IndexType> worklists_targets;
    double_buffer<IndexType> tree_sources;
    double_buffer<IndexType> tree_targets;
    double_buffer<IndexType> euler_walk;
    double_buffer<IndexType> euler_sizes;
    double_buffer<IndexType> euler_first;
    double_buffer<IndexType> cc_sizes;
    std::array<IndexType, num_buckets + 1> bucket_ranges;
    std::array<IndexType, num_buckets + 1> tree_ranges;
    IndexType* tree_levels;
};


template <typename IndexType>
void compute(std::shared_ptr<const DefaultExecutor> exec,
             const IndexType* row_ptrs, const IndexType* cols, size_type usize,
             gko::factorization::elimination_forest<IndexType>& forest)
{
    using unsigned_type = std::make_unsigned_t<IndexType>;
    const auto size = static_cast<IndexType>(usize);
    auto [targets, sources] =
        extract_lower_triangular(exec, row_ptrs, cols, size);
    const auto num_edges = static_cast<IndexType>(sources.get_size());
    elimination_forest_algorithm_state<IndexType> state{
        exec, size, num_edges, forest.levels.get_data()};
    state.init(sources, targets);
    state.run();
    auto num_tree_edges = state.tree_counter;
    const auto parents = forest.parents.get_data();
    const auto srcs = state.tree_sources.get();
    const auto tgts = state.tree_targets.get();
    components::fill_array(exec, parents, static_cast<size_type>(size), size);
#pragma omp parallel for
    for (IndexType i = 0; i < num_tree_edges; i++) {
        parents[srcs[i]] = tgts[i];
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
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_FROM_FACTOR);


}  // namespace elimination_forest
}  // namespace omp
}  // namespace kernels
}  // namespace gko
