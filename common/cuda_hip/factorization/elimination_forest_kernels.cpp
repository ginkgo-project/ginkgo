// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/elimination_forest.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#ifdef GKO_COMPILING_CUDA
#include <cub/device/device_radix_sort.cuh>
#define GKO_ASSERT_NO_CUB_ERRORS(expr) GKO_ASSERT_NO_CUDA_ERRORS(expr)
#else
#include <hipcub/device/device_radix_sort.hpp>
namespace cub = hipcub;
#define GKO_ASSERT_NO_CUB_ERRORS(expr) GKO_ASSERT_NO_HIP_ERRORS(expr)
#endif
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/detail/for_each.inl>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/detail/zip_iterator.inl>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <ginkgo/core/log/profiler_hook.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/thrust.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/disjoint_sets.hpp"
#include "common/cuda_hip/components/intrinsics.hpp"
#include "common/cuda_hip/components/memory.hpp"
#include "common/cuda_hip/components/sorting.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "core/base/index_range.hpp"
#include "core/base/intrinsics.hpp"
#include "core/components/bit_packed_storage.hpp"
#include "core/components/combined_workspace.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/elimination_forest_kernels.hpp"
#include "core/log/profiler_hook.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace elimination_forest {


constexpr int default_block_size = 512;


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


namespace kernel {


template <typename IndexType>
__device__ void guarded_atomic_min(IndexType* ptr, IndexType value)
{
    // only execute the atomic if we know that it might have an effect
    if (load_relaxed_local(ptr) > value) {
        atomic_min_relaxed(ptr, value);
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void mst_find_minimum(
    const IndexType* __restrict__ in_sources,
    const IndexType* __restrict__ in_targets,
    const IndexType* __restrict__ in_edge_ids, IndexType num_edges,
    IndexType size, const IndexType* __restrict__ parents,
    IndexType* __restrict__ min_edge, IndexType* __restrict__ worklist_sources,
    IndexType* __restrict__ worklist_targets,
    IndexType* __restrict__ worklist_edge_ids,
    IndexType* __restrict__ worklist_counter)
{
    const auto i = thread::get_thread_id_flat<IndexType>();
    if (i >= num_edges) {
        return;
    }
    disjoint_sets<const IndexType> sets{parents, size};
    const auto source = in_sources[i];
    const auto target = in_targets[i];
    const auto edge_id = in_edge_ids[i];
    const auto source_rep = sets.find_weak(source);
    const auto target_rep = sets.find_weak(target);
    if (source_rep != target_rep) {
        const auto out_i = atomic_add_relaxed(worklist_counter, 1);
        worklist_sources[out_i] = source_rep;
        worklist_targets[out_i] = target_rep;
        worklist_edge_ids[out_i] = edge_id;
        guarded_atomic_min(min_edge + source_rep, edge_id);
        guarded_atomic_min(min_edge + target_rep, edge_id);
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void mst_join_edges(
    const IndexType* __restrict__ in_sources,
    const IndexType* __restrict__ in_targets,
    const IndexType* __restrict__ in_edge_ids, IndexType num_edges,
    IndexType size, IndexType* __restrict__ parents,
    const IndexType* __restrict__ min_edge,
    const IndexType* __restrict__ edge_sources,
    const IndexType* __restrict__ edge_targets,
    IndexType* __restrict__ out_sources, IndexType* __restrict__ out_targets,
    IndexType* __restrict__ out_counter)
{
    const auto i = thread::get_thread_id_flat<IndexType>();
    if (i >= num_edges) {
        return;
    }
    disjoint_sets<IndexType> sets{parents, size};
    const auto source = in_sources[i];
    const auto target = in_targets[i];
    const auto edge_id = in_edge_ids[i];
    if (min_edge[source] == edge_id || min_edge[target] == edge_id) {
        // join source and sink
        const auto source_rep = sets.find_relaxed(source);
        const auto target_rep = sets.find_relaxed(target);
        assert(source_rep != target_rep);
        sets.join(source_rep, target_rep);
        const auto out_i = atomic_add_relaxed(out_counter, 1);
        out_sources[out_i] = edge_sources[edge_id];
        out_targets[out_i] = edge_targets[edge_id];
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void mst_reset_min_edges(
    const IndexType* __restrict__ in_sources,
    const IndexType* __restrict__ in_targets, IndexType num_edges,
    IndexType* __restrict__ min_edge)
{
    constexpr auto sentinel = std::numeric_limits<IndexType>::max();
    const auto i = thread::get_thread_id_flat<IndexType>();
    if (i >= num_edges) {
        return;
    }
    const auto source = in_sources[i];
    const auto target = in_targets[i];
    // we could write the values non-atomically, but this makes race checkers
    // happier without a performance penalty (hopefully, thanks to _local)
    store_relaxed_local(min_edge + source, sentinel);
    store_relaxed_local(min_edge + target, sentinel);
}


template <int node_count, int edge_count, typename IndexType>
__global__ __launch_bounds__(default_block_size) void compute_subtree_sizes(
    const IndexType* __restrict__ child_ptrs,
    const IndexType* __restrict__ children, IndexType size,
    IndexType* __restrict__ subtree_sizes)
{
    constexpr auto sentinel = std::numeric_limits<IndexType>::max();
    const auto i = thread::get_thread_id_flat<IndexType>();
    if (i >= size) {
        return;
    }
    const auto child_begin = child_ptrs[i];
    const auto child_end = child_ptrs[i + 1];
    IndexType local_size{node_count};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    for (const auto child_idx : irange{child_begin, child_end}) {
        const auto child = children[child_idx];
        auto child_size = load_relaxed_local(subtree_sizes + child);
        while (child_size == invalid_index<IndexType>()) {
            child_size = load_relaxed(subtree_sizes + child);
        }
        local_size += child_size + edge_count;
    }
    store_relaxed(subtree_sizes + i, local_size);
#else
    if (child_begin == child_end) {
        store_relaxed(subtree_sizes + i, local_size);
    }
    auto child_idx = child_begin;
    auto child = child_idx < child_end ? children[child_idx] : IndexType{};
    while (child_idx < child_end) {
        const auto child_size = load_relaxed(subtree_sizes + child);
        if (child_size != invalid_index<IndexType>()) {
            local_size += child_size + edge_count;
            child_idx++;
            child = child_idx < child_end ? children[child_idx] : IndexType{};
            if (child_idx == child_end) {
                store_relaxed(subtree_sizes + i, local_size);
            }
        }
    }
#endif
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void compute_postorder(
    const IndexType* __restrict__ child_ptrs,
    const IndexType* __restrict__ children,
    const IndexType* __restrict__ subtree_sizes, IndexType size,
    IndexType* __restrict__ postorder, IndexType* __restrict__ inv_postorder)
{
    constexpr auto sentinel = std::numeric_limits<IndexType>::max();
    const auto rev_i = thread::get_thread_id_flat<IndexType>();
    // we include the pseudo-root here
    if (rev_i >= size + 1) {
        return;
    }
    const auto i = size - rev_i;
    const auto child_begin = child_ptrs[i];
    const auto child_end = child_ptrs[i + 1];
    const auto subtree_size = i == size ? size + 1 : subtree_sizes[i];
    IndexType postorder_idx =
        i == size ? size : load_relaxed_local(inv_postorder + i);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    while (postorder_idx == invalid_index<IndexType>()) {
        postorder_idx = load_relaxed(inv_postorder + i);
    }
    // don't output the pseudo-root
    if (i < size) {
        postorder[postorder_idx] = i;
    }
    // index of the leftmost descendant of i
    auto postorder_base = postorder_idx - subtree_size + 1;
    for (auto child_idx : irange{child_begin, child_end}) {
        const auto child = children[child_idx];
        const auto child_subtree_size = subtree_sizes[child];
        const auto child_postorder_idx =
            postorder_base + child_subtree_size - 1;
        store_relaxed(inv_postorder + child, child_postorder_idx);
        postorder_base += child_subtree_size;
    }
#else
    auto child_idx = child_begin - 1;
    auto postorder_base = invalid_index<IndexType>();
    while (child_idx < child_end) {
        if (postorder_idx != invalid_index<IndexType>()) {
            if (child_idx == child_begin - 1) {
                // don't output the pseudo-root
                if (i < size) {
                    postorder[postorder_idx] = i;
                }
                // index of the leftmost descendant of i
                postorder_base = postorder_idx - subtree_size + 1;
            } else {
                const auto child = children[child_idx];
                const auto child_subtree_size = subtree_sizes[child];
                const auto child_postorder_idx =
                    postorder_base + child_subtree_size - 1;
                store_relaxed(inv_postorder + child, child_postorder_idx);
                postorder_base += child_subtree_size;
            }
            child_idx++;
        } else {
            postorder_idx = load_relaxed(inv_postorder + i);
        }
    }
#endif
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void compute_euler_path(
    const IndexType* __restrict__ child_ptrs,
    const IndexType* __restrict__ children,
    const IndexType* __restrict__ euler_path_sizes,
    const IndexType* __restrict__ levels, IndexType size,
    IndexType* __restrict__ euler_path, IndexType* __restrict__ euler_level,
    IndexType* __restrict__ euler_first)
{
    constexpr auto sentinel = std::numeric_limits<IndexType>::max();
    const auto rev_i = thread::get_thread_id_flat<IndexType>();
    // we include the pseudo-root here
    if (rev_i >= size + 1) {
        return;
    }
    const auto i = size - rev_i;
    const auto child_begin = child_ptrs[i];
    const auto child_end = child_ptrs[i + 1];
    const auto level = i == size ? IndexType{-1} : levels[i];
    const auto euler_path_size = i == size ? 2 * size + 1 : euler_path_sizes[i];
    IndexType euler_idx =
        i == size ? IndexType{} : load_relaxed_local(euler_first + i);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    while (euler_idx == invalid_index<IndexType>()) {
        euler_idx = load_relaxed(euler_first + i);
    }
    euler_path[euler_idx] = i;
    euler_level[euler_idx] = level;
    for (auto child_idx : irange{child_begin, child_end}) {
        euler_idx++;
        const auto child = children[child_idx];
        const auto child_subtree_size = euler_path_sizes[child];
        store_relaxed(euler_first + child, euler_idx);
        euler_idx += child_subtree_size + 1;
        euler_path[euler_idx] = i;
        euler_level[euler_idx] = level;
    }
#else
    auto child_idx = child_begin - 1;
    while (child_idx < child_end) {
        if (euler_idx != invalid_index<IndexType>()) {
            if (child_idx == child_begin - 1) {
                euler_path[euler_idx] = i;
                euler_level[euler_idx] = level;
            } else {
                euler_idx++;
                const auto child = children[child_idx];
                const auto child_subtree_size = euler_path_sizes[child];
                store_relaxed(euler_first + child, euler_idx);
                euler_idx += child_subtree_size + 1;
                euler_path[euler_idx] = i;
                euler_level[euler_idx] = level;
            }
            child_idx++;
        } else {
            euler_idx = load_relaxed(euler_first + i);
        }
    }
#endif
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void compute_levels(
    const IndexType* __restrict__ parents, IndexType size,
    IndexType* __restrict__ levels)
{
    constexpr auto sentinel = std::numeric_limits<IndexType>::lowest();
    const auto rev_i = thread::get_thread_id_flat<IndexType>();
    if (rev_i >= size) {
        return;
    }
    // iterate through nodes in reverse order
    const auto i = size - rev_i - 1;
    auto current = i;
    // how many steps up did we have to go?
    IndexType delta{0};
    auto level = sentinel;
    while (level == sentinel) {
        if (current == size) {
            // the pseudo-root is at level -1
            level = -1;
        } else {
            level = load_relaxed(levels + current);
        }
        if (level != sentinel) {
            store_relaxed(levels + i, level + delta);
            return;
        }
        current = parents[current];
        delta++;
    }
}


}  // namespace kernel


template <typename IndexType>
void sort_edges(std::shared_ptr<const DefaultExecutor> exec, IndexType* sources,
                IndexType* targets, IndexType num_edges)
{
    const auto policy = thrust_policy(exec);
    // two separate sort calls get turned into efficient RadixSort invocations
    thrust::sort_by_key(policy, targets, targets + num_edges, sources);
    thrust::stable_sort_by_key(policy, sources, sources + num_edges, targets);
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
            num_edges,  // 0: input_wl_sources
            num_edges,  // 1: input_wl_targets
            num_edges,  // 2: input_wl_edge_ids
            num_edges,  // 3: output_wl_sources
            num_edges,  // 4: output_wl_targets
            num_edges,  // 5: output_wl_edge_ids
            num_nodes,  // 6: parents
            num_nodes,  // 7: min_neighbors
            3           // 8: atomic counters
        };
    }

    static size_type storage_requirement(size_type num_nodes,
                                         size_type num_edges)
    {
        return combined_workspace<IndexType>::get_total_size(
            storage_sizes(num_nodes, num_edges));
    }

    mst_state(std::shared_ptr<const DefaultExecutor> exec, size_type num_nodes,
              size_type num_edges, IndexType* input_sources,
              IndexType* input_targets, IndexType* tree_sources,
              IndexType* tree_targets)
        : exec_{exec},
          num_nodes_{num_nodes},
          num_edges_{num_edges},
          input_sources_{input_sources},
          input_targets_{input_targets},
          workspace_{exec, storage_sizes(num_nodes, num_edges)},
          tree_sources_{tree_sources},
          tree_targets_{tree_targets},
          flip_{}
    {
        std::array<IndexType, 3> zeros{};
        exec_->copy_from(exec_->get_master(), 3, zeros.data(), tree_counter());
        components::fill_array(exec_, min_edges(), num_nodes_,
                               min_edge_sentinel);
        components::fill_seq_array(exec_, parents(), num_nodes_);
        exec_->copy(num_edges_, input_sources_, output_wl_sources());
        exec_->copy(num_edges_, input_targets_, output_wl_targets());
        components::fill_seq_array(exec_, output_wl_edge_ids(), num_edges_);
        output_to_input();
    }

    const IndexType* input_wl_sources()
    {
        return workspace_.get_pointer(flip_ ? 3 : 0);
    }

    const IndexType* input_wl_targets()
    {
        return workspace_.get_pointer(flip_ ? 4 : 1);
    }

    const IndexType* input_wl_edge_ids()
    {
        return workspace_.get_pointer(flip_ ? 5 : 2);
    }

    IndexType* output_wl_sources()
    {
        return workspace_.get_pointer(flip_ ? 0 : 3);
    }

    IndexType* output_wl_targets()
    {
        return workspace_.get_pointer(flip_ ? 1 : 4);
    }

    IndexType* output_wl_edge_ids()
    {
        return workspace_.get_pointer(flip_ ? 2 : 5);
    }

    IndexType* parents() { return workspace_.get_pointer(6); }

    IndexType* min_edges() { return workspace_.get_pointer(7); }

    IndexType* tree_counter() { return workspace_.get_pointer(8); }

    const IndexType* input_counter()
    {
        return tree_counter() + (flip_ ? 1 : 2);
    }

    IndexType* output_counter() { return tree_counter() + (flip_ ? 2 : 1); }

    void output_to_input()
    {
        flip_ = !flip_;
        // reset output counter to 0
        IndexType value{};
        exec_->copy_from(exec_->get_master(), 1, &value, output_counter());
    }

    IndexType input_size() { return exec_->copy_val_to_host(input_counter()); }

    IndexType tree_size() { return exec_->copy_val_to_host(tree_counter()); }

    void run()
    {
        auto ssize = static_cast<IndexType>(num_nodes_);
        auto input_wl_size = static_cast<IndexType>(num_edges_);
        while (input_wl_size > 0) {
            const auto num_find_blocks =
                ceildiv(input_wl_size, default_block_size);
            kernel::mst_find_minimum<<<num_find_blocks, default_block_size, 0,
                                       exec_->get_stream()>>>(
                input_wl_sources(), input_wl_targets(), input_wl_edge_ids(),
                input_wl_size, ssize, parents(), min_edges(),
                output_wl_sources(), output_wl_targets(), output_wl_edge_ids(),
                output_counter());
            output_to_input();
            input_wl_size = input_size();
            if (input_wl_size > 0) {
                // join minimal edges
                const auto num_join_blocks =
                    ceildiv(input_wl_size, default_block_size);
                kernel::mst_join_edges<<<num_join_blocks, default_block_size, 0,
                                         exec_->get_stream()>>>(
                    input_wl_sources(), input_wl_targets(), input_wl_edge_ids(),
                    input_wl_size, ssize, parents(), min_edges(),
                    input_sources_, input_targets_, tree_sources_,
                    tree_targets_, tree_counter());
                if (input_wl_size < ssize / 8) {
                    // if there are only a handful of min_edges values to reset:
                    // do it individually
                    kernel::mst_reset_min_edges<<<num_join_blocks,
                                                  default_block_size, 0,
                                                  exec_->get_stream()>>>(
                        input_wl_sources(), input_wl_targets(), input_wl_size,
                        min_edges());
                } else {
                    // otherwise reset the entire array
                    components::fill_array(exec_, min_edges(), ssize,
                                           min_edge_sentinel);
                }
            }
        }
    }

    void sort_input_edges()
    {
        sort_edges(exec_, input_sources_, input_targets_,
                   static_cast<IndexType>(num_edges_));
    }

    void sort_tree_edges()
    {
        sort_edges(exec_, tree_sources_, tree_targets_, tree_size());
    }

    constexpr static auto min_edge_sentinel =
        std::numeric_limits<IndexType>::max();
    std::shared_ptr<const DefaultExecutor> exec_;
    size_type num_nodes_;
    size_type num_edges_;
    IndexType* input_sources_;
    IndexType* input_targets_;
    IndexType* tree_sources_;
    IndexType* tree_targets_;
    combined_workspace<IndexType> workspace_;
    bool flip_;
};


template <typename IndexType>
std::pair<array<IndexType>, array<IndexType>> extract_lower_triangular(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* row_ptrs,
    const IndexType* cols, size_type size)
{
    GKO_FUNCTION_SCOPEGUARD(extract_lower_triangular);
    const auto nnz = exec->copy_val_to_host(row_ptrs + size);
    array<IndexType> rows{exec, static_cast<size_type>(nnz)};
    components::convert_ptrs_to_idxs(exec, row_ptrs, size, rows.get_data());
    const auto in_it = thrust::make_zip_iterator(rows.get_const_data(), cols);
    const auto edge_predicate = [] __device__(auto edge) {
        return thrust::get<0>(edge) > thrust::get<1>(edge);
    };
    const auto num_edges = static_cast<size_type>(thrust::count_if(
        thrust_policy(exec), in_it, in_it + nnz, edge_predicate));
    array<IndexType> out_rows{exec, num_edges};
    array<IndexType> out_cols{exec, num_edges};
    const auto out_it =
        thrust::make_zip_iterator(out_rows.get_data(), out_cols.get_data());
    thrust::copy_if(thrust_policy(exec), in_it, in_it + nnz, out_it,
                    edge_predicate);
    return std::make_pair(std::move(out_rows), std::move(out_cols));
}


template <typename IndexType>
void compute_skeleton_tree(std::shared_ptr<const DefaultExecutor> exec,
                           const IndexType* row_ptrs, const IndexType* cols,
                           size_type size, IndexType* out_row_ptrs,
                           IndexType* out_cols)
{
    // we don't filter heavy edges since the heaviest edges are necessary to
    // reach the last node and we don't need to sort since the COO format
    // already sorts by row index.
    auto [sources, targets] =
        extract_lower_triangular(exec, row_ptrs, cols, size);
    const auto num_edges = sources.get_size();

    array<IndexType> out_row_array{exec, size - 1};
    mst_state<IndexType> state{
        exec,
        size,
        num_edges,
        sources.get_data(),
        targets.get_data(),
        out_row_array.get_data(),
        out_cols,
    };
    state.run();
    state.sort_tree_edges();
    const auto num_mst_edges = state.tree_size();
    components::convert_idxs_to_ptrs(exec, out_row_array.get_const_data(),
                                     num_mst_edges, size, out_row_ptrs);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_SKELETON_TREE);


namespace kernel {


template <typename IndexType>
__device__ __forceinline__ int get_edge_level(IndexType src, IndexType tgt)
{
    using unsigned_type = std::make_unsigned_t<IndexType>;
    assert(src != tgt);
    return gko::detail::find_highest_bit(static_cast<unsigned_type>(src) ^
                                         static_cast<unsigned_type>(tgt));
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void build_conn_components(
    const IndexType* __restrict__ edge_sources,
    const IndexType* __restrict__ edge_targets, size_type num_edges,
    IndexType size, IndexType* __restrict__ parents)
{
    const auto i = thread::get_thread_id_flat();
    if (i >= num_edges) {
        return;
    }
    disjoint_sets<IndexType> sets{parents, size};
    const auto src = edge_sources[i];
    const auto tgt = edge_targets[i];
    const auto src_rep = sets.find_relaxed_compressing(src);
    const auto tgt_rep = sets.find_relaxed_compressing(tgt);
    sets.join(src_rep, tgt_rep);
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void find_min_cut_neighbors(
    const IndexType* __restrict__ edge_sources,
    const IndexType* __restrict__ edge_targets, size_type num_edges,
    IndexType size, IndexType* __restrict__ parents,
    IndexType* __restrict__ mins)
{
    const auto i = thread::get_thread_id_flat();
    if (i >= num_edges) {
        return;
    }
    disjoint_sets<IndexType> sets{parents, size};
    const auto src = edge_sources[i];
    const auto tgt = edge_targets[i];
    // cut edge: count for minimum
    const auto src_rep = sets.find_relaxed_compressing(src);
    guarded_atomic_min(mins + src_rep, tgt);
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void add_fill_edges(
    const IndexType* __restrict__ edge_sources,
    const IndexType* __restrict__ edge_targets, size_type num_edges,
    IndexType size, const IndexType* __restrict__ parents,
    const IndexType* __restrict__ mins,
    IndexType* __restrict__ new_edge_sources,
    IndexType* __restrict__ new_edge_targets)
{
    const auto i = thread::get_thread_id_flat();
    if (i >= num_edges) {
        return;
    }
    disjoint_sets<const IndexType> sets{parents, size};
    const auto src = edge_sources[i];
    const auto tgt = edge_targets[i];
    const auto src_rep = sets.find_weak(src);
    const auto min_node = mins[src_rep];
    // we may have min_node == tgt, but that will get filtered out in the
    // sorting step
    new_edge_sources[i] = min_node;
    new_edge_targets[i] = tgt;
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void add_tree_edges(
    const IndexType* __restrict__ cc_parents,
    const IndexType* __restrict__ mins, IndexType size,
    IndexType* __restrict__ forest_sources,
    IndexType* __restrict__ forest_targets,
    IndexType* __restrict__ forest_counter)
{
    const auto i = thread::get_thread_id_flat<IndexType>();
    if (i >= size) {
        return;
    }
    disjoint_sets<const IndexType> sets{cc_parents, size};
    if (sets.is_representative_weak(i) && mins[i] < size) {
        auto out_i = atomic_add_relaxed(forest_counter, 1);
        forest_sources[out_i] = i;
        forest_targets[out_i] = mins[i];
    }
}


template <typename Config, typename IndexType, typename ToSortInputIterator,
          typename SortedInputIterator, typename OutputIterator,
          typename Predicate, typename BucketIndexOp>
__global__
__launch_bounds__(Config::threadblock_size) void bucket_sort_merge_filter_distribute(
    ToSortInputIterator to_sort_begin, SortedInputIterator sorted_begin,
    OutputIterator out_begin, IndexType to_sort_size, IndexType sorted_size,
    IndexType num_sort_blocks, Predicate predicate, BucketIndexOp bucket_op,
    const IndexType* to_sort_offsets, const IndexType* sorted_offsets)
{
    constexpr auto num_buckets = Config::num_buckets;
    constexpr auto threadblock_size = Config::threadblock_size;
    const auto block_id = static_cast<IndexType>(blockIdx.x);
    const auto global_to_sort_offsets =
        to_sort_offsets + num_buckets * num_sort_blocks;
    if (block_id >= num_sort_blocks) {
        const auto relative_block = block_id - num_sort_blocks;
        const auto i =
            relative_block * threadblock_size + static_cast<int>(threadIdx.x);
        if (i >= sorted_size) {
            return;
        }
        const auto value = *(sorted_begin + i);
        const auto bucket = bucket_op(value);
        const auto bucket_offset = sorted_offsets[bucket];
        const auto relative_pos = i - bucket_offset;
        assert(relative_pos >= 0);
        const auto out_pos = global_to_sort_offsets[bucket] +
                             sorted_offsets[bucket] + relative_pos;
        *(out_begin + out_pos) = value;
    } else {
        __shared__ IndexType sh_counters[num_buckets];
        for (int i = threadIdx.x; i < num_buckets; i += threadblock_size) {
            sh_counters[i] = to_sort_offsets[i + num_buckets * block_id] +
                             global_to_sort_offsets[i] + sorted_offsets[i + 1];
        }
        __syncthreads();
        const auto base_i = Config::items_per_threadblock * block_id;
        const auto end =
            min(base_i + Config::items_per_threadblock, to_sort_size);
        for (IndexType i = base_i + threadIdx.x; i < end;
             i += threadblock_size) {
            const auto value = *(to_sort_begin + i);
            if (predicate(value)) {
                const auto bucket = bucket_op(value);
                assert(bucket >= 0 && bucket < num_buckets);
                auto out_pos =
                    atomic_add_relaxed_shared(sh_counters + bucket, 1);
                *(out_begin + out_pos) = value;
            }
        }
    }
}


}  // namespace kernel


template <typename IndexType>
struct elimination_forest_algorithm_state {
    constexpr static int num_buckets = CHAR_BIT * sizeof(IndexType) - 1;
    elimination_forest_algorithm_state(
        std::shared_ptr<const DefaultExecutor> exec, IndexType num_nodes,
        IndexType num_edges)
        : exec{exec},
          num_nodes{num_nodes},
          edge_capacity{2 * num_edges},
          num_levels{
              gko::detail::find_highest_bit(
                  static_cast<std::make_unsigned_t<IndexType>>(num_nodes - 1)) +
              1},
          ceil_num_nodes{IndexType{1} << num_levels},
          workspace{exec, workspace_sizes(num_nodes, num_edges)},
          bucket_ranges{},
          tree_ranges{},
          flip{}
    {
        bucket_ranges.back() = num_edges;
    }

    static std::vector<size_type> workspace_sizes(size_type num_nodes,
                                                  size_type num_edges)
    {
        const auto edge_capacity = 2 * num_edges;
        return {
            edge_capacity,    // 0: buf1_sources
            edge_capacity,    // 1: buf1_targets
            edge_capacity,    // 2: buf2_sources
            edge_capacity,    // 3: buf2_targets
            num_nodes - 1,    // 4: tree_sources
            num_nodes - 1,    // 5: tree_targets
            num_nodes,        // 6: tree_levels
            num_nodes,        // 7: cc_parents
            num_nodes,        // 8: cc_mins
            num_nodes,        // 9: cc_sizes
            num_buckets + 1,  // 10: device_bucket_ranges
            1,                // 11: tree_counter
            bucket_sort_workspace_size<num_buckets>(
                edge_capacity)  // 12: bucketsort_workspace
        };
    }

    static size_type storage_requirement(size_type num_nodes,
                                         size_type num_edges)
    {
        return combined_workspace<IndexType>::get_total_size(
            workspace_sizes(num_nodes, num_edges));
    }

    IndexType* buf1_edge_sources() { return workspace.get_pointer(0); }

    IndexType* buf1_edge_targets() { return workspace.get_pointer(1); }

    IndexType* buf2_edge_sources() { return workspace.get_pointer(2); }

    IndexType* buf2_edge_targets() { return workspace.get_pointer(3); }

    IndexType* tree_sources() { return workspace.get_pointer(4); }

    IndexType* tree_targets() { return workspace.get_pointer(5); }

    IndexType* tree_levels() { return workspace.get_pointer(6); }

    IndexType* cc_parents() { return workspace.get_pointer(7); }

    IndexType* cc_mins() { return workspace.get_pointer(8); }

    IndexType* cc_sizes() { return workspace.get_pointer(9); }

    IndexType* device_bucket_ranges() { return workspace.get_pointer(10); }

    IndexType* tree_counter() { return workspace.get_pointer(11); }

    IndexType* bucketsort_workspace() { return workspace.get_pointer(12); }

    array<IndexType> bucketsort_workspace_view()
    {
        return workspace.get_view(12);
    }

    IndexType num_edges() const { return bucket_ranges.back(); }

    const IndexType* in_edge_sources()
    {
        return flip ? buf2_edge_sources() : buf1_edge_sources();
    }

    const IndexType* in_edge_targets()
    {
        return flip ? buf2_edge_targets() : buf1_edge_targets();
    }

    IndexType* fill_edge_sources()
    {
        return (flip ? buf2_edge_sources() : buf1_edge_sources()) + num_edges();
    }

    IndexType* fill_edge_targets()
    {
        return (flip ? buf2_edge_targets() : buf1_edge_targets()) + num_edges();
    }

    IndexType* out_edge_sources()
    {
        return flip ? buf1_edge_sources() : buf2_edge_sources();
    }
    IndexType* out_edge_targets()
    {
        return flip ? buf1_edge_targets() : buf2_edge_targets();
    }

    void output_to_input() { flip = !flip; }

    void init(const array<IndexType>& sources, const array<IndexType>& ends)
    {
        GKO_FUNCTION_SCOPEGUARD(init);
        assert(sources.get_size() == ends.get_size());
        assert(sources.get_size() <= edge_capacity);
        exec->copy(sources.get_size(), sources.get_const_data(),
                   out_edge_sources());
        exec->copy(ends.get_size(), ends.get_const_data(), out_edge_targets());
        IndexType zero{};
        exec->copy_from(exec->get_master(), 1, &zero, tree_counter());
        bucket_ranges.back() = static_cast<IndexType>(sources.get_size());
        output_to_input();
    }

    void bucket_sort_input()
    {
        GKO_FUNCTION_SCOPEGUARD(bucket_sort_input);
        auto it =
            thrust::make_zip_iterator(in_edge_sources(), in_edge_targets());
        auto out_it =
            thrust::make_zip_iterator(out_edge_sources(), out_edge_targets());
        auto ws = bucketsort_workspace_view();
        bucket_ranges = bucket_sort<num_buckets>(
            exec, it, it + num_edges(), out_it,
            [] __device__(thrust::tuple<IndexType, IndexType> edge) {
                return kernel::get_edge_level(thrust::get<0>(edge),
                                              thrust::get<1>(edge));
            },
            ws);
        exec->copy_from(exec->get_master(), num_buckets + 1,
                        bucket_ranges.data(), device_bucket_ranges());
        output_to_input();
    }

    irange<IndexType> get_inner_edge_range(int level) const
    {
        assert(level >= 0);
        assert(level < num_levels);
        const auto end = bucket_ranges[level];
        assert(end >= 0);
        assert(end < edge_capacity);
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
        assert(end < edge_capacity);
        return irange<IndexType>{begin, end};
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
        const auto inner_edges = get_inner_edge_range(level);
        if (inner_edges.size() > 0) {
            const auto num_blocks =
                ceildiv(inner_edges.size(), default_block_size);
            kernel::build_conn_components<<<num_blocks, default_block_size, 0,
                                            exec->get_stream()>>>(
                in_edge_sources(), in_edge_targets(), inner_edges.size(),
                num_nodes, cc_parents());
        }
    }

    void find_min_cut_neighbors(int level)
    {
        GKO_FUNCTION_SCOPEGUARD(find_min_cut_neighbors);
        const auto min_sentinel = num_nodes;
        components::fill_array(exec, cc_mins(),
                               static_cast<size_type>(num_nodes), min_sentinel);
        const auto cut_edge_range = get_cut_edge_range(level);
        if (cut_edge_range.size() > 0) {
            const auto num_blocks =
                ceildiv(cut_edge_range.size(), default_block_size);
            kernel::find_min_cut_neighbors<<<num_blocks, default_block_size, 0,
                                             exec->get_stream()>>>(
                in_edge_sources() + cut_edge_range.begin_index(),
                in_edge_targets() + cut_edge_range.begin_index(),
                cut_edge_range.size(), num_nodes, cc_parents(), cc_mins());
        }
    }

    void add_fill_and_tree_edges(int level)
    {
        GKO_FUNCTION_SCOPEGUARD(add_fill_and_tree_edges);
        const auto cut_edge_range = get_cut_edge_range(level);
        if (cut_edge_range.size() > 0) {
            const auto num_blocks =
                ceildiv(cut_edge_range.size(), default_block_size);
            const auto num_node_blocks = ceildiv(num_nodes, default_block_size);
            kernel::add_fill_edges<<<num_blocks, default_block_size, 0,
                                     exec->get_stream()>>>(
                in_edge_sources() + cut_edge_range.begin_index(),
                in_edge_targets() + cut_edge_range.begin_index(),
                cut_edge_range.size(), num_nodes, cc_parents(), cc_mins(),
                fill_edge_sources(), fill_edge_targets());
            kernel::add_tree_edges<<<num_node_blocks, default_block_size, 0,
                                     exec->get_stream()>>>(
                cc_parents(), cc_mins(), num_nodes, tree_sources(),
                tree_targets(), tree_counter());
            tree_ranges[level] = exec->copy_val_to_host(tree_counter());
        }
    }

    void bucket_sort_fill_edges(int level)
    {
        GKO_FUNCTION_SCOPEGUARD(bucket_sort_fill_edges);
        const auto cut_edge_range = get_cut_edge_range(level);
        const auto inner_edge_range = get_inner_edge_range(level);
        using namespace gko::kernels::GKO_DEVICE_NAMESPACE::kernel;
        using config = bucket_sort_config<num_buckets>;
        auto fill_it =
            thrust::make_zip_iterator(fill_edge_sources(), fill_edge_targets());
        auto in_it =
            thrust::make_zip_iterator(in_edge_sources(), in_edge_targets());
        auto out_it =
            thrust::make_zip_iterator(out_edge_sources(), out_edge_targets());
        std::array<IndexType, num_buckets + 1> fill_bucket_ranges{};
        // bucket sort by edge level
        auto bucket_op =
            [] __device__(thrust::tuple<IndexType, IndexType> edge) {
                return kernel::get_edge_level(thrust::get<0>(edge),
                                              thrust::get<1>(edge));
            };
        // removing loops (which allow us to avoid atomics)
        auto predicate =
            [level] __device__(thrust::tuple<IndexType, IndexType> edge) {
                const auto src = thrust::get<0>(edge);
                const auto end = thrust::get<1>(edge);
                assert(src == end || kernel::get_edge_level(src, end) < level);
                return src != end;
            };
        assert(cut_edge_range.size() <= edge_capacity);
        // compute number of fill edges added to each level
        const auto num_sort_blocks = static_cast<IndexType>(
            ceildiv(cut_edge_range.size(), config::items_per_threadblock));
        const auto sort_workspace = bucketsort_workspace();
        if (num_sort_blocks > 0) {
            bucket_sort_filter_count<config>
                <<<num_sort_blocks, config::threadblock_size, 0,
                   exec->get_stream()>>>(fill_it, cut_edge_range.size(),
                                         predicate, bucket_op,
                                         bucketsort_workspace());
        }
        bucket_sort_prefixsum<config>
            <<<1, config::num_buckets, 0, exec->get_stream()>>>(
                bucketsort_workspace(), num_sort_blocks);
        const auto global_offsets =
            bucketsort_workspace() + num_buckets * num_sort_blocks;
        components::prefix_sum_nonnegative(exec, global_offsets,
                                           num_buckets + 1);
        exec->get_master()->copy_from(exec, num_buckets + 1, global_offsets,
                                      fill_bucket_ranges.data());
        const auto num_merge_blocks = static_cast<IndexType>(
            ceildiv(inner_edge_range.size(), config::threadblock_size));
        // TODO remove, just used for debugging
        components::fill_array(exec, out_edge_sources(), edge_capacity,
                               IndexType{-1});
        components::fill_array(exec, out_edge_targets(), edge_capacity,
                               IndexType{-1});
        // END TODO
        if (num_sort_blocks + num_merge_blocks > 0) {
            kernel::bucket_sort_merge_filter_distribute<config>
                <<<num_sort_blocks + num_merge_blocks, config::threadblock_size,
                   0, exec->get_stream()>>>(
                    fill_it, in_it, out_it, cut_edge_range.size(),
                    inner_edge_range.size(), num_sort_blocks, predicate,
                    bucket_op, bucketsort_workspace(), device_bucket_ranges());
        }
        for (int i = 0; i <= num_buckets; i++) {
            bucket_ranges[i] += fill_bucket_ranges[i];
        }
        // all edges >= level were filtered out
        for (int i = level + 1; i <= num_buckets; i++) {
            bucket_ranges[i] = bucket_ranges[level];
        }
        exec->copy_from(exec->get_master(), num_buckets + 1,
                        bucket_ranges.data(), device_bucket_ranges());
    }

    // sort the edges by src index, which also groups them by block
    // they will lose their grouping by level
    void radix_sort_edges()
    {
        GKO_FUNCTION_SCOPEGUARD(radix_sort_edges);
        std::size_t tmp_storage_size{};
        cub::DoubleBuffer<IndexType> sources{buf1_edge_sources(),
                                             buf2_edge_sources()};
        cub::DoubleBuffer<IndexType> targets{buf1_edge_targets(),
                                             buf2_edge_targets()};
        sources.selector = flip ? 1 : 0;
        targets.selector = flip ? 1 : 0;
        GKO_ASSERT_NO_CUB_ERRORS(cub::DeviceRadixSort::SortPairs(
            nullptr, tmp_storage_size, sources, targets, num_edges(), 0,
            num_levels, exec->get_stream()));
        array<char> buffer{exec, tmp_storage_size};
        // sort by column first
        GKO_ASSERT_NO_CUB_ERRORS(cub::DeviceRadixSort::SortPairs(
            buffer.get_data(), tmp_storage_size, sources, targets, num_edges(),
            0, num_levels, exec->get_stream()));
        // then by row, keeping the relative col order
        GKO_ASSERT_NO_CUB_ERRORS(cub::DeviceRadixSort::SortPairs(
            buffer.get_data(), tmp_storage_size, targets, sources, num_edges(),
            0, num_levels, exec->get_stream()));
        flip = (sources.selector == 1);
        assert(targets.selector == sources.selector);
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
            // finally we swap the double buffer
            output_to_input();
        }
    }

    std::shared_ptr<const DefaultExecutor> exec;
    IndexType num_nodes;
    IndexType edge_capacity;
    int num_levels;
    IndexType ceil_num_nodes;
    combined_workspace<IndexType> workspace;
    std::array<IndexType, num_buckets + 1> bucket_ranges;
    std::array<IndexType, num_buckets + 1> tree_ranges;
    bool flip;
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
    elimination_forest_algorithm_state<IndexType> state{exec, size, num_edges};
    state.init(sources, targets);
    state.run();
    auto num_tree_edges = exec->copy_val_to_host(state.tree_counter());
    thrust::fill_n(thrust_policy(exec), forest.parents.get_data(), size, size);
    thrust::scatter(thrust_policy(exec), state.tree_targets(),
                    state.tree_targets() + num_tree_edges, state.tree_sources(),
                    forest.parents.get_data());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ELIMINATION_FOREST_COMPUTE);


template <typename IndexType>
void compute_children(std::shared_ptr<const DefaultExecutor> exec,
                      const IndexType* parents, IndexType size,
                      IndexType* child_ptrs, IndexType* children)
{
    const auto usize = static_cast<size_type>(size);
    // build COO representation of the tree
    array<IndexType> col_idx_array{exec, usize};
    const auto col_idxs = col_idx_array.get_data();
    exec->copy(usize, parents, col_idxs);
    components::fill_seq_array(exec, children, usize);
    // group by parent
    thrust::stable_sort_by_key(thrust_policy(exec), col_idxs, col_idxs + size,
                               children);
    // create child pointers for groups of children
    components::convert_idxs_to_ptrs(exec, col_idxs, usize,
                                     usize + 1,  // rows plus sentinel
                                     child_ptrs);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_CHILDREN);


template <typename ValueType, typename IndexType>
void from_factor(std::shared_ptr<const DefaultExecutor> exec,
                 const matrix::Csr<ValueType, IndexType>* factors,
                 IndexType* parents)
{
    const auto num_rows = factors->get_size()[0];
    const auto it = thrust::make_counting_iterator(IndexType{});
    thrust::transform(thrust_policy(exec), it, it + num_rows, parents,
                      [row_ptrs = factors->get_const_row_ptrs(),
                       col_idxs = factors->get_const_col_idxs(),
                       num_rows] __device__(IndexType l_col) {
                          const auto llt_row_begin = row_ptrs[l_col];
                          const auto llt_row_end = row_ptrs[l_col + 1];
                          for (auto nz = llt_row_begin; nz < llt_row_end;
                               nz++) {
                              const auto l_row = col_idxs[nz];
                              // parent[j] = min(i | i > j and l_ij =/= 0)
                              // we read from L^T stored above the diagonal in
                              // factors assuming a sorted order of the columns
                              if (l_row > l_col) {
                                  return l_row;
                              }
                          }
                          // sentinel pseudo-root
                          return static_cast<IndexType>(num_rows);
                      });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_FROM_FACTOR);


}  // namespace elimination_forest
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
