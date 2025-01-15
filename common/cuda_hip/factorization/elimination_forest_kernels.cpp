// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/elimination_forest.hpp"

#include <algorithm>
#include <limits>
#include <memory>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <ginkgo/core/matrix/csr.hpp>

#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/thrust.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/intrinsics.hpp"
#include "common/cuda_hip/components/memory.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/factorization/elimination_forest_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace elimination_forest {


constexpr int default_block_size = 512;


namespace kernel {


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void mst_initialize_worklist(
    const IndexType* __restrict__ rows, const IndexType* __restrict__ cols,
    IndexType size, IndexType* __restrict__ worklist_sources,
    IndexType* __restrict__ worklist_targets,
    IndexType* __restrict__ worklist_edge_ids,
    IndexType* __restrict__ worklist_counter)
{
    using atomic_type = std::conditional_t<std::is_same_v<IndexType, int32>,
                                           int32, unsigned long long>;
    const auto i = thread::get_thread_id_flat<IndexType>();
    if (i >= size) {
        return;
    }
    const auto row = rows[i];
    const auto col = cols[i];
    if (col < row) {
        const auto out_i = static_cast<IndexType>(atomicAdd(
            reinterpret_cast<atomic_type*>(worklist_counter), atomic_type{1}));
        worklist_sources[out_i] = row;
        worklist_targets[out_i] = col;
        worklist_edge_ids[out_i] = i;
    }
}


template <typename IndexType>
__device__ IndexType mst_find(const IndexType* parents, IndexType node)
{
    auto parent = parents[node];
    while (parent != node) {
        node = parent;
        parent = parents[node];
    };
    return parent;
}


template <typename IndexType>
__device__ IndexType mst_find_relaxed(const IndexType* parents, IndexType node)
{
    auto parent = load_relaxed_local(parents + node);
    while (parent != node) {
        node = parent;
        parent = load_relaxed_local(parents + node);
    };
    return parent;
}


template <typename IndexType>
__device__ void guarded_atomic_min(IndexType* ptr, IndexType value)
{
    using atomic_type = std::conditional_t<std::is_same_v<IndexType, int32>,
                                           int32, unsigned long long>;
    // only execute the atomic if we know that it might have an effect
    if (load_relaxed_local(ptr) > value) {
        atomicMin(reinterpret_cast<atomic_type*>(ptr),
                  static_cast<atomic_type>(value));
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void mst_find_minimum(
    const IndexType* __restrict__ in_sources,
    const IndexType* __restrict__ in_targets,
    const IndexType* __restrict__ in_edge_ids, IndexType size,
    const IndexType* __restrict__ parents, IndexType* __restrict__ min_edge,
    IndexType* __restrict__ worklist_sources,
    IndexType* __restrict__ worklist_targets,
    IndexType* __restrict__ worklist_edge_ids,
    IndexType* __restrict__ worklist_counter)
{
    using atomic_type = std::conditional_t<std::is_same_v<IndexType, int32>,
                                           int32, unsigned long long>;
    const auto i = thread::get_thread_id_flat<IndexType>();
    if (i >= size) {
        return;
    }
    const auto source = in_sources[i];
    const auto target = in_targets[i];
    const auto edge_id = in_edge_ids[i];
    const auto source_rep = mst_find(parents, source);
    const auto target_rep = mst_find(parents, target);
    if (source_rep != target_rep) {
        const auto out_i = static_cast<IndexType>(atomicAdd(
            reinterpret_cast<atomic_type*>(worklist_counter), atomic_type{1}));
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
    const IndexType* __restrict__ in_edge_ids, IndexType size,
    IndexType* __restrict__ parents, const IndexType* __restrict__ min_edge,
    const IndexType* __restrict__ edge_sources,
    const IndexType* __restrict__ edge_targets,
    IndexType* __restrict__ out_sources, IndexType* __restrict__ out_targets,
    IndexType* __restrict__ out_counter)
{
    using atomic_type = std::conditional_t<std::is_same_v<IndexType, int32>,
                                           int32, unsigned long long>;
    const auto i = thread::get_thread_id_flat<IndexType>();
    if (i >= size) {
        return;
    }
    const auto source = in_sources[i];
    const auto target = in_targets[i];
    const auto edge_id = in_edge_ids[i];
    if (min_edge[source] == edge_id || min_edge[target] == edge_id) {
        // join source and sink
        const auto source_rep = mst_find_relaxed(parents, source);
        const auto target_rep = mst_find_relaxed(parents, target);
        assert(source_rep != target_rep);
        const auto new_rep = min(source_rep, target_rep);
        auto old_rep = max(source_rep, target_rep);
        bool repeat = false;
        do {
            repeat = false;
            auto old_parent =
                atomicCAS(reinterpret_cast<atomic_type*>(parents + old_rep),
                          static_cast<atomic_type>(old_rep),
                          static_cast<atomic_type>(new_rep));
            // if this fails, the parent of old_rep changed recently, so we need
            // to try again by updating the parent's parent (hopefully its rep)
            if (old_parent != old_rep) {
                old_rep = old_parent;
                repeat = true;
            }
        } while (repeat);
        const auto out_i = static_cast<IndexType>(atomicAdd(
            reinterpret_cast<atomic_type*>(out_counter), atomic_type{1}));
        out_sources[out_i] = edge_sources[edge_id];
        out_targets[out_i] = edge_targets[edge_id];
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void mst_reset_min_edges(
    const IndexType* __restrict__ in_sources,
    const IndexType* __restrict__ in_targets, IndexType size,
    IndexType* __restrict__ min_edge)
{
    constexpr auto sentinel = std::numeric_limits<IndexType>::max();
    const auto i = thread::get_thread_id_flat<IndexType>();
    if (i >= size) {
        return;
    }
    const auto source = in_sources[i];
    const auto target = in_targets[i];
    // we could write the values non-atomically, but this makes race checkers
    // happier without a performance penalty (hopefully, thanks to _local)
    store_relaxed_local(min_edge + source, sentinel);
    store_relaxed_local(min_edge + target, sentinel);
}


}  // namespace kernel


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
    const auto policy = thrust_policy(exec);
    const auto nnz = exec->copy_val_to_host(row_ptrs + size);
    // convert edges to COO representation
    // the edge list is sorted, since we only consider edges where row > col,
    // and the row array (= weights) is sorted coming from row_ptrs
    array<IndexType> row_array{exec, static_cast<size_type>(nnz)};
    const auto rows = row_array.get_data();
    components::convert_ptrs_to_idxs(exec, row_ptrs, size, rows);
    // we assume the matrix is symmetric, so we can remove every second edge
    // also round up the worklist size for equal cache alignment between fields
    const auto worklist_size =
        ceildiv(nnz, config::warp_size * 2) * config::warp_size;
    // create 2 worklists consisting of (start, end, edge_id)
    array<IndexType> worklist{exec, static_cast<size_type>(worklist_size * 6)};
    auto wl1_source = worklist.get_data();
    auto wl1_target = wl1_source + worklist_size;
    auto wl1_edge_id = wl1_target + worklist_size;
    auto wl2_source = wl1_source + 3 * worklist_size;
    auto wl2_target = wl1_target + 3 * worklist_size;
    auto wl2_edge_id = wl1_edge_id + 3 * worklist_size;
    // atomic counters for worklists and output edge list
    array<IndexType> counters{exec, 3};
    auto wl1_counter = counters.get_data();
    auto wl2_counter = wl1_counter + 1;
    auto output_counter = wl2_counter + 1;
    components::fill_array(exec, counters.get_data(), counters.get_size(),
                           IndexType{});
    // helpers for interacting with worklists
    const auto clear_wl1 = [&] {
        IndexType value{};
        exec->copy_from(exec->get_master(), 1, &value, wl1_counter);
    };
    const auto clear_wl2 = [&] {
        IndexType value{};
        exec->copy_from(exec->get_master(), 1, &value, wl2_counter);
    };
    const auto get_wl1_size = [&] {
        return exec->copy_val_to_host(wl1_counter);
    };
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
    // initialize worklist1 with forward edges
    {
        const auto num_blocks = ceildiv(nnz, default_block_size);
        kernel::mst_initialize_worklist<<<num_blocks, default_block_size>>>(
            rows, cols, nnz, wl1_source, wl1_target, wl1_edge_id, wl1_counter);
    }
    auto wl1_size = get_wl1_size();
    while (wl1_size > 0) {
        clear_wl2();
        // attach each node to its smallest adjacent non-cycle edge
        {
            const auto num_blocks = ceildiv(wl1_size, default_block_size);
            kernel::mst_find_minimum<<<num_blocks, default_block_size>>>(
                wl1_source, wl1_target, wl1_edge_id, wl1_size, parents,
                min_edges, wl2_source, wl2_target, wl2_edge_id, wl2_counter);
        }
        clear_wl1();
        swap_wl1_wl2();
        wl1_size = get_wl1_size();
        if (wl1_size > 0) {
            // join minimal edges
            const auto num_blocks = ceildiv(wl1_size, default_block_size);
            kernel::mst_join_edges<<<num_blocks, default_block_size>>>(
                wl1_source, wl1_target, wl1_edge_id, wl1_size, parents,
                min_edges, rows, cols, out_rows, out_cols, output_counter);
            kernel::mst_reset_min_edges<<<num_blocks, default_block_size>>>(
                wl1_source, wl1_target, wl1_size, min_edges);
        }
    }
    const auto num_mst_edges = exec->copy_val_to_host(output_counter);
    thrust::sort_by_key(policy, out_cols, out_cols + num_mst_edges, out_rows);
    thrust::stable_sort_by_key(policy, out_rows, out_rows + num_mst_edges,
                               out_cols);
    components::convert_idxs_to_ptrs(exec, out_rows, num_mst_edges, size,
                                     out_row_ptrs);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_SKELETON_TREE);


template <typename IndexType>
void build_children_from_parents(
    std::shared_ptr<const DefaultExecutor> exec,
    gko::factorization::elimination_forest<IndexType>& forest)
{
    const auto num_rows = forest.parents.get_size();
    // build COO representation of the tree
    array<IndexType> col_idx_array{exec, num_rows};
    const auto col_idxs = col_idx_array.get_data();
    const auto parents = forest.parents.get_const_data();
    const auto children = forest.children.get_data();
    const auto child_ptrs = forest.child_ptrs.get_data();
    exec->copy(num_rows, parents, col_idxs);
    thrust::sequence(thrust_policy(exec), children, children + num_rows,
                     IndexType{});
    // group by parent
    thrust::stable_sort_by_key(thrust_policy(exec), col_idxs,
                               col_idxs + num_rows, children);
    // create child pointers for groups of children
    components::convert_idxs_to_ptrs(exec, col_idxs, num_rows,
                                     num_rows + 1,  // rows plus sentinel
                                     child_ptrs);
}


template <typename ValueType, typename IndexType>
void from_factor(std::shared_ptr<const DefaultExecutor> exec,
                 const matrix::Csr<ValueType, IndexType>* factors,
                 gko::factorization::elimination_forest<IndexType>& forest)
{
    const auto num_rows = factors->get_size()[0];
    const auto it = thrust::make_counting_iterator(IndexType{});
    thrust::transform(
        thrust_policy(exec), it, it + num_rows, forest.parents.get_data(),
        [row_ptrs = factors->get_const_row_ptrs(),
         col_idxs = factors->get_const_col_idxs(),
         num_rows] __device__(IndexType l_col) {
            const auto llt_row_begin = row_ptrs[l_col];
            const auto llt_row_end = row_ptrs[l_col + 1];
            for (auto nz = llt_row_begin; nz < llt_row_end; nz++) {
                const auto l_row = col_idxs[nz];
                // parent[j] = min(i | i > j and l_ij =/= 0)
                // we read from L^T stored above the diagonal in factors
                // assuming a sorted order of the columns
                if (l_row > l_col) {
                    return l_row;
                }
            }
            // sentinel pseudo-root
            return static_cast<IndexType>(num_rows);
        });
    build_children_from_parents(exec, forest);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_FROM_FACTOR);


}  // namespace elimination_forest
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
