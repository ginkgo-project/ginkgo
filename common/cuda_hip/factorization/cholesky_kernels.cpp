// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/cholesky_kernels.hpp"

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
#include "common/cuda_hip/base/sparselib_bindings.hpp"
#include "common/cuda_hip/base/thrust.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/intrinsics.hpp"
#include "common/cuda_hip/components/memory.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "common/cuda_hip/components/syncfree.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/lu_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Cholesky namespace.
 *
 * @ingroup factor
 */
namespace cholesky {


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


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void build_postorder_cols(
    IndexType num_rows, const IndexType* cols, const IndexType* row_ptrs,
    const IndexType* inv_postorder, IndexType* postorder_cols,
    IndexType* lower_ends)
{
    const auto row = thread::get_thread_id_flat<IndexType>();
    if (row >= num_rows) {
        return;
    }
    const auto row_begin = row_ptrs[row];
    const auto row_end = row_ptrs[row + 1];
    auto lower_end = row_begin;
    for (auto nz = row_begin; nz < row_end; nz++) {
        const auto col = cols[nz];
        if (col < row) {
            postorder_cols[lower_end] = inv_postorder[cols[nz]];
            lower_end++;
        }
    }
    // fill the rest with sentinels
    for (auto nz = lower_end; nz < row_end; nz++) {
        postorder_cols[nz] = num_rows - 1;
    }
    lower_ends[row] = lower_end;
}


template <int subwarp_size, typename IndexType>
__global__ __launch_bounds__(default_block_size) void symbolic_count(
    IndexType num_rows, const IndexType* row_ptrs, const IndexType* lower_ends,
    const IndexType* inv_postorder, const IndexType* postorder_cols,
    const IndexType* postorder_parent, IndexType* row_nnz)
{
    const auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    if (row >= num_rows) {
        return;
    }
    const auto row_begin = row_ptrs[row];
    // instead of relying on the input containing a diagonal, we artificially
    // introduce the diagonal entry (in postorder indexing) as a sentinel after
    // the last lower triangular entry.
    const auto diag_postorder = inv_postorder[row];
    const auto lower_end = lower_ends[row];
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto lane = subwarp.thread_rank();
    IndexType count{};
    for (auto nz = row_begin + lane; nz < lower_end; nz += subwarp_size) {
        auto node = postorder_cols[nz];
        const auto next_node =
            nz < lower_end - 1 ? postorder_cols[nz + 1] : diag_postorder;
        while (node < next_node) {
            count++;
            node = postorder_parent[node];
        }
    }
    // lower entries plus diagonal
    count = reduce(subwarp, count, thrust::plus<IndexType>{}) + 1;
    if (lane == 0) {
        row_nnz[row] = count;
    }
}


template <int subwarp_size, typename IndexType>
__global__ __launch_bounds__(default_block_size) void symbolic_factorize(
    IndexType num_rows, const IndexType* row_ptrs, const IndexType* lower_ends,
    const IndexType* inv_postorder, const IndexType* postorder_cols,
    const IndexType* postorder, const IndexType* postorder_parent,
    const IndexType* out_row_ptrs, IndexType* out_cols)
{
    const auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    if (row >= num_rows) {
        return;
    }
    const auto row_begin = row_ptrs[row];
    // instead of relying on the input containing a diagonal, we artificially
    // introduce the diagonal entry (in postorder indexing) as a sentinel after
    // the last lower triangular entry.
    const auto diag_postorder = inv_postorder[row];
    const auto lower_end = lower_ends[row];
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto lane = subwarp.thread_rank();
    const auto prefix_mask = (config::lane_mask_type(1) << lane) - 1;
    auto out_base = out_row_ptrs[row];
    for (auto base = row_begin; base < lower_end; base += subwarp_size) {
        auto nz = base + lane;
        auto node = nz < lower_end ? postorder_cols[nz] : diag_postorder;
        const auto next_node =
            nz < lower_end - 1 ? postorder_cols[nz + 1] : diag_postorder;
        bool pred = node < next_node;
        auto mask = subwarp.ballot(pred);
        while (mask) {
            if (pred) {
                const auto out_nz = out_base + popcnt(mask & prefix_mask);
                out_cols[out_nz] = postorder[node];
                node = postorder_parent[node];
                pred = node < next_node;
            }
            out_base += popcnt(mask);
            mask = subwarp.ballot(pred);
        }
    }
    // add diagonal entry
    if (lane == 0) {
        out_cols[out_base] = row;
    }
}


template <bool full_fillin, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void factorize(
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ cols,
    const IndexType* __restrict__ storage_offsets,
    const int32* __restrict__ storage, const int64* __restrict__ row_descs,
    const IndexType* __restrict__ diag_idxs,
    const IndexType* __restrict__ transpose_idxs, ValueType* __restrict__ vals,
    syncfree_storage dep_storage, size_type num_rows)
{
    using scheduler_t =
        syncfree_scheduler<default_block_size, config::warp_size, IndexType>;
    __shared__ typename scheduler_t::shared_storage sh_dep_storage;
    scheduler_t scheduler(dep_storage, sh_dep_storage);
    const auto row = scheduler.get_work_id();
    if (row >= num_rows) {
        return;
    }
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto lane = warp.thread_rank();
    const auto row_begin = row_ptrs[row];
    const auto row_diag = diag_idxs[row];
    const auto row_end = row_ptrs[row + 1];
    gko::matrix::csr::device_sparsity_lookup<IndexType> lookup{
        row_ptrs, cols,      storage_offsets,
        storage,  row_descs, static_cast<size_type>(row)};
    // for each lower triangular entry: eliminate with corresponding column
    for (auto lower_nz = row_begin; lower_nz < row_diag; lower_nz++) {
        const auto dep = cols[lower_nz];
        scheduler.wait(dep);
        const auto scale = vals[lower_nz];
        const auto diag_idx = diag_idxs[dep];
        const auto dep_end = row_ptrs[dep + 1];
        // subtract column dep from current column
        for (auto upper_nz = diag_idx + lane; upper_nz < dep_end;
             upper_nz += config::warp_size) {
            const auto upper_col = cols[upper_nz];
            if (upper_col >= row) {
                const auto upper_val = vals[upper_nz];
                if constexpr (full_fillin) {
                    const auto output_pos =
                        lookup.lookup_unsafe(upper_col) + row_begin;
                    vals[output_pos] -= scale * upper_val;
                } else {
                    const auto pos = lookup[upper_col];
                    if (pos != invalid_index<IndexType>()) {
                        vals[row_begin + pos] -= scale * upper_val;
                    }
                }
            }
        }
    }
    auto diag_val = sqrt(vals[row_diag]);
    for (auto upper_nz = row_diag + 1 + lane; upper_nz < row_end;
         upper_nz += config::warp_size) {
        vals[upper_nz] /= diag_val;
        // copy the upper triangular entries to the transpose
        vals[transpose_idxs[upper_nz]] = conj(vals[upper_nz]);
    }
    if (lane == 0) {
        // store computed diagonal
        vals[row_diag] = diag_val;
    }
    scheduler.mark_ready();
}


}  // namespace kernel


template <typename IndexType>
void compute_skeleton_tree(std::shared_ptr<const DefaultExecutor> exec,
                           const IndexType* row_ptrs, const IndexType* cols,
                           size_type size, IndexType* out_row_ptrs,
                           IndexType* out_cols)
{
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
    array<IndexType> worklist{exec, static_cast<size_type>(worklist_size * 8)};
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

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CHOLESKY_COMPUTE_SKELETON_TREE);


template <typename ValueType, typename IndexType>
void symbolic_factorize(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* mtx,
    const factorization::elimination_forest<IndexType>& forest,
    matrix::Csr<ValueType, IndexType>* l_factor,
    const array<IndexType>& tmp_storage)
{
    const auto num_rows = static_cast<IndexType>(mtx->get_size()[0]);
    if (num_rows == 0) {
        return;
    }
    const auto mtx_nnz = static_cast<IndexType>(mtx->get_num_stored_elements());
    const auto postorder_cols = tmp_storage.get_const_data();
    const auto lower_ends = postorder_cols + mtx_nnz;
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto postorder = forest.postorder.get_const_data();
    const auto inv_postorder = forest.inv_postorder.get_const_data();
    const auto postorder_parent = forest.postorder_parents.get_const_data();
    const auto out_row_ptrs = l_factor->get_const_row_ptrs();
    const auto out_cols = l_factor->get_col_idxs();
    const auto num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    kernel::symbolic_factorize<config::warp_size>
        <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
            num_rows, row_ptrs, lower_ends, inv_postorder, postorder_cols,
            postorder, postorder_parent, out_row_ptrs, out_cols);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CHOLESKY_SYMBOLIC_FACTORIZE);


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
void forest_from_factor(
    std::shared_ptr<const DefaultExecutor> exec,
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
    GKO_DECLARE_CHOLESKY_FOREST_FROM_FACTOR);


template <typename ValueType, typename IndexType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Csr<ValueType, IndexType>* mtx,
                const IndexType* factor_lookup_offsets,
                const int64* factor_lookup_descs,
                const int32* factor_lookup_storage, IndexType* diag_idxs,
                IndexType* transpose_idxs,
                matrix::Csr<ValueType, IndexType>* factors)
{
    lu_factorization::initialize(exec, mtx, factor_lookup_offsets,
                                 factor_lookup_descs, factor_lookup_storage,
                                 diag_idxs, factors);
    // convert to COO
    const auto nnz = factors->get_num_stored_elements();
    array<IndexType> row_idx_array{exec, nnz};
    array<IndexType> col_idx_array{exec, nnz};
    const auto row_idxs = row_idx_array.get_data();
    const auto col_idxs = col_idx_array.get_data();
    exec->copy(nnz, factors->get_const_col_idxs(), col_idxs);
    components::convert_ptrs_to_idxs(exec, factors->get_const_row_ptrs(),
                                     factors->get_size()[0], row_idxs);
    components::fill_seq_array(exec, transpose_idxs, nnz);
    // compute nonzero permutation for sparse transpose
    // to profit from cub/rocPRIM's fast radix sort, we do it in two steps
    thrust::stable_sort_by_key(thrust_policy(exec), row_idxs, row_idxs + nnz,
                               transpose_idxs);
    thrust::stable_sort_by_key(thrust_policy(exec), col_idxs, col_idxs + nnz,
                               transpose_idxs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CHOLESKY_INITIALIZE);


template <typename ValueType, typename IndexType>
void factorize(std::shared_ptr<const DefaultExecutor> exec,
               const IndexType* lookup_offsets, const int64* lookup_descs,
               const int32* lookup_storage, const IndexType* diag_idxs,
               const IndexType* transpose_idxs,
               const factorization::elimination_forest<IndexType>& forest,
               matrix::Csr<ValueType, IndexType>* factors, bool full_fillin,
               array<int>& tmp_storage)
{
    const auto num_rows = factors->get_size()[0];
    if (num_rows > 0) {
        syncfree_storage storage(exec, tmp_storage, num_rows);
        const auto num_blocks =
            ceildiv(num_rows, default_block_size / config::warp_size);
        if (!full_fillin) {
            kernel::factorize<false>
                <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
                    factors->get_const_row_ptrs(),
                    factors->get_const_col_idxs(), lookup_offsets,
                    lookup_storage, lookup_descs, diag_idxs, transpose_idxs,
                    as_device_type(factors->get_values()), storage, num_rows);
        } else {
            kernel::factorize<true>
                <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
                    factors->get_const_row_ptrs(),
                    factors->get_const_col_idxs(), lookup_offsets,
                    lookup_storage, lookup_descs, diag_idxs, transpose_idxs,
                    as_device_type(factors->get_values()), storage, num_rows);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CHOLESKY_FACTORIZE);


template <typename ValueType, typename IndexType>
void symbolic_count(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* mtx,
                    const factorization::elimination_forest<IndexType>& forest,
                    IndexType* row_nnz, array<IndexType>& tmp_storage)
{
    const auto num_rows = static_cast<IndexType>(mtx->get_size()[0]);
    if (num_rows == 0) {
        return;
    }
    const auto mtx_nnz = static_cast<IndexType>(mtx->get_num_stored_elements());
    tmp_storage.resize_and_reset(mtx_nnz + num_rows);
    const auto postorder_cols = tmp_storage.get_data();
    const auto lower_ends = postorder_cols + mtx_nnz;
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto cols = mtx->get_const_col_idxs();
    const auto inv_postorder = forest.inv_postorder.get_const_data();
    const auto postorder_parent = forest.postorder_parents.get_const_data();
    // transform col indices to postorder indices
    {
        const auto num_blocks = ceildiv(num_rows, default_block_size);
        kernel::build_postorder_cols<<<num_blocks, default_block_size, 0,
                                       exec->get_stream()>>>(
            num_rows, cols, row_ptrs, inv_postorder, postorder_cols,
            lower_ends);
    }
    // sort postorder_cols inside rows
    {
        const auto handle = exec->get_sparselib_handle();
        auto descr = sparselib::create_mat_descr();
        array<IndexType> permutation_array(exec, mtx_nnz);
        auto permutation = permutation_array.get_data();
        components::fill_seq_array(exec, permutation, mtx_nnz);
        size_type buffer_size{};
        sparselib::csrsort_buffer_size(handle, num_rows, num_rows, mtx_nnz,
                                       row_ptrs, postorder_cols, buffer_size);
        array<char> buffer_array{exec, buffer_size};
        auto buffer = buffer_array.get_data();
        sparselib::csrsort(handle, num_rows, num_rows, mtx_nnz, descr, row_ptrs,
                           postorder_cols, permutation, buffer);
        sparselib::destroy(descr);
    }
    // count nonzeros per row of L
    {
        const auto num_blocks =
            ceildiv(num_rows, default_block_size / config::warp_size);
        kernel::symbolic_count<config::warp_size>
            <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
                num_rows, row_ptrs, lower_ends, inv_postorder, postorder_cols,
                postorder_parent, row_nnz);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CHOLESKY_SYMBOLIC_COUNT);


}  // namespace cholesky
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
