// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/elimination_forest.hpp"

#include <algorithm>
#include <limits>
#include <memory>

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

#include <ginkgo/core/matrix/csr.hpp>

#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/thrust.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/disjoint_sets.hpp"
#include "common/cuda_hip/components/intrinsics.hpp"
#include "common/cuda_hip/components/memory.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "core/base/index_range.hpp"
#include "core/base/intrinsics.hpp"
#include "core/components/bit_packed_storage.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/elimination_forest_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace elimination_forest {


constexpr int default_block_size = 512;


namespace kernel {


template <int size>
struct small_disjoint_sets {
    bit_packed_array<ceil_log2_constexpr(size), size, uint64> reps{};

    constexpr small_disjoint_sets()
    {
        for (int i = 0; i < size; i++) {
            reps.set_from_zero(i, i);
        }
    }

    // path-compressing find
    constexpr int find(int node)
    {
        int cur = node;
        int rep = reps.get(cur);
        // first find root
        while (cur != rep) {
            cur = rep;
            rep = reps.get(cur);
        }
        // then path-compress
        cur = node;
        while (cur != rep) {
            int new_rep = reps.get(cur);
            reps.set(cur, rep);
            rep = new_rep;
        }
        return rep;
    }

    constexpr int join_reps(int a_rep, int b_rep)
    {
        assert(reps.get(a_rep) == a_rep);
        assert(reps.get(b_rep) == b_rep);
        // always make the largest value the root
        auto new_root = max(a_rep, b_rep);
        auto new_child = min(a_rep, b_rep);
        reps.set(new_child, new_root);
        return new_root;
    }
};


template <int size, typename EdgeAccessor>
__device__ bit_packed_array<ceil_log2_constexpr(size), size>
base_case_elimination_forest(int num_edges, EdgeAccessor edges)
{
    bit_packed_array<ceil_log2_constexpr(size), size, uint64> forest_parents{};
    small_disjoint_sets<size> disjoint_sets{};
    for (int i = 0; i < size; i++) {
        forest_parents.set_from_zero(i, i);
    }
    // assuming edges are sorted by row indices
    int current_row = 0;
    for (int i = 0; i < num_edges; i++) {
        const auto [col, row] = edges(i);
        assert(row >= current_row);
        assert(col < row);
        if (row > current_row) {
            current_row = row;
        }
        auto col_rep = disjoint_sets.find(col);
        if (forest_parents.get(col_rep) == col_rep && col_rep != row) {
            assert(disjoint_sets.find(row) == row);
            disjoint_sets.join_reps(col_rep, row);
            forest_parents.set(col_rep, row);
        }
    }
    return forest_parents;
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void mst_extract_edges_count(
    const IndexType* __restrict__ row_ptrs, const IndexType* cols,
    IndexType num_rows, IndexType* lower_nnz)
{
    const auto row = thread::get_thread_id_flat<IndexType>();
    if (row >= num_rows) {
        return;
    }
    IndexType counter{};
    for (auto nz : irange{row_ptrs[row], row_ptrs[row + 1]}) {
        const auto col = cols[nz];
        counter += col < row ? 1 : 0;
    }
    lower_nnz[row] = counter;
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void mst_extract_edges(
    const IndexType* __restrict__ row_ptrs, const IndexType* cols,
    IndexType num_rows, const IndexType* out_ptrs, IndexType* out_source,
    IndexType* out_target)
{
    const auto row = thread::get_thread_id_flat<IndexType>();
    if (row >= num_rows) {
        return;
    }
    auto out_idx = out_ptrs[row];
    for (auto nz : irange{row_ptrs[row], row_ptrs[row + 1]}) {
        const auto col = cols[nz];
        if (col < row) {
            out_source[out_idx] = row;
            out_target[out_idx] = col;
            out_idx++;
        }
    }
}


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


template <typename IndexType>
struct mst_state {
    static size_type storage_requirement(IndexType num_nodes,
                                         IndexType num_edges)
    {
        return 6 * static_cast<size_type>(num_edges) +
               2 * static_cast<size_type>(num_nodes) + 3;
    }

    mst_state(std::shared_ptr<const DefaultExecutor> exec, IndexType num_nodes,
              IndexType num_edges, IndexType* input_sources,
              IndexType* input_targets, IndexType* tree_sources,
              IndexType* tree_targets)
        : exec_{exec},
          num_nodes_{num_nodes},
          num_edges_{num_edges},
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

    IndexType* tree_counter() { return min_edges() + num_nodes_; }

    const IndexType* input_counter()
    {
        return tree_counter() + (flip_ ? 1 : 2);
    }

    IndexType* output_counter() { return tree_counter() + (flip_ ? 2 : 1); }

    void output_to_input()
    {
        flip_ = !flip_;
        IndexType value{};
        exec_->copy_from(exec_->get_master(), 1, &value, output_counter());
    }

    IndexType input_size() { return exec_->copy_val_to_host(input_counter()); }

    IndexType tree_size() { return exec_->copy_val_to_host(tree_counter()); }

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
    }

    void run()
    {
        std::array<IndexType, 3> zeros{};
        exec_->copy_from(exec_->get_master(), 3, zeros.data(), tree_counter());
        components::fill_array(exec_, min_edges(),
                               static_cast<size_type>(num_nodes_),
                               min_node_sentinel);
        components::fill_seq_array(exec_, parents(),
                                   static_cast<size_type>(num_nodes_));
        exec_->copy(num_edges_, input_sources_, output_wl_sources());
        exec_->copy(num_edges_, input_targets_, output_wl_targets());
        components::fill_seq_array(exec_, output_wl_edge_ids(),
                                   static_cast<size_type>(num_edges_));
        output_to_input();
        auto input_wl_size = num_edges_;
        while (input_wl_size > 0) {
            const auto num_find_blocks =
                ceildiv(input_wl_size, default_block_size);
            kernel::mst_find_minimum<<<num_find_blocks, default_block_size, 0,
                                       exec_->get_stream()>>>(
                input_wl_sources(), input_wl_targets(), input_wl_edge_ids(),
                input_wl_size, num_nodes_, parents(), min_edges(),
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
                    input_wl_size, num_nodes_, parents(), min_edges(),
                    input_sources_, input_targets_, tree_sources_,
                    tree_targets_, tree_counter());
                if (input_wl_size < num_nodes_ / 8) {
                    // if there are only a handful of min_edges values to reset:
                    // do it individually
                    kernel::mst_reset_min_edges<<<num_join_blocks,
                                                  default_block_size, 0,
                                                  exec_->get_stream()>>>(
                        input_wl_sources(), input_wl_targets(), input_wl_size,
                        min_edges());
                } else {
                    // otherwise reset the entire array
                    components::fill_array(exec_, min_edges(),
                                           static_cast<size_type>(num_nodes_),
                                           min_node_sentinel);
                }
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

    constexpr static auto min_node_sentinel =
        std::numeric_limits<IndexType>::max();
    std::shared_ptr<const DefaultExecutor> exec_;
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
    array<IndexType> sources{exec};
    array<IndexType> targets{exec};
    if (size > 0) {
        const auto ssize = static_cast<IndexType>(size);
        array<IndexType> out_ptrs{exec, size + 1};
        const auto num_blocks = ceildiv(ssize, default_block_size);
        kernel::mst_extract_edges_count<<<num_blocks, default_block_size, 0,
                                          exec->get_stream()>>>(
            row_ptrs, cols, ssize, out_ptrs.get_data());
        components::prefix_sum_nonnegative(exec, out_ptrs.get_data(), size + 1);
        const auto num_edges =
            exec->copy_val_to_host(out_ptrs.get_const_data() + ssize);
        sources.resize_and_reset(num_edges);
        targets.resize_and_reset(num_edges);
        kernel::mst_extract_edges<<<num_blocks, default_block_size, 0,
                                    exec->get_stream()>>>(
            row_ptrs, cols, ssize, out_ptrs.get_const_data(),
            sources.get_data(), targets.get_data());
    }
    return std::make_pair(std::move(sources), std::move(targets));
}


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
    auto [sources, targets] =
        extract_lower_triangular(exec, row_ptrs, cols, size);
    const auto ssize = static_cast<IndexType>(size);
    const auto num_edges = static_cast<IndexType>(sources.get_size());

    array<IndexType> out_row_array{exec, size - 1};
    mst_state<IndexType> state{
        exec,
        ssize,
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
__device__ __forceinline__ int get_edge_level(IndexType start, IndexType end)
{
    using unsigned_type = std::make_unsigned_t<IndexType>;
    assert(start != end);
    return gko::detail::find_lowest_bit(static_cast<unsigned_type>(start) ^
                                        static_cast<unsigned_type>(end));
}


template <typename IndexType>
struct elimination_forest_kernel_config {
    constexpr static int work_per_thread = 16;
    constexpr static int blocksize = 512;
    constexpr static int work_per_threadblock = work_per_thread * blocksize;
    constexpr static int max_levels = sizeof(IndexType) * CHAR_BIT;
    constexpr static int prefixsum_blocksize = max_levels;
};


/*
template <typename Config, typename IndexType>
__global__ __launch_bounds__(Config::blocksize) void count_edge_buckets(
    const IndexType* __restrict__ edge_starts,
    const IndexType* __restrict__ edge_ends, size_type num_edges,
    IndexType* __restrict__ block_counts)
{
    const auto threadblock = group::this_thread_block();
    __shared__ int local_count[Config::max_levels];
    for (int i = threadIdx.x; i < Config::max_levels; i += Config::blocksize) {
        local_count[i] = 0;
    }
    __syncthreads();
    const auto base_i = Config::work_per_threadblock * blockIdx.x;
    for (int local_i = threadIdx.x; local_i < Config::work_per_threadblock;
         local_i += Config::blocksize) {
        const auto i = local_i + base_i;
        const auto start = edge_starts[i];
        const auto end = edge_ends[i];
        if (start < end) {
            const auto level = get_edge_level(start, end);
            atomicAdd(local_count + level, 1);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < Config::max_levels; i += Config::blocksize) {
        block_counts[i + Config::max_levels * blockIdx.x] = local_count[i];
    }
}


template <typename Config, typename IndexType>
__global__
__launch_bounds__(Config::prefixsum_blocksize) void bucket_count_prefixsum(
    IndexType* __restrict__ block_counts, IndexType num_blocks)
{
    if (threadIdx.x > Config::max_levels) {
        return;
    }
    const auto bucket = static_cast<IndexType>(threadIdx.x);
    IndexType sum{};
    for (const auto block : irange{num_blocks}) {
        const auto idx = bucket + Config::max_levels * block;
        const auto count = block_counts[idx];
        block_counts[idx] = sum;
        sum += count;
    }
    block_counts[bucket + Config::max_levels * num_blocks] = sum;
}


template <typename Config, typename IndexType>
__global__ __launch_bounds__(Config::blocksize) void distribute_edge_buckets(
    const IndexType* __restrict__ edge_starts,
    const IndexType* __restrict__ edge_ends, size_type num_edges,
    const IndexType* __restrict__ block_count_prefixsum,
    IndexType* __restrict__ out_edge_starts,
    IndexType* __restrict__ out_edge_ends)
{
    const auto threadblock = group::this_thread_block();
    __shared__ int local_count[Config::max_levels];
    for (int i = threadIdx.x; i < Config::max_levels; i += Config::blocksize) {
        local_count[i] =
            block_count_prefixsum[i + Config::max_levels * blockIdx.x];
    }
    __syncthreads();
    const auto base_i = Config::work_per_threadblock * blockIdx.x;
    for (int local_i = threadIdx.x; local_i < Config::work_per_threadblock;
         local_i += Config::blocksize) {
        const auto i = local_i + base_i;
        const auto start = edge_starts[i];
        const auto end = edge_ends[i];
        if (start < end) {
            const auto level = get_edge_level(start, end);
            const auto out_i = atomicAdd(local_count + level, 1);
            out_edge_starts[out_i] = start;
            out_edge_ends[out_i] = end;
        }
    }
}*/
template <typename Config, typename IndexType>
__global__ __launch_bounds__(Config::blocksize) void count_edge_buckets(
    const IndexType* __restrict__ edge_starts,
    const IndexType* __restrict__ edge_ends, size_type num_edges,
    IndexType* __restrict__ global_counts)
{
    const auto i = thread::get_thread_id_flat();
    if (i >= num_edges) {
        return;
    }
    const auto start = edge_starts[i];
    const auto end = edge_ends[i];
    assert(start < end);
    const auto level = get_edge_level(start, end);
    atomic_add(global_counts + level, 1);
}


template <typename Config, typename IndexType>
__global__ __launch_bounds__(Config::blocksize) void distribute_edge_buckets(
    const IndexType* __restrict__ edge_starts,
    const IndexType* __restrict__ edge_ends, size_type num_edges,
    const IndexType* __restrict__ global_offsets,
    IndexType* __restrict__ out_edge_starts,
    IndexType* __restrict__ out_edge_ends)
{
    const auto i = thread::get_thread_id_flat();
    if (i >= num_edges) {
        return;
    }
    const auto start = edge_starts[i];
    const auto end = edge_ends[i];
    assert(start < end);
    const auto level = get_edge_level(start, end);
    auto output_pos = atomic_add(global_offsets + level, 1);
    out_edge_starts[i] = start;
    out_edge_ends[i] = end;
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void build_conn_components(
    const IndexType* __restrict__ edge_starts,
    const IndexType* __restrict__ edge_ends, size_type num_edges,
    IndexType size, IndexType block_size, IndexType* __restrict__ parents)
{
    const auto i = thread::get_thread_id_flat();
    if (i >= num_edges) {
        return;
    }
    disjoint_sets<IndexType> sets{parents, size};
    const auto start = edge_starts[i];
    const auto end = edge_ends[i];
    const auto half_block_size = block_size / 2;
    // interior edge: join
    if (start / half_block_size == end / half_block_size) {
        const auto start_rep = sets.find_relaxed_compressing(start);
        const auto end_rep = sets.find_relaxed_compressing(end);
        sets.join(start_rep, end_rep);
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void find_min_cut_neighbors(
    const IndexType* __restrict__ edge_starts,
    const IndexType* __restrict__ edge_ends, size_type num_edges,
    IndexType size, IndexType block_size, IndexType* __restrict__ parents,
    IndexType* __restrict__ mins, IndexType* __restrict__ cut_edge_counter)
{
    const auto i = thread::get_thread_id_flat();
    if (i >= num_edges) {
        return;
    }
    disjoint_sets<IndexType> sets{parents, size};
    const auto start = edge_starts[i];
    const auto end = edge_ends[i];
    const auto half_block_size = block_size / 2;
    if (start / block_size == end / block_size &&
        start / half_block_size < end / half_block_size) {
        // cut edge: count for minimum
        const auto start_rep = sets.find_relaxed_compressing(start);
        kernel::guarded_atomic_min(mins + start_rep, end);
        atomic_add_relaxed(cut_edge_counter, 1);
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void add_fill_edges(
    const IndexType* __restrict__ edge_starts,
    const IndexType* __restrict__ edge_ends, size_type num_edges,
    IndexType size, IndexType block_size, const IndexType* __restrict__ parents,
    const IndexType* __restrict__ mins, IndexType* __restrict__ new_edge_starts,
    IndexType* __restrict__ new_edge_ends,
    IndexType* __restrict__ cut_edge_counter)
{
    const auto i = thread::get_thread_id_flat();
    if (i >= num_edges) {
        return;
    }
    disjoint_sets<const IndexType> sets{parents, size};
    const auto start = edge_starts[i];
    const auto end = edge_ends[i];
    const auto half_block_size = block_size / 2;
    if (start / block_size == end / block_size &&
        start / half_block_size < end / half_block_size) {
        // cut edge: potentially add fill-in
        const auto start_rep = sets.find_weak(start);
        const auto min_node = mins[start_rep];
        if (min_node != end) {
            const auto out_i = atomic_add_relaxed(cut_edge_counter, 1);
            new_edge_starts[out_i] = min_node;
            new_edge_ends[out_i] = end;
        }
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void add_tree_edges(
    const IndexType* __restrict__ parents, const IndexType* __restrict__ mins,
    IndexType size, IndexType* __restrict__ forest_parents)
{
    const auto i = thread::get_thread_id_flat<IndexType>();
    if (i >= size) {
        return;
    }
    disjoint_sets<const IndexType> sets{parents, size};
    if (sets.is_representative_weak(i)) {
        forest_parents[i] = mins[i];
    }
}


}  // namespace kernel


template <typename ValueType>
static void ensure_storage(array<ValueType>& arr, size_type new_size)
{
    if (arr.get_size() >= new_size) {
        return;
    }
    arr.resize(2 * new_size);
}


template <typename IndexType>
void bucket_sort_levels(std::shared_ptr<const DefaultExecutor> exec,
                        const IndexType* in_edge_starts,
                        const IndexType* in_edge_ends, size_type num_edges,
                        array<IndexType>& out_edge_starts,
                        array<IndexType>& out_edge_ends,
                        array<IndexType>& local_block_offsets,
                        array<IndexType>& level_block_offsets)
{
    using Config = kernel::elimination_forest_kernel_config<IndexType>;
    const auto num_bucketsort_blocks =
        ceildiv(num_edges, Config::work_per_threadblock);
    ensure_storage(local_block_offsets,
                   (num_bucketsort_blocks + 1) * Config::max_levels);
    kernel::count_edge_buckets<Config>
        <<<num_bucketsort_blocks, Config::blocksize, 0, exec->get_stream()>>>(
            in_edge_starts, in_edge_ends, num_edges,
            local_block_offsets.get_data());
    kernel::bucket_count_prefixsum<Config>
        <<<1, Config::prefixsum_blocksize, 0, exec->get_stream()>>>(
            local_block_offsets.get_data(), num_bucketsort_blocks);
    std::array<IndexType, Config::max_levels> edge_counts{};
    exec->get_master()->copy_from(
        exec, Config::max_levels,
        local_block_offsets.get_const_data() +
            num_bucketsort_blocks * Config::max_levels,
        edge_counts.data());
    const auto total_edge_count =
        std::accumulate(edge_counts.begin(), edge_counts.end(), IndexType{});
    // ensure_storage();
}


template <typename IndexType>
struct disjoint_set_levels {
    disjoint_set_levels(IndexType num_nodes)
        : num_nodes{num_nodes},
          num_levels{get_num_levels(num_nodes)},
          parent_levels{exec, static_cast<size_type>(num_nodes) *
                                  static_cast<size_type>(num_levels)}
    {
        components::fill_seq_array(exec, parent_levels.get_data(),
                                   static_cast<size_type>(num_nodes));
    }

    void init_from_previous(IndexType level)
    {
        assert(level > 0);
        assert(level < num_levels);
        const auto data =
            parent_levels.get_data() + static_cast<int64>(level) * num_nodes;
        parent_levels->get_executor()->copy(static_cast<size_type>(num_nodes),
                                            data, data + num_nodes);
    }

    disjoint_sets<IndexType> get_level(IndexType level)
    {
        assert(level >= 0);
        assert(level < num_levels);
        const auto data =
            parent_levels.get_data() + static_cast<int64>(level) * num_nodes;
        return disjoint_sets<IndexType>{data, num_nodes};
    }

    IndexType num_nodes;
    IndexType num_levels;
    array<IndexType> parent_levels;
};


template <typename IndexType>
struct elimination_forest_state {
    elimination_forest_state(std::shared_ptr<const DefaultExecutor> exec,
                             IndexType num_nodes)
        : num_nodes{num_nodes},
          num_levels{get_num_levels(num_nodes)},
          parent_levels{exec, static_cast<size_type>(num_nodes) *
                                  static_cast<size_type>(num_levels)}
    {
        components::fill_seq_array(exec, parent_levels.get_data(),
                                   static_cast<size_type>(num_nodes));
    }

    static int get_num_levels(IndexType num_nodes)
    {
        return gko::detail::find_highest_bit(
                   static_cast<std::make_unsigned_t<IndexType>>(num_nodes -
                                                                1)) +
               1;
    }

    void init_cc_from_previous(IndexType level)
    {
        assert(level > 0);
        assert(level < num_levels);
        const auto data =
            parent_levels.get_data() + static_cast<int64>(level) * num_nodes;
        parent_levels->get_executor()->copy(static_cast<size_type>(num_nodes),
                                            data, data + num_nodes);
    }

    disjoint_sets<IndexType> get_cc_level(IndexType level)
    {
        assert(level >= 0);
        assert(level < num_levels);
        const auto data =
            parent_levels.get_data() + static_cast<int64>(level) * num_nodes;
        return disjoint_sets<IndexType>{data, num_nodes};
    }

    IndexType num_nodes;
    IndexType num_levels;
    array<IndexType> parent_levels;
};


template <typename IndexType>
void compute(std::shared_ptr<const DefaultExecutor> exec,
             const IndexType* row_ptrs, const IndexType* cols, size_type usize,
             gko::factorization::elimination_forest<IndexType>& forest)
{
    using unsigned_type = std::make_unsigned_t<IndexType>;
    const auto size = static_cast<IndexType>(usize);
    if (size == 0) {
        return;
    }
    const auto forest_parents = forest.parents.get_data();
    components::fill_array(exec, forest_parents, usize, size);
    const auto nnz =
        static_cast<size_type>(exec->copy_val_to_host(row_ptrs + size));
    if (nnz == 0) {
        return;
    }
    const auto policy = thrust_policy(exec);
    array<IndexType> row_idx_array{exec, nnz};
    const auto row_idxs = row_idx_array.get_data();
    components::convert_ptrs_to_idxs(exec, row_ptrs, size, row_idxs);
    const auto nz_it = thrust::make_zip_iterator(cols, row_idxs);
    const auto edge_predicate =
        [] __device__(thrust::tuple<IndexType, IndexType> tuple) {
            return thrust::get<0>(tuple) < thrust::get<1>(tuple);
        };
    auto num_edges = static_cast<size_type>(
        thrust::count_if(policy, nz_it, nz_it + nnz, edge_predicate));
    array<IndexType> edge_start_array{exec, num_edges};
    array<IndexType> edge_end_array{exec, num_edges};
    auto edge_starts = edge_start_array.get_data();
    auto edge_ends = edge_end_array.get_data();
    auto edge_it = thrust::make_zip_iterator(edge_starts, edge_ends);
    thrust::copy_if(policy, nz_it, nz_it + nnz, edge_it, edge_predicate);
    // round up size to the next power of two
    const auto rounded_up_size = IndexType{1}
                                 << (gko::detail::find_highest_bit(
                                         static_cast<unsigned_type>(size - 1)) +
                                     1);
    // insert fill-in edges top-down
    for (auto block_size = rounded_up_size; block_size > 1; block_size /= 2) {
        // initialize parent array
        array<IndexType> parent_array{exec, usize};
        const auto parents = parent_array.get_data();
        components::fill_seq_array(exec, parents, usize);
        // build connected components
        const auto num_blocks = ceildiv(num_edges, default_block_size);
        kernel::build_conn_components<<<num_blocks, default_block_size, 0,
                                        exec->get_stream()>>>(
            edge_starts, edge_ends, num_edges, size, block_size, parents);
        // now find the smallest upper node adjacent to a cc in a lower block
        array<IndexType> min_array{exec, usize};
        const auto mins = min_array.get_data();
        components::fill_array(exec, mins, usize, size);
        array<IndexType> counter_array{exec, 1};
        components::fill_array(exec, counter_array.get_data(), 1, IndexType{});
        kernel::find_min_cut_neighbors<<<num_blocks, default_block_size, 0,
                                         exec->get_stream()>>>(
            edge_starts, edge_ends, num_edges, size, block_size, parents, mins,
            counter_array.get_data());
        const auto new_edge_space = static_cast<size_type>(
            exec->copy_val_to_host(counter_array.get_const_data()));
        array<IndexType> new_edge_start_array{exec, new_edge_space};
        array<IndexType> new_edge_end_array{exec, new_edge_space};
        // now add new edges for every one of those cut edges
        components::fill_array(exec, counter_array.get_data(), 1, IndexType{});
        kernel::add_fill_edges<<<num_blocks, default_block_size, 0,
                                 exec->get_stream()>>>(
            edge_starts, edge_ends, num_edges, size, block_size, parents, mins,
            new_edge_start_array.get_data(), new_edge_end_array.get_data(),
            counter_array.get_data());
        const auto new_edges =
            exec->copy_val_to_host(counter_array.get_const_data());
        array<IndexType> new_full_edge_start_array{exec, num_edges + new_edges};
        array<IndexType> new_full_edge_end_array{exec, num_edges + new_edges};
        exec->copy(num_edges, edge_starts,
                   new_full_edge_start_array.get_data());
        exec->copy(num_edges, edge_ends, new_full_edge_end_array.get_data());
        exec->copy(new_edges, new_edge_start_array.get_data(),
                   new_full_edge_start_array.get_data() + num_edges);
        exec->copy(new_edges, new_edge_end_array.get_data(),
                   new_full_edge_end_array.get_data() + num_edges);
        num_edges = new_full_edge_start_array.get_size();
        edge_start_array = std::move(new_full_edge_start_array);
        edge_end_array = std::move(new_full_edge_end_array);
        edge_starts = edge_start_array.get_data();
        edge_ends = edge_end_array.get_data();
    }
    // initialize parent array
    array<IndexType> parent_array{exec, usize};
    const auto parents = parent_array.get_data();
    components::fill_seq_array(exec, parents, usize);
    for (IndexType block_size = 2; block_size <= rounded_up_size;
         block_size *= 2) {
        // build connected components
        const auto num_blocks = ceildiv(num_edges, default_block_size);
        kernel::build_conn_components<<<num_blocks, default_block_size, 0,
                                        exec->get_stream()>>>(
            edge_starts, edge_ends, num_edges, size, block_size, parents);
        // now find the smallest upper node adjacent to a cc in a lower block
        array<IndexType> min_array{exec, usize};
        const auto mins = min_array.get_data();
        components::fill_array(exec, mins, usize, size);
        array<IndexType> counter_array{exec, 1};
        components::fill_array(exec, counter_array.get_data(), 1, IndexType{});
        kernel::find_min_cut_neighbors<<<num_blocks, default_block_size, 0,
                                         exec->get_stream()>>>(
            edge_starts, edge_ends, num_edges, size, block_size, parents, mins,
            counter_array.get_data());
        // add edges
        const auto num_node_blocks = ceildiv(size, default_block_size);
        kernel::add_tree_edges<<<num_node_blocks, default_block_size, 0,
                                 exec->get_stream()>>>(parents, mins, size,
                                                       forest_parents);
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ELIMINATION_FOREST_COMPUTE);


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
