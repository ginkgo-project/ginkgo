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
#include "common/cuda_hip/components/sorting.hpp"
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
               2 * static_cast<size_type>(num_nodes) + 3;
    }

    mst_state(std::shared_ptr<const DefaultExecutor> exec, IndexType num_nodes,
              IndexType num_edges, IndexType* input_sources,
              IndexType* input_targets, IndexType* tree_sources,
              IndexType* tree_targets, array<IndexType> workspace)
        : exec_{exec},
          num_nodes_{num_nodes},
          num_edges_{num_edges},
          input_sources_{input_sources},
          input_targets_{input_targets},
          work_array_{std::move(workspace)},
          tree_sources_{tree_sources},
          tree_targets_{tree_targets},
          flip_{}
    {
        assert(work_array_.get_executor() == exec);
        assert(work_array_.get_size() >=
               storage_requirement(num_nodes, num_edges));
        reset(input_sources, input_targets_, num_edges_);
    }

    mst_state(std::shared_ptr<const DefaultExecutor> exec, IndexType num_nodes,
              IndexType num_edges, IndexType* input_sources,
              IndexType* input_targets, IndexType* tree_sources,
              IndexType* tree_targets)
        : mst_state(
              exec, num_nodes, num_edges, input_sources, input_targets,
              tree_sources, tree_targets,
              array<IndexType>{exec, storage_requirement(num_nodes, num_edges)})
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
        // reset output counter to 0
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
        std::array<IndexType, 3> zeros{};
        exec_->copy_from(exec_->get_master(), 3, zeros.data(), tree_counter());
        components::fill_array(exec_, min_edges(),
                               static_cast<size_type>(num_nodes_),
                               min_edge_sentinel);
        components::fill_seq_array(exec_, parents(),
                                   static_cast<size_type>(num_nodes_));
        exec_->copy(num_edges_, input_sources_, output_wl_sources());
        exec_->copy(num_edges_, input_targets_, output_wl_targets());
        components::fill_seq_array(exec_, output_wl_edge_ids(),
                                   static_cast<size_type>(num_edges_));
    }

    void run()
    {
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
                                           min_edge_sentinel);
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

    constexpr static auto min_edge_sentinel =
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
__global__ __launch_bounds__(default_block_size) void add_fill_and_tree_edges(
    const IndexType* __restrict__ edge_sources,
    const IndexType* __restrict__ edge_targets, size_type num_edges,
    IndexType size, const IndexType* __restrict__ parents,
    const IndexType* __restrict__ mins,
    IndexType* __restrict__ new_edge_sources,
    IndexType* __restrict__ new_edge_targets,
    IndexType* __restrict__ forest_parents)
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
    assert(min_node < size);
    // we may have min_node == tgt, but that will get filtered out in the
    // sorting step
    new_edge_sources[i] = min_node;
    new_edge_targets[i] = tgt;
    store_relaxed_local(forest_parents + src_rep, min_node);
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void add_tree_edges(
    const IndexType* __restrict__ cc_parents,
    const IndexType* __restrict__ mins, IndexType size,
    IndexType* __restrict__ forest_parents)
{
    const auto i = thread::get_thread_id_flat<IndexType>();
    if (i >= size) {
        return;
    }
    disjoint_sets<const IndexType> sets{cc_parents, size};
    if (sets.is_representative_weak(i) && mins[i] < size) {
        forest_parents[i] = mins[i];
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
        // TODO do this manually and explicitly
        if (predicate(value)) {
            const auto bucket = bucket_op(value);
            const auto bucket_offset = sorted_offsets[bucket];
            const auto relative_pos = i - bucket_offset;
            assert(relative_pos >= 0);
            const auto out_pos = global_to_sort_offsets[bucket] +
                                 sorted_offsets[bucket] + relative_pos;
            *(out_begin + out_pos) = value;
        }
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


template <typename ValueType>
static void ensure_storage(array<ValueType>& arr, size_type new_size)
{
    if (arr.get_size() >= new_size) {
        return;
    }
    arr.resize(2 * new_size);
}


template <typename IndexType>
struct elimination_forest_algorithm_state {
    constexpr static int num_buckets = CHAR_BIT * sizeof(IndexType) - 1;
    elimination_forest_algorithm_state(
        std::shared_ptr<const DefaultExecutor> exec, IndexType num_nodes,
        IndexType num_edges)
        : exec{exec},
          workspace{exec, storage_requirement(num_nodes, num_edges)},
          num_nodes{num_nodes},
          edge_capacity{2 * num_edges},
          num_levels{
              gko::detail::find_highest_bit(
                  static_cast<std::make_unsigned_t<IndexType>>(num_nodes - 1)) +
              1},
          ceil_num_nodes{IndexType{1} << num_levels},
          bucket_ranges{},
          flip{}
    {
        bucket_ranges.back() = num_edges;
    }

    static size_type storage_requirement(size_type num_nodes,
                                         size_type num_edges)
    {
        const auto edge_capacity = 2 * num_edges;
        return edge_capacity      // buf1_edge_sources
               + edge_capacity    // buf1_edge_targets
               + edge_capacity    // buf2_edge_sources
               + edge_capacity    // buf2_edge_targets
               + num_nodes        // tree_parents
               + num_nodes + 2    // tree_child_ptrs
               + num_nodes        // tree_children
               + num_nodes        // cc_parents
               + num_nodes        // cc_mins
               + num_buckets + 1  // device_bucket_ranges
               + 1                // tree_counter
               + bucket_sort_workspace_size<num_buckets>(edge_capacity);
    }

    IndexType* buf1_edge_sources() { return workspace.get_data(); }

    IndexType* buf1_edge_targets()
    {
        return buf1_edge_sources() + edge_capacity;
    }

    IndexType* buf2_edge_sources()
    {
        return buf1_edge_targets() + edge_capacity;
    }

    IndexType* buf2_edge_targets()
    {
        return buf2_edge_sources() + edge_capacity;
    }

    IndexType* tree_parents() { return buf2_edge_targets() + edge_capacity; }

    IndexType* tree_child_ptrs() { return tree_parents() + num_nodes; }

    IndexType* tree_children() { return tree_child_ptrs() + num_nodes + 2; }

    IndexType* cc_parents() { return tree_children() + num_nodes; }

    IndexType* cc_mins() { return cc_parents() + num_nodes; }

    IndexType* device_bucket_ranges() { return cc_mins() + num_nodes; }

    IndexType* tree_counter() { return device_bucket_ranges() + num_nodes + 1; }

    IndexType* bucket_sort_workspace() { return tree_counter() + 1; }

    array<IndexType> bucket_sort_workspace_view()
    {
        return make_array_view(
            exec, bucket_sort_workspace_size<num_buckets>(edge_capacity),
            bucket_sort_workspace());
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
        assert(sources.get_size() == ends.get_size());
        assert(sources.get_size() <= edge_capacity);
        exec->copy(sources.get_size(), sources.get_const_data(),
                   out_edge_sources());
        exec->copy(ends.get_size(), ends.get_const_data(), out_edge_targets());
        bucket_ranges.back() = static_cast<IndexType>(sources.get_size());
        components::fill_array(exec, tree_parents(),
                               static_cast<size_type>(num_nodes), num_nodes);
        output_to_input();
    }

    void bucket_sort_input()
    {
        auto sort_workspace = bucket_sort_workspace_view();
        auto it =
            thrust::make_zip_iterator(in_edge_sources(), in_edge_targets());
        auto out_it =
            thrust::make_zip_iterator(out_edge_sources(), out_edge_targets());
        array<IndexType> tmp{exec};
        bucket_ranges = bucket_sort<num_buckets>(
            exec, it, it + num_edges(), out_it,
            [] __device__(thrust::tuple<IndexType, IndexType> edge) {
                return kernel::get_edge_level(thrust::get<0>(edge),
                                              thrust::get<1>(edge));
            },
            tmp);  // sort_workspace);
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

    void find_connected_components(int level)
    {
        components::fill_seq_array(exec, cc_parents(),
                                   static_cast<size_type>(num_nodes));
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
        const auto cut_edge_range = get_cut_edge_range(level);
        if (cut_edge_range.size() > 0) {
            const auto num_blocks =
                ceildiv(cut_edge_range.size(), default_block_size);
            /*if (num_cut_edges < num_nodes) {
                kernel::add_fill_and_tree_edges<<<
                    num_blocks, default_block_size, 0, exec->get_stream()>>>(
                    in_edge_sources() + begin_cut_edges,
                    in_edge_targets() + begin_cut_edges, num_cut_edges,
            num_nodes, cc_parents(), cc_mins(), fill_edge_sources(),
                    fill_edge_targets(), tree_parents());
            } else*/
            {
                const auto num_node_blocks =
                    ceildiv(num_nodes, default_block_size);
                kernel::add_fill_edges<<<num_blocks, default_block_size, 0,
                                         exec->get_stream()>>>(
                    in_edge_sources() + cut_edge_range.begin_index(),
                    in_edge_targets() + cut_edge_range.begin_index(),
                    cut_edge_range.size(), num_nodes, cc_parents(), cc_mins(),
                    fill_edge_sources(), fill_edge_targets());
                kernel::add_tree_edges<<<num_node_blocks, default_block_size, 0,
                                         exec->get_stream()>>>(
                    cc_parents(), cc_mins(), num_nodes, tree_parents());
            }
        }
    }

    void bucket_sort_fill_edges(int level)
    {
        const auto cut_edge_range = get_cut_edge_range(level);
        using namespace gko::kernels::GKO_DEVICE_NAMESPACE::kernel;
        using config = bucket_sort_config<num_buckets>;
        const auto num_fill_edges = cut_edge_range.size();
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
        // removing cut edges from the current level and above
        // and loops (which allow us to avoid atomics)
        auto predicate =
            [level] __device__(thrust::tuple<IndexType, IndexType> edge) {
                const auto src = thrust::get<0>(edge);
                const auto end = thrust::get<1>(edge);
                return src != end && kernel::get_edge_level(src, end) <= level;
            };
        assert(num_fill_edges <= edge_capacity);
        const auto num_blocks = std::max<IndexType>(
            1, static_cast<IndexType>(
                   ceildiv(num_fill_edges, config::items_per_threadblock)));
        array<IndexType> tmp{exec,
                             bucket_sort_workspace_size<num_buckets, config>(
                                 std::max<IndexType>(1, num_fill_edges))};
        const auto workspace = tmp.get_data();  // bucket_sort_workspace();
        bucket_sort_filter_count<config>
            <<<num_blocks, config::threadblock_size, 0, exec->get_stream()>>>(
                fill_it, num_fill_edges, predicate, bucket_op, workspace);
        bucket_sort_prefixsum<config>
            <<<1, config::num_buckets, 0, exec->get_stream()>>>(workspace,
                                                                num_blocks);
        const auto global_offsets = workspace + num_buckets * num_blocks;
        components::prefix_sum_nonnegative(exec, global_offsets,
                                           num_buckets + 1);
        exec->get_master()->copy_from(exec, num_buckets + 1, global_offsets,
                                      fill_bucket_ranges.data());
        const auto num_merge_blocks = static_cast<IndexType>(
            ceildiv(num_edges(), config::threadblock_size));
        components::fill_array(exec, out_edge_sources(), edge_capacity,
                               IndexType{-1});
        components::fill_array(exec, out_edge_targets(), edge_capacity,
                               IndexType{-1});
        kernel::bucket_sort_merge_filter_distribute<config>
            <<<num_blocks + num_merge_blocks, config::threadblock_size, 0,
               exec->get_stream()>>>(
                fill_it, in_it, out_it, num_fill_edges, num_edges(), num_blocks,
                predicate, bucket_op, workspace, device_bucket_ranges());
        for (int i = 0; i <= level; i++) {
            bucket_ranges[i] += fill_bucket_ranges[i];
        }
        // all edges >= level were filtered out
        for (int i = level + 1; i <= num_levels; i++) {
            bucket_ranges[i] = bucket_ranges[level];
        }
        exec->copy_from(exec->get_master(), num_buckets + 1,
                        bucket_ranges.data(), device_bucket_ranges());
    }

    void run()
    {
        bucket_sort_input();
        for (int level = num_levels - 1; level >= 0; level--) {
            find_connected_components(level);
            find_min_cut_neighbors(level);
            add_fill_and_tree_edges(level);
            bucket_sort_fill_edges(level);
            output_to_input();
        }
    }

    std::shared_ptr<const DefaultExecutor> exec;
    array<IndexType> workspace;
    IndexType num_nodes;
    IndexType edge_capacity;
    int num_levels;
    IndexType ceil_num_nodes;
    std::array<IndexType, num_buckets + 1> bucket_ranges;
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
    exec->copy(size, state.tree_parents(), forest.parents.get_data());
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
