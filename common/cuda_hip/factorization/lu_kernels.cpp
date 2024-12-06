// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/lu_kernels.hpp"

#include <algorithm>
#include <memory>

#include <thrust/copy.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include <ginkgo/core/matrix/csr.hpp>

#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/thrust.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "common/cuda_hip/components/syncfree.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "core/base/allocator.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The LU namespace.
 *
 * @ingroup factor
 */
namespace lu_factorization {


constexpr static int default_block_size = 512;


namespace kernel {


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void initialize(
    const IndexType* __restrict__ mtx_row_ptrs,
    const IndexType* __restrict__ mtx_cols,
    const ValueType* __restrict__ mtx_vals,
    const IndexType* __restrict__ factor_row_ptrs,
    const IndexType* __restrict__ factor_cols,
    const IndexType* __restrict__ factor_storage_offsets,
    const int32* __restrict__ factor_storage,
    const int64* __restrict__ factor_row_descs,
    ValueType* __restrict__ factor_vals, IndexType* __restrict__ diag_idxs,
    size_type num_rows)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    if (row >= num_rows) {
        return;
    }
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    // first zero out this row of the factor
    const auto factor_begin = factor_row_ptrs[row];
    const auto factor_end = factor_row_ptrs[row + 1];
    const auto lane = static_cast<int>(warp.thread_rank());
    for (auto nz = factor_begin + lane; nz < factor_end;
         nz += config::warp_size) {
        factor_vals[nz] = zero<ValueType>();
    }
    warp.sync();
    // then fill in the values from mtx
    gko::matrix::csr::device_sparsity_lookup<IndexType> lookup{
        factor_row_ptrs, factor_cols,      factor_storage_offsets,
        factor_storage,  factor_row_descs, row};
    const auto row_begin = mtx_row_ptrs[row];
    const auto row_end = mtx_row_ptrs[row + 1];
    for (auto nz = row_begin + lane; nz < row_end; nz += config::warp_size) {
        const auto col = mtx_cols[nz];
        const auto val = mtx_vals[nz];
        factor_vals[lookup.lookup_unsafe(col) + factor_begin] = val;
    }
    if (lane == 0) {
        diag_idxs[row] = lookup.lookup_unsafe(row) + factor_begin;
    }
}


template <bool full_fillin, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void factorize(
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ cols,
    const IndexType* __restrict__ storage_offsets,
    const int32* __restrict__ storage, const int64* __restrict__ row_descs,
    const IndexType* __restrict__ diag_idxs, ValueType* __restrict__ vals,
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
    // for each lower triangular entry: eliminate with corresponding row
    for (auto lower_nz = row_begin; lower_nz < row_diag; lower_nz++) {
        const auto dep = cols[lower_nz];
        // we can load the value before synchronizing because the following
        // updates only go past the diagonal of the dependency row, i.e. at
        // least column dep + 1
        const auto val = vals[lower_nz];
        const auto diag_idx = diag_idxs[dep];
        const auto dep_end = row_ptrs[dep + 1];
        scheduler.wait(dep);
        const auto diag = vals[diag_idx];
        const auto scale = val / diag;
        if (lane == 0) {
            vals[lower_nz] = scale;
        }
        // subtract all entries past the diagonal
        for (auto upper_nz = diag_idx + 1 + lane; upper_nz < dep_end;
             upper_nz += config::warp_size) {
            const auto upper_col = cols[upper_nz];
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
    scheduler.mark_ready();
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void symbolic_factorize_simple(
    const IndexType* __restrict__ mtx_row_ptrs,
    const IndexType* __restrict__ mtx_cols,
    const IndexType* __restrict__ factor_row_ptrs,
    const IndexType* __restrict__ factor_cols,
    const IndexType* __restrict__ storage_offsets,
    const int32* __restrict__ storage, const int64* __restrict__ row_descs,
    IndexType* __restrict__ diag_idxs, ValueType* __restrict__ factor_vals,
    IndexType* __restrict__ out_row_nnz, syncfree_storage dep_storage,
    size_type num_rows)
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
    const auto factor_begin = factor_row_ptrs[row];
    const auto factor_end = factor_row_ptrs[row + 1];
    const auto mtx_begin = mtx_row_ptrs[row];
    const auto mtx_end = mtx_row_ptrs[row + 1];
    gko::matrix::csr::device_sparsity_lookup<IndexType> lookup{
        factor_row_ptrs, factor_cols, storage_offsets,
        storage,         row_descs,   static_cast<size_type>(row)};
    const auto row_diag = lookup.lookup_unsafe(row) + factor_begin;
    // fill with zeros first
    for (auto nz = factor_begin + lane; nz < factor_end;
         nz += config::warp_size) {
        factor_vals[nz] = zero<float>();
    }
    warp.sync();
    // then fill in the system matrix
    for (auto nz = mtx_begin + lane; nz < mtx_end; nz += config::warp_size) {
        const auto col = mtx_cols[nz];
        factor_vals[lookup.lookup_unsafe(col) + factor_begin] = one<float>();
    }
    // finally set diagonal and store diagonal index
    if (lane == 0) {
        diag_idxs[row] = row_diag;
        factor_vals[row_diag] = one<float>();
    }
    warp.sync();
    // for each lower triangular entry: eliminate with corresponding row
    for (auto lower_nz = factor_begin; lower_nz < row_diag; lower_nz++) {
        const auto dep = factor_cols[lower_nz];
        const auto dep_end = factor_row_ptrs[dep + 1];
        scheduler.wait(dep);
        // read the diag entry after we are sure it was written.
        const auto diag_idx = diag_idxs[dep];
        if (factor_vals[lower_nz] == one<float>()) {
            // eliminate with upper triangle/entries past the diagonal
            for (auto upper_nz = diag_idx + 1 + lane; upper_nz < dep_end;
                 upper_nz += config::warp_size) {
                const auto upper_col = factor_cols[upper_nz];
                const auto upper_val = factor_vals[upper_nz];
                const auto output_pos =
                    lookup.lookup_unsafe(upper_col) + factor_begin;
                if (upper_val == one<float>()) {
                    factor_vals[output_pos] = one<float>();
                }
            }
        }
    }
    scheduler.mark_ready();
    IndexType row_nnz{};
    for (auto nz = factor_begin + lane; nz < factor_end;
         nz += config::warp_size) {
        row_nnz += factor_vals[nz] == one<float>() ? 1 : 0;
    }
    row_nnz = reduce(warp, row_nnz, thrust::plus<IndexType>{});
    if (lane == 0) {
        out_row_nnz[row] = row_nnz;
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Csr<ValueType, IndexType>* mtx,
                const IndexType* factor_lookup_offsets,
                const int64* factor_lookup_descs,
                const int32* factor_lookup_storage, IndexType* diag_idxs,
                matrix::Csr<ValueType, IndexType>* factors)
{
    const auto num_rows = mtx->get_size()[0];
    if (num_rows > 0) {
        const auto num_blocks =
            ceildiv(num_rows, default_block_size / config::warp_size);
        kernel::initialize<<<num_blocks, default_block_size, 0,
                             exec->get_stream()>>>(
            mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
            as_device_type(mtx->get_const_values()),
            factors->get_const_row_ptrs(), factors->get_const_col_idxs(),
            factor_lookup_offsets, factor_lookup_storage, factor_lookup_descs,
            as_device_type(factors->get_values()), diag_idxs, num_rows);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LU_INITIALIZE);


template <typename ValueType, typename IndexType>
void factorize(std::shared_ptr<const DefaultExecutor> exec,
               const IndexType* lookup_offsets, const int64* lookup_descs,
               const int32* lookup_storage, const IndexType* diag_idxs,
               matrix::Csr<ValueType, IndexType>* factors, bool full_fillin,
               array<int>& tmp_storage)
{
    const auto num_rows = factors->get_size()[0];
    if (num_rows > 0) {
        syncfree_storage storage(exec, tmp_storage, num_rows);
        const auto num_blocks =
            ceildiv(num_rows, default_block_size / config::warp_size);
        if (full_fillin) {
            kernel::factorize<true>
                <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
                    factors->get_const_row_ptrs(),
                    factors->get_const_col_idxs(), lookup_offsets,
                    lookup_storage, lookup_descs, diag_idxs,
                    as_device_type(factors->get_values()), storage, num_rows);
        } else {
            kernel::factorize<false>
                <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
                    factors->get_const_row_ptrs(),
                    factors->get_const_col_idxs(), lookup_offsets,
                    lookup_storage, lookup_descs, diag_idxs,
                    as_device_type(factors->get_values()), storage, num_rows);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LU_FACTORIZE);


template <typename IndexType>
void symbolic_factorize_simple(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* row_ptrs,
    const IndexType* col_idxs, const IndexType* lookup_offsets,
    const int64* lookup_descs, const int32* lookup_storage,
    matrix::Csr<float, IndexType>* factors, IndexType* out_row_nnz)
{
    const auto num_rows = factors->get_size()[0];
    const auto factor_row_ptrs = factors->get_const_row_ptrs();
    const auto factor_cols = factors->get_const_col_idxs();
    const auto factor_vals = factors->get_values();
    array<IndexType> diag_idx_array{exec, num_rows};
    array<int> tmp_storage{exec};
    const auto diag_idxs = diag_idx_array.get_data();
    if (num_rows > 0) {
        syncfree_storage dep_storage(exec, tmp_storage, num_rows);
        const auto num_blocks =
            ceildiv(num_rows, default_block_size / config::warp_size);
        kernel::symbolic_factorize_simple<<<num_blocks, default_block_size, 0,
                                            exec->get_stream()>>>(
            row_ptrs, col_idxs, factor_row_ptrs, factor_cols, lookup_offsets,
            lookup_storage, lookup_descs, diag_idxs, factor_vals, out_row_nnz,
            dep_storage, num_rows);
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_LU_SYMMETRIC_FACTORIZE_SIMPLE);


struct first_eq_one_functor {
    template <typename Pair>
    __device__ __forceinline__ bool operator()(Pair pair) const
    {
        return thrust::get<0>(pair) == one<float>();
    }
};


struct return_second_functor {
    template <typename Pair>
    __device__ __forceinline__ auto operator()(Pair pair) const
    {
        return thrust::get<1>(pair);
    }
};


template <typename IndexType>
void symbolic_factorize_simple_finalize(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<float, IndexType>* factors, IndexType* out_col_idxs)
{
    const auto col_idxs = factors->get_const_col_idxs();
    const auto vals = factors->get_const_values();
    const auto input_it =
        thrust::make_zip_iterator(thrust::make_tuple(vals, col_idxs));
    const auto output_it = thrust::make_transform_output_iterator(
        out_col_idxs, return_second_functor{});
    thrust::copy_if(thrust_policy(exec), input_it,
                    input_it + factors->get_num_stored_elements(), output_it,
                    first_eq_one_functor{});
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_LU_SYMMETRIC_FACTORIZE_SIMPLE_FINALIZE);


template <typename IndexType>
struct double_buffered_frontier {
    IndexType* input;
    IndexType* output;
    struct shared_storage {
        IndexType input_pos;
        IndexType output_pos;
    };
    shared_storage& shared;

    __device__ double_buffered_frontier(IndexType* input, IndexType* output,
                                        shared_storage& shared)
        : input{input}, output{output}, shared{shared}
    {
        if (threadIdx.x == 0) {
            shared.input_pos = 0;
            shared.output_pos = 0;
        }
    }

    __device__ void add(IndexType value)
    {
        output[atomic_add_relaxed_shared(&shared.output_pos, 1)] = value;
    }

    __device__ IndexType output_to_input()
    {
        if (threadIdx.x == 0) {
            shared.input_pos = shared.output_pos;
            shared.output_pos = 0;
        }
        // swap input and output
        auto old_input = input;
        input = output;
        output = old_input;
        __syncthreads();
        const auto frontier_size = shared.input_pos;
        __syncthreads();
        return frontier_size;
    }
};


template <typename Config, typename IndexType>
class device_block_memory_pool {
    constexpr static auto block_size = Config::block_size;

public:
    constexpr device_block_memory_pool(IndexType* pool, IndexType* next_block,
                                       IndexType num_blocks,
                                       IndexType* block_counter)
        : pool_{pool},
          next_block_{next_block},
          num_blocks_{num_blocks},
          block_counter_{block_counter}
    {}

    struct block {
        IndexType id;
        IndexType* data;
    };

    __device__ block get_block(IndexType block_id) const
    {
        assert(block_id >= 0);
        assert(block_id < num_blocks_);
        return block{block_id, pool_ + block_id * block_size};
    }

    // write a block successor with a threadblock-local atomic release store
    __device__ void set_next_block_release(IndexType block_id,
                                           IndexType next_block)
    {
        assert(block_id >= 0);
        assert(block_id < num_blocks_);
        assert(next_block >= 0);
        assert(next_block < num_blocks_);
        // TOCTOU issue, but fixing it would require a CAS, so we don't care
        // since it's just an assertion that just might accidentally not fail
        // when it should be failing if next_block[block_id] gets updated
        // between this and the following statement.
        assert(load_relaxed_local(next_block_ + block_id) ==
               invalid_index<IndexType>());
        store_release_local(next_block_ + block_id, next_block);
    }

    // read a block successor with a threadblock-local atomic acquire load
    __device__ IndexType get_next_block_acquire(IndexType block_id) const
    {
        assert(block_id >= 0);
        assert(block_id < num_blocks_);
        return load_acquire_local(next_block_ + block_id);
    }

    // read a block successor with a non-atomic load
    __device__ IndexType get_next_block(IndexType block_id) const
    {
        assert(block_id >= 0);
        assert(block_id < num_blocks_);
        return next_block_[block_id];
    }

    __device__ IndexType alloc()
    {
        return atomic_add_relaxed(block_counter_, 1);
    }

private:
    IndexType* pool_;
    IndexType* next_block_;
    IndexType num_blocks_;
    IndexType* block_counter_;
};


// A linked list of blocks from a device_block_memory_pool
template <typename Config, typename IndexType>
class threadblock_shared_block_list {
    constexpr static auto block_size = Config::block_size;
    static_assert((block_size & (block_size - 1)) == 0,
                  "block_size should be a power of two");
    using pool_type = device_block_memory_pool<Config, IndexType>;
    using block_type = typename pool_type::block;

public:
    struct shared_storage {
        // a single row will not exceed 2B entries
        int output_idx;
    };

    __device__ threadblock_shared_block_list(pool_type pool,
                                             IndexType* out_block_index,
                                             shared_storage& shared)
        : pool_{pool}, shared_{shared}
    {
        if (threadIdx.x == 0) {
            *out_block_index = pool.alloc();
            shared_.output_idx = 0;
        }
        __syncthreads();
        const auto first_block_id = *out_block_index;
        available_size = block_size;
        current_block = pool_.get_block(first_block_id);
    }

    __device__ void output(IndexType value)
    {
        const auto output_idx =
            atomic_add_relaxed_shared(&shared_.output_idx, 1);
        // this needs to be a while loop because we could output more than
        // block_size entries between two output entries in this thread
        while (output_idx >= available_size) {
            auto new_block = invalid_index<IndexType>();
            if constexpr (Config::debug_output) {
                printf(
                    "threadblock_shared_block_list: Block %d (%d) not "
                    "sufficient for index %d\n",
                    int(current_block.id), int(available_size), output_idx);
            }
            // wait until the block was allocated
            while ((new_block = pool_.get_next_block_acquire(
                        current_block.id)) == invalid_index<IndexType>()) {
                // we need to check the existing new_block value once to
                // facilitate memory reuse if we already have preallocated
                // blocks available
                if (output_idx == available_size) {
                    new_block = pool_.alloc();
                    if constexpr (Config::debug_output) {
                        printf(
                            "threadblock_shared_block_list: Allocated new "
                            "block %d after %d for index %d\n",
                            int(new_block), int(current_block.id), output_idx);
                    }
                    pool_.set_next_block_release(current_block.id, new_block);
                    break;
                }
                // TODO potentially nanosleep here
            }
            // move to next block
            current_block = pool_.get_block(new_block);
            available_size += block_size;
        }
        const auto block_base = available_size - block_size;
        assert(output_idx >= block_base);
        current_block.data[output_idx - block_base] = value;
    }

private:
    pool_type pool_;
    shared_storage& shared_;

    // these two need to be kept in sync
    // using "current block" to refer to the last block known to this thread.
    // all previous blocks should be non-empty
    int available_size;  // total memory available in all blocks up until the
                         // current block
    block_type current_block;  // information about the current block
};


template <typename Config, typename IndexType>
class block_memory_pool {
    constexpr static auto block_size = Config::block_size;
    static_assert((block_size & (block_size - 1)) == 0,
                  "block_size should be a power of two");

public:
    explicit block_memory_pool(std::shared_ptr<const DefaultExecutor> exec,
                               IndexType num_blocks)
        : pool_{exec, static_cast<size_type>(num_blocks * block_size)},
          next_block_{exec, static_cast<size_type>(num_blocks)},
          counter_{exec, 1},
          overflow_{exec, 0}
    {
        // initialize overflow array to nullptr
        thrust::uninitialized_fill_n(thrust_policy(exec), overflow_.get_data(),
                                     overflow_.get_size(), nullptr);
        // initialize counters to zero
        thrust::uninitialized_fill_n(thrust_policy(exec), counter_.get_data(),
                                     counter_.get_size(), 0);
        // initialize counters to invalid_index
        thrust::uninitialized_fill_n(
            thrust_policy(exec), next_block_.get_data(), next_block_.get_size(),
            invalid_index<IndexType>());
    }

    device_block_memory_pool<Config, IndexType> device_view()
    {
        return {pool_.get_data(), next_block_.get_data(),
                static_cast<IndexType>(pool_.get_size() / block_size),
                counter_.get_data()};
    }

    block_memory_pool(block_memory_pool&&) = delete;

    block_memory_pool(const block_memory_pool&) = delete;

    block_memory_pool& operator=(block_memory_pool&&) = delete;

    block_memory_pool& operator=(const block_memory_pool&) = delete;

    ~block_memory_pool() { free_overflow(); }

    void free_overflow()
    {
        // delete all allocated overflow entries
        thrust::for_each_n(
            thrust_policy(std::static_pointer_cast<const DefaultExecutor>(
                overflow_.get_executor())),
            overflow_.get_const_data(), overflow_.get_size(),
            [] __device__(ValueType * ptr) {
                if (ptr) {
                    free(ptr);
                }
            });
    }

private:
    array<IndexType> pool_;
    array<IndexType> next_block_;
    array<IndexType> counter_;
    array<IndexType*> overflow_;
};


struct symbolic_factorize_config {
    constexpr static bool skip_visited = false;
    constexpr static bool debug = false;
    constexpr static bool debug_output = false;
    constexpr static bool debug_pool = true;
    constexpr static int block_size = 32;
};


template <typename Config, typename IndexType>
__global__ void symbolic_factorize_gsofa(
    const IndexType* row_ptrs, const IndexType* cols, IndexType size,
    IndexType* atomics, IndexType* global_max_id, IndexType* global_fill,
    IndexType* global_frontier, IndexType* global_new_frontier,
    IndexType* out_row_sizes, IndexType* out_block_ids,
    device_block_memory_pool<Config, IndexType> pool)
{
    using block_list_type = threadblock_shared_block_list<Config, IndexType>;
    __shared__ typename double_buffered_frontier<IndexType>::shared_storage
        frontier_storage;
    __shared__ typename block_list_type::shared_storage block_list_storage;
    __shared__ IndexType source_idx;
    const auto frontier = global_frontier + size * blockIdx.x;
    const auto new_frontier = global_new_frontier + size * blockIdx.x;
    const auto max_id = global_max_id + size * blockIdx.x;
    const auto fill = global_fill + size * blockIdx.x;
    if (threadIdx.x == 0) {
        source_idx = atomic_add_relaxed(atomics, 1);
        if constexpr (Config::debug) {
            printf("block %d handling source %d\n", blockIdx.x,
                   int(source_idx));
        }
    }
    __syncthreads();
    IndexType source = source_idx;
    while (source < size) {
        double_buffered_frontier<IndexType> frontiers{frontier, new_frontier,
                                                      frontier_storage};
        for (IndexType i = threadIdx.x; i < size; i += blockDim.x) {
            max_id[i] = device_numeric_limits<IndexType>::max;
        }
        block_list_type blocks{pool, &out_block_ids[source],
                               block_list_storage};
        __syncthreads();
        const auto output = [&](IndexType col) {
            if constexpr (Config::debug) {
                printf("%d: Output %d\n", int(source), int(col));
            }
            blocks.output(col);
        };
        const auto row_begin = row_ptrs[source];
        const auto row_end = row_ptrs[source + 1];
        // TODO check this works with missing diagonals
        if (threadIdx.x == 0) {
            fill[source] = 0;
            max_id[source] = 0;
            output(source);
        }
        for (auto i = row_begin + threadIdx.x; i < row_end; i += blockDim.x) {
            const auto col = cols[i];
            if (col == source) {
                continue;
            }
            fill[col] = 0;
            max_id[col] = 0;
            output(col);
            if (col < source) {
                if constexpr (Config::debug) {
                    printf("%d: Queueing %d initially\n", int(source),
                           int(col));
                }
                frontiers.add(col);
            }
        }
        auto frontier_size = frontiers.output_to_input();
        constexpr auto fine_size = config::warp_size;
        const auto coarse_size = blockDim.x / fine_size;
        int frontier_number = 0;
        while (frontier_size > 0) {
            const auto coarse_id = threadIdx.x / fine_size;
            const auto fine_id = threadIdx.x % fine_size;
            for (IndexType frontier_i = coarse_id; frontier_i < frontier_size;
                 frontier_i += coarse_size) {
                const auto frontier = frontiers.input[frontier_i];
                const auto new_max_id = max(frontier, max_id[frontier]);
                const auto frontier_begin = row_ptrs[frontier];
                const auto frontier_end = row_ptrs[frontier + 1];
                for (auto neighbor_i = frontier_begin + fine_id;
                     neighbor_i < frontier_end; neighbor_i += fine_size) {
                    const auto neighbor = cols[neighbor_i];
                    if constexpr (Config::debug) {
                        printf("%d: Considering %d via %d\n", int(source),
                               int(neighbor), int(frontier));
                    }
                    if constexpr (Config::skip_visited) {
                        if (fill[neighbor] == source) {
                            continue;
                        }
                    }
                    if (atomic_min_relaxed_local(&max_id[neighbor],
                                                 new_max_id) > new_max_id) {
                        if (neighbor > new_max_id) {
                            // fill is increasing monotonically
                            const auto result = atomic_max_relaxed_local(
                                &fill[neighbor], source);
                            if (result < source) {
                                output(neighbor);
                            } else {
                                if constexpr (Config::debug) {
                                    printf(
                                        "%d: Skipping %d via %d because it was "
                                        "already output via %d\n",
                                        int(source), int(neighbor),
                                        int(frontier), int(result));
                                }
                                continue;
                            }
                        }
                        if (neighbor < source) {
                            if constexpr (Config::debug) {
                                printf("%d: Queueing %d via %d\n", int(source),
                                       int(neighbor), int(frontier));
                            }
                            frontiers.add(neighbor);
                        }
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                if constexpr (Config::debug) {
                    printf("%d: Finished queue %d\n", int(source),
                           int(frontier_number));
                }
            }
            frontier_number++;
            frontier_size = frontiers.output_to_input();
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            out_row_sizes[source] = block_list_storage.output_idx;
            source_idx = atomic_add_relaxed(atomics, 1);
        }
        __syncthreads();
        source = source_idx;
    }
}


template <typename Config, typename IndexType>
__global__ void symbolic_factorize_collect_rows(
    const IndexType* row_ptrs, const IndexType* first_block_ids, IndexType size,
    device_block_memory_pool<Config, IndexType> pool, IndexType* output)
{
    constexpr auto block_size = Config::block_size;
    const auto row = thread::get_subwarp_id_flat<block_size>();
    const auto lane = threadIdx.x % block_size;
    if (row >= size) {
        return;
    }
    auto block_id = first_block_ids[row];
    const auto row_begin = row_ptrs[row];
    const auto row_end = row_ptrs[row + 1];
    for (auto out_idx = row_begin + lane; out_idx < row_end;
         out_idx += block_size, block_id = pool.get_next_block(block_id)) {
        const auto block_data = pool.get_block(block_id).data;
        output[out_idx] = block_data[lane];
    }
}


template <typename IndexType>
void symbolic_factorize_general(std::shared_ptr<const DefaultExecutor> exec,
                                const IndexType* row_ptrs,
                                const IndexType* col_idxs, size_type size,
                                IndexType* out_row_ptrs,
                                array<IndexType>& out_col_idxs)
{
    using Config = symbolic_factorize_config;
    const auto num_blocks = 100;
    array<IndexType> max_id_array{exec, size * num_blocks};
    array<IndexType> fill_array{exec, size * num_blocks};
    array<IndexType> frontier_array{exec, size * num_blocks};
    array<IndexType> new_frontier_array{exec, size * num_blocks};
    array<IndexType> output_block_ids{exec, size};
    array<IndexType> atomic_array{exec, 1};
    components::fill_array(exec, fill_array.get_data(), size * num_blocks,
                           IndexType{});
    // TODO proper initial value for allocation
    block_memory_pool<Config, IndexType> pool{
        exec, static_cast<IndexType>(size * 32000 / Config::block_size)};
    components::fill_array(exec, atomic_array.get_data(),
                           atomic_array.get_size(), IndexType{});
    constexpr auto threadblock_size = 1024;
    symbolic_factorize_gsofa<Config>
        <<<num_blocks, threadblock_size, 0, exec->get_stream()>>>(
            row_ptrs, col_idxs, static_cast<IndexType>(size),
            atomic_array.get_data(), max_id_array.get_data(),
            fill_array.get_data(), frontier_array.get_data(),
            new_frontier_array.get_data(), out_row_ptrs,
            output_block_ids.get_data(), pool.device_view());
    components::prefix_sum_nonnegative(exec, out_row_ptrs, size + 1);
    const auto nnz = exec->copy_val_to_host(out_row_ptrs + size);
    out_col_idxs.resize_and_reset(nnz);
    symbolic_factorize_collect_rows<Config>
        <<<ceildiv(size, threadblock_size / Config::block_size),
           threadblock_size>>>(out_row_ptrs, output_block_ids.get_data(),
                               static_cast<IndexType>(size), pool.device_view(),
                               out_col_idxs.get_data());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_LU_SYMBOLIC_FACTORIZE_GENERAL);


}  // namespace lu_factorization
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
