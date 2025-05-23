// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
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
        auto mask = group::ballot(subwarp, pred);
        while (mask) {
            if (pred) {
                const auto out_nz = out_base + popcnt(mask & prefix_mask);
                out_cols[out_nz] = postorder[node];
                node = postorder_parent[node];
                pred = node < next_node;
            }
            out_base += popcnt(mask);
            mask = group::ballot(subwarp, pred);
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
