// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/elimination_forest.hpp"
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


template <typename ValueType, typename IndexType>
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
                const auto output_pos =
                    lookup.lookup_unsafe(upper_col) + row_begin;
                vals[output_pos] -= scale * upper_val;
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
               matrix::Csr<ValueType, IndexType>* factors,
               array<int>& tmp_storage)
{
    const auto num_rows = factors->get_size()[0];
    if (num_rows > 0) {
        syncfree_storage storage(exec, tmp_storage, num_rows);
        const auto num_blocks =
            ceildiv(num_rows, default_block_size / config::warp_size);
        kernel::factorize<<<num_blocks, default_block_size, 0,
                            exec->get_stream()>>>(
            factors->get_const_row_ptrs(), factors->get_const_col_idxs(),
            lookup_offsets, lookup_storage, lookup_descs, diag_idxs,
            transpose_idxs, as_device_type(factors->get_values()), storage,
            num_rows);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CHOLESKY_FACTORIZE);
