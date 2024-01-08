// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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


template <typename ValueType, typename IndexType>
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
            const auto output_pos = lookup.lookup_unsafe(upper_col) + row_begin;
            vals[output_pos] -= scale * upper_val;
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
            as_device_type(factors->get_values()), storage, num_rows);
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
