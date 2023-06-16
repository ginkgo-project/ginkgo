/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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
        auto val = vals[lower_nz];
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
