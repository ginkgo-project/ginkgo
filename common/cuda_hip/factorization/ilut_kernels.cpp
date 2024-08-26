// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/ilut_kernels.hpp"

#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/syncfree.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "core/base/allocator.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"

namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The ILUT factorization namespace.
 *
 * @ingroup factor
 */
namespace ilut_factorization {


constexpr int default_block_size = 512;


namespace kernel {


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void initialize(
    const IndexType* __restrict__ mtx_row_ptrs,
    const IndexType* __restrict__ mtx_cols,
    const ValueType* __restrict__ mtx_vals,
    const IndexType* __restrict__ l_factor_row_ptrs,
    const IndexType* __restrict__ l_factor_cols,
    const IndexType* __restrict__ u_factor_row_ptrs,
    const IndexType* __restrict__ u_factor_cols,
    const IndexType* __restrict__ l_factor_storage_offsets,
    const int32* __restrict__ l_factor_storage,
    const int64* __restrict__ l_factor_row_descs,
    const IndexType* __restrict__ u_factor_storage_offsets,
    const int32* __restrict__ u_factor_storage,
    const int64* __restrict__ u_factor_row_descs,
    ValueType* __restrict__ l_factor_vals,
    ValueType* __restrict__ u_factor_vals, size_type num_rows)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    if (row >= num_rows) {
        return;
    }
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    // first zero out this row of the factors
    const auto l_factor_begin = l_factor_row_ptrs[row];
    const auto l_factor_end = l_factor_row_ptrs[row + 1];
    const auto lane = static_cast<int>(warp.thread_rank());
    for (auto nz = l_factor_begin + lane; nz < l_factor_end;
         nz += config::warp_size) {
        l_factor_vals[nz] = zero<ValueType>();
    }
    const auto u_factor_begin = u_factor_row_ptrs[row];
    const auto u_factor_end = u_factor_row_ptrs[row + 1];
    for (auto nz = u_factor_begin + lane; nz < u_factor_end;
         nz += config::warp_size) {
        u_factor_vals[nz] = zero<ValueType>();
    }
    warp.sync();
    // then fill in the values from mtx
    gko::matrix::csr::device_sparsity_lookup<IndexType> l_lookup{
        l_factor_row_ptrs, l_factor_cols,      l_factor_storage_offsets,
        l_factor_storage,  l_factor_row_descs, row};
    gko::matrix::csr::device_sparsity_lookup<IndexType> u_lookup{
        u_factor_row_ptrs, u_factor_cols,      u_factor_storage_offsets,
        u_factor_storage,  u_factor_row_descs, row};
    const auto row_begin = mtx_row_ptrs[row];
    const auto row_end = mtx_row_ptrs[row + 1];
    for (auto nz = row_begin + lane; nz < row_end; nz += config::warp_size) {
        const auto col = mtx_cols[nz];
        const auto val = mtx_vals[nz];
        if (col > row) {
            u_factor_vals[u_lookup.lookup_unsafe(col) + u_factor_begin] = val;
        } else if (col == row) {
            u_factor_vals[u_lookup.lookup_unsafe(col) + u_factor_begin] = val;
            l_factor_vals[l_lookup.lookup_unsafe(col) + l_factor_begin] =
                one<ValueType>();
        } else {
            l_factor_vals[l_lookup.lookup_unsafe(col) + l_factor_begin] = val;
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void factorize(
    const IndexType* __restrict__ l_row_ptrs,
    const IndexType* __restrict__ l_cols,
    const IndexType* __restrict__ u_row_ptrs,
    const IndexType* __restrict__ u_cols,
    const IndexType* __restrict__ l_storage_offsets,
    const int32* __restrict__ l_storage, const int64* __restrict__ l_row_descs,
    const IndexType* __restrict__ u_storage_offsets,
    const int32* __restrict__ u_storage, const int64* __restrict__ u_row_descs,
    ValueType* __restrict__ l_vals, ValueType* __restrict__ u_vals,
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
    const auto row_begin = l_row_ptrs[row];
    const auto row_end = l_row_ptrs[row + 1];
    gko::matrix::csr::device_sparsity_lookup<IndexType> l_lookup{
        l_row_ptrs, l_cols,      l_storage_offsets,
        l_storage,  l_row_descs, static_cast<size_type>(row)};
    gko::matrix::csr::device_sparsity_lookup<IndexType> u_lookup{
        u_row_ptrs, u_cols,      u_storage_offsets,
        u_storage,  u_row_descs, static_cast<size_type>(row)};
    // for each lower triangular entry: eliminate with corresponding row
    for (auto lower_nz = row_begin; lower_nz < row_end - 1; lower_nz++) {
        const auto dep = l_cols[lower_nz];
        // we can load the value before synchronizing because the following
        // updates only go past the diagonal of the dependency row, i.e. at
        // least column dep + 1
        const auto val = l_vals[lower_nz];
        const auto diag_idx = u_row_ptrs[dep];
        const auto dep_end = u_row_ptrs[dep + 1];
        scheduler.wait(dep);
        const auto diag = u_vals[diag_idx];
        const auto scale = val / diag;
        if (lane == 0) {
            l_vals[lower_nz] = scale;
        }
        // subtract all entries past the diagonal
        const auto u_row_begin = u_row_ptrs[row];
        const auto u_row_end = u_row_ptrs[row + 1];
        for (auto upper_nz = diag_idx + 1 + lane; upper_nz < dep_end;
             upper_nz += config::warp_size) {
            const auto upper_col = u_cols[upper_nz];
            const auto upper_val = u_vals[upper_nz];
            if (upper_col >= row) {
                const auto output_pos =
                    u_lookup.lookup_unsafe(upper_col) + u_row_begin;
                if (output_pos >= u_row_begin && output_pos < u_row_end &&
                    u_cols[output_pos] == upper_col) {
                    u_vals[output_pos] -= scale * upper_val;
                }
            } else {
                const auto output_pos =
                    l_lookup.lookup_unsafe(upper_col) + row_begin;
                if (output_pos >= row_begin && output_pos < row_end - 1 &&
                    l_cols[output_pos] == upper_col) {
                    l_vals[output_pos] -= scale * upper_val;
                }
            }
        }
    }
    scheduler.mark_ready();
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Csr<ValueType, IndexType>* mtx,
                matrix::Csr<ValueType, IndexType>* l_factor,
                const IndexType* l_lookup_offsets, const int64* l_lookup_descs,
                const int32* l_lookup_storage,
                matrix::Csr<ValueType, IndexType>* u_factor,
                const IndexType* u_lookup_offsets, const int64* u_lookup_descs,
                const int32* u_lookup_storage)
{
    const auto num_rows = mtx->get_size()[0];
    if (num_rows > 0) {
        const auto num_blocks =
            ceildiv(num_rows, default_block_size / config::warp_size);
        kernel::initialize<<<num_blocks, default_block_size, 0,
                             exec->get_stream()>>>(
            mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
            as_device_type(mtx->get_const_values()),
            l_factor->get_const_row_ptrs(), l_factor->get_const_col_idxs(),
            u_factor->get_const_row_ptrs(), u_factor->get_const_col_idxs(),
            l_lookup_offsets, l_lookup_storage, l_lookup_descs,
            u_lookup_offsets, u_lookup_storage, u_lookup_descs,
            as_device_type(l_factor->get_values()),
            as_device_type(u_factor->get_values()), num_rows);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ILUT_INITIALIZE_KERNEL);


template <typename ValueType, typename IndexType>
void compute_l_u_factors(std::shared_ptr<const DefaultExecutor> exec,
                         matrix::Csr<ValueType, IndexType>* l,
                         const IndexType* l_lookup_offsets,
                         const int64* l_lookup_descs,
                         const int32* l_lookup_storage,
                         matrix::Csr<ValueType, IndexType>* u,
                         const IndexType* u_lookup_offsets,
                         const int64* u_lookup_descs,
                         const int32* u_lookup_storage, array<int>& tmp_storage)
{
    const auto num_rows = l->get_size()[0];
    if (num_rows > 0) {
        syncfree_storage storage(exec, tmp_storage, num_rows);
        const auto num_blocks =
            ceildiv(num_rows, default_block_size / config::warp_size);
        kernel::factorize<<<num_blocks, default_block_size, 0,
                            exec->get_stream()>>>(
            l->get_const_row_ptrs(), l->get_const_col_idxs(),
            u->get_const_row_ptrs(), u->get_const_col_idxs(), l_lookup_offsets,
            l_lookup_storage, l_lookup_descs, u_lookup_offsets,
            u_lookup_storage, u_lookup_descs, as_device_type(l->get_values()),
            as_device_type(u->get_values()), storage, num_rows);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ILUT_COMPUTE_LU_FACTORS_KERNEL);


}  // namespace ilut_factorization
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
