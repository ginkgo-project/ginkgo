// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/ilu_kernels.hpp"

#include <ginkgo/core/base/array.hpp>

#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/sparselib_bindings.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "common/cuda_hip/components/syncfree.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "core/matrix/csr_lookup.hpp"

namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace ilu_factorization {


template <typename ValueType, typename IndexType>
void sparselib_ilu(std::shared_ptr<const DefaultExecutor> exec,
                   matrix::Csr<ValueType, IndexType>* m)
{
    const auto id = exec->get_device_id();
    auto handle = exec->get_sparselib_handle();
    auto desc = sparselib::create_mat_descr();
    auto info = sparselib::create_ilu0_info();

    // get buffer size for ILU
    IndexType num_rows = m->get_size()[0];
    IndexType nnz = m->get_num_stored_elements();
    size_type buffer_size{};
    sparselib::ilu0_buffer_size(handle, num_rows, nnz, desc,
                                m->get_const_values(), m->get_const_row_ptrs(),
                                m->get_const_col_idxs(), info, buffer_size);

    array<char> buffer{exec, buffer_size};

    // set up ILU(0)
    sparselib::ilu0_analysis(handle, num_rows, nnz, desc, m->get_const_values(),
                             m->get_const_row_ptrs(), m->get_const_col_idxs(),
                             info, SPARSELIB_SOLVE_POLICY_USE_LEVEL,
                             buffer.get_data());

    sparselib::ilu0(handle, num_rows, nnz, desc, m->get_values(),
                    m->get_const_row_ptrs(), m->get_const_col_idxs(), info,
                    SPARSELIB_SOLVE_POLICY_USE_LEVEL, buffer.get_data());

    // CUDA 11.4 has a use-after-free bug on Turing
#if defined(GKO_COMPILING_CUDA) && (CUDA_VERSION >= 11040)
    exec->synchronize();
#endif

    sparselib::destroy_ilu0_info(info);
    sparselib::destroy(desc);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ILU_SPARSELIB_ILU_KERNEL);


constexpr static int default_block_size = 512;

namespace kernel {


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void factorize_on_both(
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ cols,
    const IndexType* __restrict__ storage_offsets,
    const int32* __restrict__ storage, const int64* __restrict__ row_descs,
    const IndexType* __restrict__ diag_idxs, ValueType* __restrict__ vals,
    const IndexType* __restrict__ matrix_row_ptrs,
    const IndexType* __restrict__ matrix_cols,
    const IndexType* __restrict__ matrix_storage_offsets,
    const int32* __restrict__ matrix_storage,
    const int64* __restrict__ matrix_row_descs,
    ValueType* __restrict__ matrix_vals, syncfree_storage dep_storage,
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
    const auto row_begin = row_ptrs[row];
    const auto row_diag = diag_idxs[row];
    const auto row_end = row_ptrs[row + 1];
    gko::matrix::csr::device_sparsity_lookup<IndexType> lookup{
        row_ptrs, cols,      storage_offsets,
        storage,  row_descs, static_cast<size_type>(row)};
    gko::matrix::csr::device_sparsity_lookup<IndexType> matrix_lookup{
        matrix_row_ptrs, matrix_cols,      matrix_storage_offsets,
        matrix_storage,  matrix_row_descs, static_cast<size_type>(row)};
    auto factor_nz = row_begin;
    const auto matrix_row_begin = matrix_row_ptrs[row];
    auto matrix_nz = matrix_row_begin;
    const auto matrix_row_diag = matrix_lookup.lookup_unsafe(row) + matrix_nz;
    // for each lower triangular entry: eliminate with corresponding row
    while (matrix_nz < matrix_row_diag || factor_nz < row_diag) {
        auto dep_matrix = matrix_nz < matrix_row_diag
                              ? matrix_cols[matrix_nz]
                              : device_numeric_limits<IndexType>::max();
        auto dep_factor = factor_nz < row_diag
                              ? cols[factor_nz]
                              : device_numeric_limits<IndexType>::max();
        auto dep = min(dep_matrix, dep_factor);
        // we can load the value before synchronizing because the following
        // updates only go past the diagonal of the dependency row, i.e. at
        // least column dep + 1
        const auto val =
            (dep == dep_factor) ? vals[factor_nz] : matrix_vals[matrix_nz];
        const auto diag_idx = diag_idxs[dep];
        const auto dep_end = row_ptrs[dep + 1];
        scheduler.wait(dep);
        const auto diag = vals[diag_idx];
        const auto scale = val / diag;
        if (lane == 0) {
            vals[factor_nz] = scale;
        }
        // subtract all entries past the diagonal
        // we only need to consider the entries in the factor not entire
        // one.
        for (auto upper_nz = diag_idx + 1 + lane; upper_nz < dep_end;
             upper_nz += config::warp_size) {
            const auto upper_col = cols[upper_nz];
            const auto upper_val = vals[upper_nz];

            const auto idx = lookup[upper_col];
            if (idx != invalid_index<IndexType>()) {
                vals[row_begin + idx] -= scale * upper_val;
            }
            // but we still need to operate on the matrix because we drop
            // the entries after row operation need to keep the track here.
            const auto matrix_idx = matrix_lookup[upper_col];
            if (matrix_idx != invalid_index<IndexType>()) {
                matrix_vals[matrix_row_begin + matrix_idx] -= scale * val;
            }
        }
        matrix_nz += (dep == dep_matrix);
        factor_nz += (dep == dep_factor);
    }
    scheduler.mark_ready();
}

}  // namespace kernel

template <typename ValueType, typename IndexType>
void factorize_on_both(std::shared_ptr<const DefaultExecutor> exec,
                       const IndexType* lookup_offsets,
                       const int64* lookup_descs, const int32* lookup_storage,
                       const IndexType* diag_idxs,
                       matrix::Csr<ValueType, IndexType>* factors,
                       const IndexType* matrix_lookup_offsets,
                       const int64* matrix_lookup_descs,
                       const int32* matrix_lookup_storage,
                       matrix::Csr<ValueType, IndexType>* matrix,
                       array<int>& tmp_storage)
{
    const auto num_rows = factors->get_size()[0];
    if (num_rows > 0) {
        syncfree_storage storage(exec, tmp_storage, num_rows);
        const auto num_blocks =
            ceildiv(num_rows, default_block_size / config::warp_size);
        kernel::factorize_on_both<<<num_blocks, default_block_size, 0,
                                    exec->get_stream()>>>(
            factors->get_const_row_ptrs(), factors->get_const_col_idxs(),
            lookup_offsets, lookup_storage, lookup_descs, diag_idxs,
            as_device_type(factors->get_values()), matrix->get_const_row_ptrs(),
            matrix->get_const_col_idxs(), matrix_lookup_offsets,
            matrix_lookup_storage, matrix_lookup_descs,
            as_device_type(matrix->get_values()), storage, num_rows);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ILU_FACTORIZE_ON_BOTH_KERNEL);


}  // namespace ilu_factorization
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
