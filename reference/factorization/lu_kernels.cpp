// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/lu_kernels.hpp"


#include <algorithm>
#include <memory>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/allocator.hpp"
#include "core/matrix/csr_lookup.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The LU namespace.
 *
 * @ingroup factor
 */
namespace lu_factorization {


template <typename ValueType, typename IndexType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Csr<ValueType, IndexType>* mtx,
                const IndexType* factor_lookup_offsets,
                const int64* factor_lookup_descs,
                const int32* factor_lookup_storage, IndexType* diag_idxs,
                matrix::Csr<ValueType, IndexType>* factors)
{
    const auto num_rows = mtx->get_size()[0];
    const auto mtx_row_ptrs = mtx->get_const_row_ptrs();
    const auto factor_row_ptrs = factors->get_const_row_ptrs();
    const auto mtx_cols = mtx->get_const_col_idxs();
    const auto factor_cols = factors->get_const_col_idxs();
    const auto mtx_vals = mtx->get_const_values();
    const auto factor_vals = factors->get_values();
    for (size_type row = 0; row < num_rows; row++) {
        const auto factor_begin = factor_row_ptrs[row];
        const auto factor_end = factor_row_ptrs[row + 1];
        const auto mtx_begin = mtx_row_ptrs[row];
        const auto mtx_end = mtx_row_ptrs[row + 1];
        std::fill(factor_vals + factor_begin, factor_vals + factor_end,
                  zero<ValueType>());
        matrix::csr::device_sparsity_lookup<IndexType> lookup{
            factor_row_ptrs,       factor_cols,         factor_lookup_offsets,
            factor_lookup_storage, factor_lookup_descs, row};
        for (auto nz = mtx_row_ptrs[row]; nz < mtx_row_ptrs[row + 1]; nz++) {
            const auto col = mtx_cols[nz];
            factor_vals[lookup.lookup_unsafe(col) + factor_begin] =
                mtx_vals[nz];
        }
        diag_idxs[row] = lookup.lookup_unsafe(row) + factor_begin;
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
    const auto row_ptrs = factors->get_const_row_ptrs();
    const auto cols = factors->get_const_col_idxs();
    const auto vals = factors->get_values();
    for (size_type row = 0; row < num_rows; row++) {
        const auto row_begin = row_ptrs[row];
        const auto row_diag = diag_idxs[row];
        matrix::csr::device_sparsity_lookup<IndexType> lookup{
            row_ptrs, cols, lookup_offsets, lookup_storage, lookup_descs, row};
        for (auto lower_nz = row_begin; lower_nz < row_diag; lower_nz++) {
            const auto dep = cols[lower_nz];
            const auto dep_diag_idx = diag_idxs[dep];
            const auto dep_diag = vals[dep_diag_idx];
            const auto dep_end = row_ptrs[dep + 1];
            const auto scale = vals[lower_nz] / dep_diag;
            vals[lower_nz] = scale;
            for (auto dep_nz = dep_diag_idx + 1; dep_nz < dep_end; dep_nz++) {
                const auto col = cols[dep_nz];
                const auto val = vals[dep_nz];
                const auto nz = row_begin + lookup.lookup_unsafe(col);
                vals[nz] -= scale * val;
            }
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
    const auto diag_idxs = diag_idx_array.get_data();
    for (size_type row = 0; row < num_rows; row++) {
        matrix::csr::device_sparsity_lookup<IndexType> lookup{
            factor_row_ptrs, factor_cols,  lookup_offsets,
            lookup_storage,  lookup_descs, row};
        const auto factor_begin = factor_row_ptrs[row];
        const auto factor_end = factor_row_ptrs[row + 1];
        const auto mtx_begin = row_ptrs[row];
        const auto mtx_end = row_ptrs[row + 1];
        // initialize the row
        std::fill(factor_vals + factor_begin, factor_vals + factor_end,
                  zero<float>());
        for (auto nz = mtx_begin; nz < mtx_end; nz++) {
            const auto col = col_idxs[nz];
            factor_vals[lookup.lookup_unsafe(col) + factor_begin] =
                one<float>();
        }
        diag_idxs[row] = lookup.lookup_unsafe(row) + factor_begin;
        const auto row_diag = diag_idxs[row];
        factor_vals[row_diag] = one<float>();
        // apply factorization
        for (auto lower_nz = factor_begin; lower_nz < row_diag; lower_nz++) {
            const auto dep = factor_cols[lower_nz];
            const auto dep_diag_idx = diag_idxs[dep];
            const auto dep_end = factor_row_ptrs[dep + 1];
            if (factor_vals[lower_nz] == one<float>()) {
                for (auto dep_nz = dep_diag_idx + 1; dep_nz < dep_end;
                     dep_nz++) {
                    const auto col = factor_cols[dep_nz];
                    const auto val = factor_vals[dep_nz];
                    const auto nz = factor_begin + lookup.lookup_unsafe(col);
                    if (val == one<float>()) {
                        factor_vals[nz] = one<float>();
                    }
                }
            }
        }
        IndexType row_nnz{};
        for (auto nz = factor_begin; nz < factor_end; nz++) {
            row_nnz += factor_vals[nz] == one<float>() ? 1 : 0;
        }
        out_row_nnz[row] = row_nnz;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_LU_SYMMETRIC_FACTORIZE_SIMPLE);


template <typename IndexType>
void symbolic_factorize_simple_finalize(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<float, IndexType>* factors, IndexType* out_col_idxs)
{
    const auto col_idxs = factors->get_const_col_idxs();
    const auto vals = factors->get_const_values();
    size_type output_idx{};
    // copy all nonzero entries from the symmetric factor to the unsymmetric
    // factor
    for (size_type i = 0; i < factors->get_num_stored_elements(); i++) {
        if (vals[i] == one<float>()) {
            out_col_idxs[output_idx] = col_idxs[i];
            ++output_idx;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_LU_SYMMETRIC_FACTORIZE_SIMPLE_FINALIZE);


}  // namespace lu_factorization
}  // namespace reference
}  // namespace kernels
}  // namespace gko
