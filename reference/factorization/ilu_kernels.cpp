// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/ilu_kernels.hpp"

#include <algorithm>

#include <ginkgo/core/base/math.hpp>

#include "core/base/allocator.hpp"
#include "core/matrix/csr_lookup.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The ilu factorization namespace.
 *
 * @ingroup factor
 */
namespace ilu_factorization {


template <typename ValueType, typename IndexType>
void sparselib_ilu(std::shared_ptr<const DefaultExecutor> exec,
                   matrix::Csr<ValueType, IndexType>* m)
{
    vector<IndexType> diagonals{m->get_size()[0], -1, exec};
    const auto row_ptrs = m->get_const_row_ptrs();
    const auto col_idxs = m->get_const_col_idxs();
    const auto values = m->get_values();
    for (IndexType row = 0; row < static_cast<IndexType>(m->get_size()[0]);
         row++) {
        const auto begin = row_ptrs[row];
        const auto end = row_ptrs[row + 1];
        for (auto nz = begin; nz < end; nz++) {
            const auto col = col_idxs[nz];
            if (col == row) {
                diagonals[row] = nz;
            }
            auto value = values[nz];
            for (auto l_nz = begin; l_nz < end; l_nz++) {
                // for each lower triangular entry l_ik
                const auto l_col = col_idxs[l_nz];
                if (l_col >= min(row, col)) {
                    continue;
                }
                // find corresponding entry u_kj
                const auto u_begin_it = col_idxs + row_ptrs[l_col];
                const auto u_end_it = col_idxs + row_ptrs[l_col + 1];
                const auto u_it = std::lower_bound(u_begin_it, u_end_it, col);
                const auto u_nz = std::distance(col_idxs, u_it);
                if (u_it != u_end_it && *u_it == col) {
                    value -= values[l_nz] * values[u_nz];
                }
            }
            if (row <= col) {
                values[nz] = value;
            } else {
                GKO_ASSERT(diagonals[col] != -1);
                values[nz] = value / values[diagonals[col]];
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ILU_SPARSELIB_ILU_KERNEL);


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
    const auto row_ptrs = factors->get_const_row_ptrs();
    const auto cols = factors->get_const_col_idxs();
    const auto vals = factors->get_values();
    for (size_type row = 0; row < num_rows; row++) {
        const auto row_begin = row_ptrs[row];
        const auto row_diag = diag_idxs[row];
        matrix::csr::device_sparsity_lookup<IndexType> lookup{
            row_ptrs, cols, lookup_offsets, lookup_storage, lookup_descs, row};
        matrix::csr::device_sparsity_lookup<IndexType> matrix_lookup{
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            matrix_lookup_offsets,        matrix_lookup_storage,
            matrix_lookup_descs,          row};
        auto factor_nz = row_begin;
        const auto matrix_row_begin = matrix->get_const_row_ptrs()[row];
        auto matrix_nz = matrix_row_begin;
        const auto matrix_row_diag =
            matrix_lookup.lookup_unsafe(row) + matrix_nz;
        while (matrix_nz < matrix_row_diag || factor_nz < row_diag) {
            auto dep_matrix = matrix_nz < matrix_row_diag
                                  ? matrix->get_const_col_idxs()[matrix_nz]
                                  : std::numeric_limits<IndexType>::max();
            auto dep_factor = factor_nz < row_diag
                                  ? cols[factor_nz]
                                  : std::numeric_limits<IndexType>::max();
            auto dep = min(dep_matrix, dep_factor);
            const auto dep_diag_idx = diag_idxs[dep];
            const auto dep_diag = vals[dep_diag_idx];
            const auto dep_end = row_ptrs[dep + 1];
            const auto scale =
                ((dep == dep_factor) ? vals[factor_nz]
                                     : matrix->get_const_values()[matrix_nz]) /
                dep_diag;
            if (dep == dep_factor) {
                vals[factor_nz] = scale;
            }
            if (dep == dep_matrix) {
                matrix->get_values()[matrix_nz] = scale;
            }
            // we only need to consider the entries in the factor not entire
            // one.
            for (auto dep_nz = dep_diag_idx + 1; dep_nz < dep_end; dep_nz++) {
                const auto col = cols[dep_nz];
                const auto val = vals[dep_nz];
                const auto idx = lookup[col];
                if (idx != invalid_index<IndexType>()) {
                    vals[row_begin + idx] -= scale * val;
                }
                // but we still need to operate on the matrix because we drop
                // the entries after row operation need to keep the track here.
                const auto matrix_idx = matrix_lookup[col];
                if (matrix_idx != invalid_index<IndexType>()) {
                    matrix->get_values()[matrix_row_begin + matrix_idx] -=
                        scale * val;
                }
            }
            matrix_nz += (dep == dep_matrix);
            factor_nz += (dep == dep_factor);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ILU_FACTORIZE_ON_BOTH_KERNEL);


}  // namespace ilu_factorization
}  // namespace reference
}  // namespace kernels
}  // namespace gko
