// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/ilu_kernels.hpp"

#include "core/matrix/csr_lookup.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The ilu factorization namespace.
 *
 * @ingroup factor
 */
namespace ilu_factorization {


template <typename ValueType, typename IndexType>
void sparselib_ilu(std::shared_ptr<const DefaultExecutor> exec,
                   matrix::Csr<ValueType, IndexType>* m) GKO_NOT_IMPLEMENTED;

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
    // TODO parallelize
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
}  // namespace omp
}  // namespace kernels
}  // namespace gko
