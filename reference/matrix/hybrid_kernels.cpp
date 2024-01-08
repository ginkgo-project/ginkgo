// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/hybrid_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>


#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/ell_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Hybrid matrix format namespace.
 * @ref Hybrid
 * @ingroup hybrid
 */
namespace hybrid {


void compute_coo_row_ptrs(std::shared_ptr<const DefaultExecutor> exec,
                          const array<size_type>& row_nnz, size_type ell_lim,
                          int64* coo_row_ptrs)
{
    for (size_type row = 0; row < row_nnz.get_size(); row++) {
        if (row_nnz.get_const_data()[row] <= ell_lim) {
            coo_row_ptrs[row] = 0;
        } else {
            coo_row_ptrs[row] = row_nnz.get_const_data()[row] - ell_lim;
        }
    }
    components::prefix_sum_nonnegative(exec, coo_row_ptrs,
                                       row_nnz.get_size() + 1);
}


void compute_row_nnz(std::shared_ptr<const DefaultExecutor> exec,
                     const array<int64>& row_ptrs, size_type* row_nnzs)
{
    for (size_type i = 0; i < row_ptrs.get_size() - 1; i++) {
        row_nnzs[i] =
            row_ptrs.get_const_data()[i + 1] - row_ptrs.get_const_data()[i];
    }
}


template <typename ValueType, typename IndexType>
void fill_in_matrix_data(std::shared_ptr<const DefaultExecutor> exec,
                         const device_matrix_data<ValueType, IndexType>& data,
                         const int64* row_ptrs, const int64*,
                         matrix::Hybrid<ValueType, IndexType>* result)
{
    const auto num_rows = result->get_size()[0];
    const auto ell_max_nnz = result->get_ell_num_stored_elements_per_row();
    const auto values = data.get_const_values();
    const auto row_idxs = data.get_const_row_idxs();
    const auto col_idxs = data.get_const_col_idxs();
    size_type coo_nz{};
    for (size_type row = 0; row < num_rows; row++) {
        size_type ell_nz{};
        for (auto nz = row_ptrs[row]; nz < row_ptrs[row + 1]; nz++) {
            if (ell_nz < ell_max_nnz) {
                result->ell_col_at(row, ell_nz) = col_idxs[nz];
                result->ell_val_at(row, ell_nz) = values[nz];
                ell_nz++;
            } else {
                result->get_coo_row_idxs()[coo_nz] = row_idxs[nz];
                result->get_coo_col_idxs()[coo_nz] = col_idxs[nz];
                result->get_coo_values()[coo_nz] = values[nz];
                coo_nz++;
            }
        }
        for (; ell_nz < ell_max_nnz; ell_nz++) {
            result->ell_col_at(row, ell_nz) = invalid_index<IndexType>();
            result->ell_val_at(row, ell_nz) = zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYBRID_FILL_IN_MATRIX_DATA_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Hybrid<ValueType, IndexType>* source,
                    const IndexType*, const IndexType*,
                    matrix::Csr<ValueType, IndexType>* result)
{
    auto csr_val = result->get_values();
    auto csr_col_idxs = result->get_col_idxs();
    auto csr_row_ptrs = result->get_row_ptrs();
    const auto ell = source->get_ell();
    const auto max_nnz_per_row = ell->get_num_stored_elements_per_row();
    const auto coo_val = source->get_const_coo_values();
    const auto coo_col = source->get_const_coo_col_idxs();
    const auto coo_row = source->get_const_coo_row_idxs();
    const auto coo_nnz = source->get_coo_num_stored_elements();
    csr_row_ptrs[0] = 0;
    size_type csr_idx = 0;
    size_type coo_idx = 0;
    for (IndexType row = 0; row < source->get_size()[0]; row++) {
        // Ell part
        for (IndexType i = 0; i < max_nnz_per_row; i++) {
            const auto val = ell->val_at(row, i);
            const auto col = ell->col_at(row, i);
            if (col != invalid_index<IndexType>()) {
                csr_val[csr_idx] = val;
                csr_col_idxs[csr_idx] = col;
                csr_idx++;
            }
        }
        // Coo part (row should be ascending)
        while (coo_idx < coo_nnz && coo_row[coo_idx] == row) {
            csr_val[csr_idx] = coo_val[coo_idx];
            csr_col_idxs[csr_idx] = coo_col[coo_idx];
            csr_idx++;
            coo_idx++;
        }
        csr_row_ptrs[row + 1] = csr_idx;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYBRID_CONVERT_TO_CSR_KERNEL);


}  // namespace hybrid
}  // namespace reference
}  // namespace kernels
}  // namespace gko
