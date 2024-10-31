// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/factorization_kernels.hpp"

#include <algorithm>
#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "reference/factorization/factorization_helpers.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {


template <typename IndexType>
size_type count_missing_elements(IndexType num_rows, IndexType num_cols,
                                 const IndexType* col_idxs,
                                 const IndexType* row_ptrs)
{
    size_type missing_elements{};
    // if row >= num_cols, diagonal elements no longer exist
    for (IndexType row = 0; row < num_rows && row < num_cols; ++row) {
        bool was_diagonal_found{false};
        for (IndexType idx = row_ptrs[row]; idx < row_ptrs[row + 1]; ++idx) {
            const auto col = col_idxs[idx];
            if (col == row) {
                was_diagonal_found = true;
                break;
            }
        }
        if (!was_diagonal_found) {
            ++missing_elements;
        }
    }
    return missing_elements;
}


template <typename ValueType, typename IndexType>
void add_diagonal_elements(std::shared_ptr<const ReferenceExecutor> exec,
                           matrix::Csr<ValueType, IndexType>* mtx,
                           bool /*is_sorted*/)
{
    const auto values = mtx->get_const_values();
    const auto col_idxs = mtx->get_const_col_idxs();
    auto row_ptrs = mtx->get_row_ptrs();
    auto num_rows = static_cast<IndexType>(mtx->get_size()[0]);
    auto num_cols = static_cast<IndexType>(mtx->get_size()[1]);

    auto missing_elements =
        count_missing_elements(num_rows, num_cols, col_idxs, row_ptrs);

    if (missing_elements == 0) {
        return;
    }

    const auto old_nnz = mtx->get_num_stored_elements();
    const size_type new_nnz = old_nnz + missing_elements;
    array<ValueType> new_values_array{exec, new_nnz};
    array<IndexType> new_col_idxs_array{exec, new_nnz};
    auto new_values = new_values_array.get_data();
    auto new_col_idxs = new_col_idxs_array.get_data();
    IndexType added_elements{};
    // row_ptrs will be updated in-place

    for (IndexType row = 0; row < num_rows; ++row) {
        bool diagonal_handled{false};
        const IndexType old_row_ptrs_start{row_ptrs[row]};
        const IndexType old_row_ptrs_end{row_ptrs[row + 1]};
        const IndexType new_row_ptrs_start =
            old_row_ptrs_start + added_elements;

        row_ptrs[row] = new_row_ptrs_start;
        for (IndexType old_idx = old_row_ptrs_start; old_idx < old_row_ptrs_end;
             ++old_idx) {
            auto new_idx = old_idx + added_elements;
            const auto col_idx = col_idxs[old_idx];
            if (!diagonal_handled && col_idx > row) {
                const auto start_cols = col_idxs + old_idx;
                const auto end_cols = col_idxs + old_row_ptrs_end;
                // expect row to not be sorted, so search for a diagonal entry
                if (std::find(start_cols, end_cols, row) != end_cols) {
                    // no need to add diagonal since diagonal is already present
                    diagonal_handled = true;
                }
                // if diagonal was not found, add it
                if (!diagonal_handled) {
                    new_values[new_idx] = zero<ValueType>();
                    new_col_idxs[new_idx] = row;
                    ++added_elements;
                    new_idx = old_idx + added_elements;
                    diagonal_handled = true;
                }
            }
            if (row >= num_cols || col_idx == row) {
                diagonal_handled = true;
            }
            new_values[new_idx] = values[old_idx];
            new_col_idxs[new_idx] = col_idx;
        }
        if (row < num_cols && !diagonal_handled) {
            const auto new_idx = old_row_ptrs_end + added_elements;
            new_values[new_idx] = zero<ValueType>();
            new_col_idxs[new_idx] = row;
            diagonal_handled = true;
            ++added_elements;
        }
    }
    row_ptrs[num_rows] = static_cast<IndexType>(new_nnz);

    matrix::CsrBuilder<ValueType, IndexType> mtx_builder{mtx};
    mtx_builder.get_value_array() = std::move(new_values_array);
    mtx_builder.get_col_idx_array() = std::move(new_col_idxs_array);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_WITH_HALF(
    GKO_DECLARE_FACTORIZATION_ADD_DIAGONAL_ELEMENTS_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_row_ptrs_l_u(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    IndexType* l_row_ptrs, IndexType* u_row_ptrs)
{
    auto row_ptrs = system_matrix->get_const_row_ptrs();
    auto col_idxs = system_matrix->get_const_col_idxs();
    IndexType l_nnz{};
    IndexType u_nnz{};

    l_row_ptrs[0] = 0;
    u_row_ptrs[0] = 0;
    for (size_type row = 0; row < system_matrix->get_size()[0]; ++row) {
        for (auto el = row_ptrs[row]; el < row_ptrs[row + 1]; ++el) {
            auto col = col_idxs[el];
            // don't count diagonal
            l_nnz += col < row;
            u_nnz += col > row;
        }
        // add diagonal again
        l_nnz++;
        u_nnz++;
        l_row_ptrs[row + 1] = l_nnz;
        u_row_ptrs[row + 1] = u_nnz;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_WITH_HALF(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_U_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_l_u(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* system_matrix,
                    matrix::Csr<ValueType, IndexType>* csr_l,
                    matrix::Csr<ValueType, IndexType>* csr_u)
{
    helpers::initialize_l_u(
        system_matrix, csr_l, csr_u,
        helpers::triangular_mtx_closure([](auto) { return one<ValueType>(); },
                                        helpers::identity{}),
        helpers::triangular_mtx_closure(helpers::identity{},
                                        helpers::identity{}));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_WITH_HALF(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_L_U_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_row_ptrs_l(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    IndexType* l_row_ptrs)
{
    auto row_ptrs = system_matrix->get_const_row_ptrs();
    auto col_idxs = system_matrix->get_const_col_idxs();
    size_type l_nnz{};

    l_row_ptrs[0] = 0;
    for (size_type row = 0; row < system_matrix->get_size()[0]; ++row) {
        for (size_type el = row_ptrs[row]; el < row_ptrs[row + 1]; ++el) {
            size_type col = col_idxs[el];
            // skip diagonal
            l_nnz += col < row;
        }
        // add diagonal again
        l_nnz++;
        l_row_ptrs[row + 1] = l_nnz;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_WITH_HALF(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_l(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Csr<ValueType, IndexType>* system_matrix,
                  matrix::Csr<ValueType, IndexType>* csr_l, bool diag_sqrt)
{
    helpers::initialize_l(system_matrix, csr_l,
                          helpers::triangular_mtx_closure(
                              [diag_sqrt](auto val) {
                                  if (diag_sqrt) {
                                      val = sqrt(val);
                                      if (!is_finite(val)) {
                                          val = one<ValueType>();
                                      }
                                  }
                                  return val;
                              },
                              helpers::identity{}));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_WITH_HALF(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_L_KERNEL);


}  // namespace factorization
}  // namespace reference
}  // namespace kernels
}  // namespace gko
