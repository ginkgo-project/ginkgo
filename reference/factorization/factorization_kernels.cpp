/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/factorization/factorization_kernels.hpp"


#include <algorithm>
#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/components/prefix_sum.hpp"
#include "core/matrix/csr_builder.hpp"


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
                                 const IndexType *col_idxs,
                                 const IndexType *row_ptrs)
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
                           matrix::Csr<ValueType, IndexType> *mtx,
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
    Array<ValueType> new_values_array{exec, new_nnz};
    Array<IndexType> new_col_idxs_array{exec, new_nnz};
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
    row_ptrs[num_rows] = new_nnz;

    matrix::CsrBuilder<ValueType, IndexType> mtx_builder{mtx};
    mtx_builder.get_value_array() = std::move(new_values_array);
    mtx_builder.get_col_idx_array() = std::move(new_col_idxs_array);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_ADD_DIAGONAL_ELEMENTS_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_row_ptrs_l_u(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Csr<ValueType, IndexType> *system_matrix,
    IndexType *l_row_ptrs, IndexType *u_row_ptrs)
{
    auto row_ptrs = system_matrix->get_const_row_ptrs();
    auto col_idxs = system_matrix->get_const_col_idxs();
    size_type l_nnz{};
    size_type u_nnz{};

    l_row_ptrs[0] = 0;
    u_row_ptrs[0] = 0;
    for (size_type row = 0; row < system_matrix->get_size()[0]; ++row) {
        for (size_type el = row_ptrs[row]; el < row_ptrs[row + 1]; ++el) {
            size_type col = col_idxs[el];
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

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_U_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_l_u(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *system_matrix,
                    matrix::Csr<ValueType, IndexType> *csr_l,
                    matrix::Csr<ValueType, IndexType> *csr_u)
{
    const auto row_ptrs = system_matrix->get_const_row_ptrs();
    const auto col_idxs = system_matrix->get_const_col_idxs();
    const auto vals = system_matrix->get_const_values();

    const auto row_ptrs_l = csr_l->get_const_row_ptrs();
    auto col_idxs_l = csr_l->get_col_idxs();
    auto vals_l = csr_l->get_values();

    const auto row_ptrs_u = csr_u->get_const_row_ptrs();
    auto col_idxs_u = csr_u->get_col_idxs();
    auto vals_u = csr_u->get_values();

    for (size_type row = 0; row < system_matrix->get_size()[0]; ++row) {
        size_type current_index_l = row_ptrs_l[row];
        size_type current_index_u =
            row_ptrs_u[row] + 1;  // we treat the diagonal separately
        // if there is no diagonal value, set it to 1 by default
        auto diag_val = one<ValueType>();
        for (size_type el = row_ptrs[row]; el < row_ptrs[row + 1]; ++el) {
            const auto col = col_idxs[el];
            const auto val = vals[el];
            if (col < row) {
                col_idxs_l[current_index_l] = col;
                vals_l[current_index_l] = val;
                ++current_index_l;
            } else if (col == row) {
                // save diagonal value
                diag_val = val;
            } else {  // col > row
                col_idxs_u[current_index_u] = col;
                vals_u[current_index_u] = val;
                ++current_index_u;
            }
        }
        // store diagonal values separately
        auto l_diag_idx = row_ptrs_l[row + 1] - 1;
        auto u_diag_idx = row_ptrs_u[row];
        col_idxs_l[l_diag_idx] = row;
        col_idxs_u[u_diag_idx] = row;
        vals_l[l_diag_idx] = one<ValueType>();
        vals_u[u_diag_idx] = diag_val;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_L_U_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_row_ptrs_l(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Csr<ValueType, IndexType> *system_matrix,
    IndexType *l_row_ptrs)
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

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_l(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Csr<ValueType, IndexType> *system_matrix,
                  matrix::Csr<ValueType, IndexType> *csr_l, bool diag_sqrt)
{
    const auto row_ptrs = system_matrix->get_const_row_ptrs();
    const auto col_idxs = system_matrix->get_const_col_idxs();
    const auto vals = system_matrix->get_const_values();

    const auto row_ptrs_l = csr_l->get_const_row_ptrs();
    auto col_idxs_l = csr_l->get_col_idxs();
    auto vals_l = csr_l->get_values();

    for (size_type row = 0; row < system_matrix->get_size()[0]; ++row) {
        size_type current_index_l = row_ptrs_l[row];
        // if there is no diagonal value, set it to 1 by default
        auto diag_val = one<ValueType>();
        for (size_type el = row_ptrs[row]; el < row_ptrs[row + 1]; ++el) {
            const auto col = col_idxs[el];
            const auto val = vals[el];
            if (col < row) {
                col_idxs_l[current_index_l] = col;
                vals_l[current_index_l] = val;
                ++current_index_l;
            } else if (col == row) {
                // save diagonal value
                diag_val = val;
            }
        }
        // store diagonal values separately
        auto l_diag_idx = row_ptrs_l[row + 1] - 1;
        col_idxs_l[l_diag_idx] = row;
        // compute square root with sentinel
        if (diag_sqrt) {
            diag_val = sqrt(diag_val);
            if (!is_finite(diag_val)) {
                diag_val = one<ValueType>();
            }
        }
        vals_l[l_diag_idx] = diag_val;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_L_KERNEL);


}  // namespace factorization
}  // namespace reference
}  // namespace kernels
}  // namespace gko
