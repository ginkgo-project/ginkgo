// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/csr.hpp>

#include "core/factorization/factorization_helpers.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace factorization {
namespace helpers {


using namespace ::gko::factorization;


template <typename ValueType, typename IndexType, typename LClosure,
          typename UClosure>
void initialize_l_u(const matrix::Csr<ValueType, IndexType>* system_matrix,
                    matrix::Csr<ValueType, IndexType>* csr_l,
                    matrix::Csr<ValueType, IndexType>* csr_u,
                    LClosure&& l_closure, UClosure&& u_closure)
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
                vals_l[current_index_l] = l_closure.map_off_diag(val);
                ++current_index_l;
            } else if (col == row) {
                // save diagonal value
                diag_val = val;
            } else {  // col > row
                col_idxs_u[current_index_u] = col;
                vals_u[current_index_u] = u_closure.map_off_diag(val);
                ++current_index_u;
            }
        }
        // store diagonal values separately
        auto l_diag_idx = row_ptrs_l[row + 1] - 1;
        auto u_diag_idx = row_ptrs_u[row];
        col_idxs_l[l_diag_idx] = row;
        col_idxs_u[u_diag_idx] = row;
        vals_l[l_diag_idx] = l_closure.map_diag(diag_val);
        vals_u[u_diag_idx] = u_closure.map_diag(diag_val);
    }
}


template <typename ValueType, typename IndexType, typename Closure>
void initialize_l(const matrix::Csr<ValueType, IndexType>* system_matrix,
                  matrix::Csr<ValueType, IndexType>* csr_l, Closure&& closure)
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
                vals_l[current_index_l] = closure.map_off_diag(val);
                ++current_index_l;
            } else if (col == row) {
                // save diagonal value
                diag_val = val;
            }
        }
        // store diagonal values separately
        auto l_diag_idx = row_ptrs_l[row + 1] - 1;
        col_idxs_l[l_diag_idx] = row;
        vals_l[l_diag_idx] = closure.map_diag(diag_val);
    }
}


}  // namespace helpers
}  // namespace factorization
}  // namespace reference
}  // namespace kernels
}  // namespace gko
