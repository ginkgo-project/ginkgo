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


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {


namespace kernel {
namespace detail {


template <bool IsSorted>
struct find_helper {
    template <typename ForwardIt, typename IndexType>
    static inline bool find(ForwardIt first, ForwardIt last, IndexType value)
    {
        return std::find(first, last, value) != last;
    }
};


template <>
struct find_helper<true> {
    template <typename ForwardIt, typename IndexType>
    static inline bool find(ForwardIt first, ForwardIt last, IndexType value)
    {
        return std::binary_search(first, last, value);
    }
};


}  // namespace detail


template <bool IsSorted, typename ValueType, typename IndexType>
void find_missing_diagonal_elements(
    const matrix::Csr<ValueType, IndexType>* mtx,
    IndexType* elements_to_add_per_row, bool* changes_required)
{
    auto num_rows = static_cast<IndexType>(mtx->get_size()[0]);
    auto num_cols = static_cast<IndexType>(mtx->get_size()[1]);
    auto col_idxs = mtx->get_const_col_idxs();
    auto row_ptrs = mtx->get_const_row_ptrs();
    bool local_change{false};
#pragma omp parallel for reduction(|| : local_change)
    for (IndexType row = 0; row < num_rows; ++row) {
        if (row >= num_cols) {
            elements_to_add_per_row[row] = 0;
            continue;
        }
        const auto* start_cols = col_idxs + row_ptrs[row];
        const auto* end_cols = col_idxs + row_ptrs[row + 1];
        if (detail::find_helper<IsSorted>::find(start_cols, end_cols, row)) {
            elements_to_add_per_row[row] = 0;
        } else {
            elements_to_add_per_row[row] = 1;
            local_change = true;
        }
    }
    *changes_required = local_change;
}


template <typename ValueType, typename IndexType>
void add_missing_diagonal_elements(const matrix::Csr<ValueType, IndexType>* mtx,
                                   ValueType* new_values,
                                   IndexType* new_col_idxs,
                                   const IndexType* row_ptrs_addition)
{
    const auto num_rows = static_cast<IndexType>(mtx->get_size()[0]);
    const auto old_values = mtx->get_const_values();
    const auto old_col_idxs = mtx->get_const_col_idxs();
    const auto row_ptrs = mtx->get_const_row_ptrs();
#pragma omp parallel for
    for (IndexType row = 0; row < num_rows; ++row) {
        const IndexType old_row_start{row_ptrs[row]};
        const IndexType old_row_end{row_ptrs[row + 1]};
        const IndexType new_row_start{old_row_start + row_ptrs_addition[row]};
        const IndexType new_row_end{old_row_end + row_ptrs_addition[row + 1]};

        // if no element needs to be added, do a simple copy
        if (new_row_end - new_row_start == old_row_end - old_row_start) {
            for (IndexType i = 0; i < new_row_end - new_row_start; ++i) {
                const IndexType new_idx = new_row_start + i;
                const IndexType old_idx = old_row_start + i;
                new_values[new_idx] = old_values[old_idx];
                new_col_idxs[new_idx] = old_col_idxs[old_idx];
            }
        } else {
            IndexType new_idx = new_row_start;
            bool diagonal_added{false};
            for (IndexType old_idx = old_row_start; old_idx < old_row_end;
                 ++old_idx) {
                const auto col_idx = old_col_idxs[old_idx];
                if (!diagonal_added && row < col_idx) {
                    new_values[new_idx] = zero<ValueType>();
                    new_col_idxs[new_idx] = row;
                    ++new_idx;
                    diagonal_added = true;
                }
                new_values[new_idx] = old_values[old_idx];
                new_col_idxs[new_idx] = col_idx;
                ++new_idx;
            }
            if (!diagonal_added) {
                new_values[new_idx] = zero<ValueType>();
                new_col_idxs[new_idx] = row;
                diagonal_added = true;
            }
        }
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void add_diagonal_elements(std::shared_ptr<const OmpExecutor> exec,
                           matrix::Csr<ValueType, IndexType>* mtx,
                           bool is_sorted)
{
    auto mtx_size = mtx->get_size();
    size_type row_ptrs_size = mtx_size[0] + 1;
    array<IndexType> row_ptrs_addition{exec, row_ptrs_size};
    bool needs_change{};
    if (is_sorted) {
        kernel::find_missing_diagonal_elements<true>(
            mtx, row_ptrs_addition.get_data(), &needs_change);
    } else {
        kernel::find_missing_diagonal_elements<false>(
            mtx, row_ptrs_addition.get_data(), &needs_change);
    }
    if (!needs_change) {
        return;
    }

    row_ptrs_addition.get_data()[row_ptrs_size - 1] = 0;
    components::prefix_sum_nonnegative(exec, row_ptrs_addition.get_data(),
                                       row_ptrs_size);

    size_type new_num_elems = mtx->get_num_stored_elements() +
                              row_ptrs_addition.get_data()[row_ptrs_size - 1];
    array<ValueType> new_values{exec, new_num_elems};
    array<IndexType> new_col_idxs{exec, new_num_elems};
    kernel::add_missing_diagonal_elements(mtx, new_values.get_data(),
                                          new_col_idxs.get_data(),
                                          row_ptrs_addition.get_const_data());

    auto old_row_ptrs_ptr = mtx->get_row_ptrs();
    auto row_ptrs_addition_ptr = row_ptrs_addition.get_const_data();
#pragma omp parallel for
    for (IndexType i = 0; i < row_ptrs_size; ++i) {
        old_row_ptrs_ptr[i] += row_ptrs_addition_ptr[i];
    }

    matrix::CsrBuilder<ValueType, IndexType> mtx_builder{mtx};
    mtx_builder.get_value_array() = std::move(new_values);
    mtx_builder.get_col_idx_array() = std::move(new_col_idxs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_ADD_DIAGONAL_ELEMENTS_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_row_ptrs_l_u(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    IndexType* l_row_ptrs, IndexType* u_row_ptrs)
{
    auto num_rows = system_matrix->get_size()[0];
    auto row_ptrs = system_matrix->get_const_row_ptrs();
    auto col_idxs = system_matrix->get_const_col_idxs();

// Calculate the NNZ per row first
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        size_type l_nnz{};
        size_type u_nnz{};
        for (size_type el = row_ptrs[row]; el < row_ptrs[row + 1]; ++el) {
            size_type col = col_idxs[el];
            // don't count diagonal
            l_nnz += col < row;
            u_nnz += col > row;
        }
        // add diagonal again
        l_row_ptrs[row] = l_nnz + 1;
        u_row_ptrs[row] = u_nnz + 1;
    }

    // Now, compute the prefix-sum, to get proper row_ptrs for L and U
    components::prefix_sum_nonnegative(exec, l_row_ptrs, num_rows + 1);
    components::prefix_sum_nonnegative(exec, u_row_ptrs, num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_U_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_l_u(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* system_matrix,
                    matrix::Csr<ValueType, IndexType>* csr_l,
                    matrix::Csr<ValueType, IndexType>* csr_u)
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

#pragma omp parallel for
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
                // save value for later
                diag_val = val;
            } else {  // col > row
                col_idxs_u[current_index_u] = col;
                vals_u[current_index_u] = val;
                ++current_index_u;
            }
        }
        // store diagonal entries
        size_type l_diag_idx = row_ptrs_l[row + 1] - 1;
        size_type u_diag_idx = row_ptrs_u[row];
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
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    IndexType* l_row_ptrs)
{
    auto num_rows = system_matrix->get_size()[0];
    auto row_ptrs = system_matrix->get_const_row_ptrs();
    auto col_idxs = system_matrix->get_const_col_idxs();

// Calculate the NNZ per row first
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        size_type l_nnz{};
        for (size_type el = row_ptrs[row]; el < row_ptrs[row + 1]; ++el) {
            size_type col = col_idxs[el];
            // skip diagonal
            l_nnz += col < row;
        }
        // add diagonal again
        l_row_ptrs[row] = l_nnz + 1;
    }

    // Now, compute the prefix-sum, to get proper row_ptrs for L
    components::prefix_sum_nonnegative(exec, l_row_ptrs, num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_l(std::shared_ptr<const OmpExecutor> exec,
                  const matrix::Csr<ValueType, IndexType>* system_matrix,
                  matrix::Csr<ValueType, IndexType>* csr_l, bool diag_sqrt)
{
    const auto row_ptrs = system_matrix->get_const_row_ptrs();
    const auto col_idxs = system_matrix->get_const_col_idxs();
    const auto vals = system_matrix->get_const_values();

    const auto row_ptrs_l = csr_l->get_const_row_ptrs();
    auto col_idxs_l = csr_l->get_col_idxs();
    auto vals_l = csr_l->get_values();

#pragma omp parallel for
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
                // save value for later
                diag_val = val;
            }
        }
        // store diagonal entries
        size_type l_diag_idx = row_ptrs_l[row + 1] - 1;
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
}  // namespace omp
}  // namespace kernels
}  // namespace gko
