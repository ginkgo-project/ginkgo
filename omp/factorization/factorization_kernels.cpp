// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/factorization_kernels.hpp"

#include <algorithm>
#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/allocator.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "omp/factorization/factorization_helpers.hpp"


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
    helpers::initialize_l_u(
        system_matrix, csr_l, csr_u,
        helpers::triangular_mtx_closure([](auto) { return one<ValueType>(); },
                                        helpers::identity{}),
        helpers::triangular_mtx_closure(helpers::identity{},
                                        helpers::identity{}));
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

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_L_KERNEL);


template <typename IndexType>
bool symbolic_validate_impl(std::shared_ptr<const DefaultExecutor> exec,
                            const IndexType* row_ptrs, const IndexType* cols,
                            const IndexType* factor_row_ptrs,
                            const IndexType* factor_cols, IndexType size)
{
    unordered_set<IndexType> columns(exec);
    bool valid = true;
#pragma omp parallel for firstprivate(columns) reduction(&& : valid)
    for (IndexType row = 0; row < size; row++) {
        const auto in_begin = cols + row_ptrs[row];
        const auto in_end = cols + row_ptrs[row + 1];
        const auto factor_begin = factor_cols + factor_row_ptrs[row];
        const auto factor_end = factor_cols + factor_row_ptrs[row + 1];
        if (!valid) {
            continue;
        }
        columns.clear();
        // the factor needs to contain the original matrix
        // plus the diagonal if that was missing
        columns.insert(in_begin, in_end);
        columns.insert(row);
        for (auto col_it = factor_begin; col_it < factor_end; ++col_it) {
            const auto col = *col_it;
            if (col >= row) {
                break;
            }
            const auto dep_begin = factor_cols + factor_row_ptrs[col];
            const auto dep_end = factor_cols + factor_row_ptrs[col + 1];
            // insert the upper triangular part of the row
            const auto dep_diag = std::find(dep_begin, dep_end, col);
            columns.insert(dep_diag, dep_end);
        }
        // the factor should contain exactly these columns, no more
        if (factor_end - factor_begin != columns.size()) {
            valid = false;
        }
        for (auto col_it = factor_begin; col_it < factor_end; ++col_it) {
            if (columns.find(*col_it) == columns.end()) {
                valid = false;
            }
        }
    }
    return valid;
}

template <typename ValueType, typename IndexType>
void symbolic_validate(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    const matrix::Csr<ValueType, IndexType>* factors,
    const matrix::csr::lookup_data<IndexType>& factors_lookup, bool& valid)
{
    valid = symbolic_validate_impl(
        exec, system_matrix->get_const_row_ptrs(),
        system_matrix->get_const_col_idxs(), factors->get_const_row_ptrs(),
        factors->get_const_col_idxs(),
        static_cast<IndexType>(system_matrix->get_size()[0]));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_SYMBOLIC_VALIDATE_KERNEL);


}  // namespace factorization
}  // namespace omp
}  // namespace kernels
}  // namespace gko
