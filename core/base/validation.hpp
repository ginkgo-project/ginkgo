// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_VALIDATION_HPP_
#define GKO_CORE_BASE_VALIDATION_HPP_


#include <cmath>
#include <string>
#include <unordered_set>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


namespace gko {
namespace validation {


#define GKO_VALIDATE(_expression, _message)                       \
    {                                                             \
        auto result = (_expression);                              \
        if (!result.is_valid) {                                   \
            throw gko::InvalidData(                               \
                __FILE__, __LINE__, typeid(decltype(*this)),      \
                "Exception occurs: " + result.exception_message + \
                    " [" _message "](" #_expression ")");         \
        }                                                         \
    }


struct validation_result {
    bool is_valid;
    std::string exception_message;

    explicit operator bool() const noexcept { return is_valid; }
};


template <typename IndexType>
validation_result is_sorted(const gko::array<IndexType>& idxs_array)
{
    const auto host_idxs_array = idxs_array.copy_to_host();
    for (size_type i = 0; i + 1 < host_idxs_array.size(); ++i) {
        if (host_idxs_array[i] > host_idxs_array[i + 1]) {
            return {false, "index: " + std::to_string(i)};
        }
    }
    return {true, ""};
}


template <typename IndexType>
validation_result is_within_nonegative_bounds(
    const gko::array<IndexType>& idxs_array, const IndexType upper_bound)
{
    const auto host_idxs_array = idxs_array.copy_to_host();
    size_type min_pos = 0;
    size_type max_pos = 0;

    for (size_type i = 1; i < host_idxs_array.size(); ++i) {
        if (host_idxs_array[i] < host_idxs_array[min_pos]) {
            min_pos = i;
        }
        if (host_idxs_array[i] > host_idxs_array[max_pos]) {
            max_pos = i;
        }
    }
    if (host_idxs_array[min_pos] < 0) {
        return {false,
                "The minimum " + std::to_string(host_idxs_array[min_pos]) +
                    " at index " + std::to_string(min_pos) + " is less than 0"};
    }
    if (host_idxs_array[max_pos] >= upper_bound) {
        return {false, "The maximum " +
                           std::to_string(host_idxs_array[max_pos]) +
                           " at index " + std::to_string(max_pos) +
                           " is greater than or equal to the upper bound " +
                           std::to_string(upper_bound)};
    }

    return {true, ""};
}


template <typename ValueType>
validation_result sparse_matrix_values_are_finite(
    const gko::array<ValueType>& values)
{
    const auto host_values = values.copy_to_host();
    for (size_type i = 0; i < host_values.size(); ++i) {
        if (!is_finite(host_values[i])) {
            return {false, "index: " + std::to_string(i)};
        }
    }
    return {true, ""};
}


template <typename IndexType>
validation_result has_unique_idxs_in_row(const gko::array<IndexType>& row_ptrs,
                                         const gko::array<IndexType>& col_idxs)
{
    const auto host_row_ptrs = row_ptrs.copy_to_host();
    const auto host_col_idxs = col_idxs.copy_to_host();

    const auto num_rows = host_row_ptrs.size() - 1;

    if (host_row_ptrs.size() == 0) {
        return {true, ""};
    }

    for (IndexType row = 0; row < num_rows; row++) {
        const auto begin = host_row_ptrs[row];
        const auto end = host_row_ptrs[row + 1];
        const auto size = end - begin;
        std::unordered_set<IndexType> unique_ptrs(host_col_idxs.begin() + begin,
                                                  host_col_idxs.begin() + end);

        if (unique_ptrs.size() < size) {
            return {false, "row: " + std::to_string(row)};
        }
    }
    return {true, ""};
}


template <typename ValueType, typename IndexType>
validation_result is_triangular_system_matrix(std::shared_ptr<const LinOp> mtx,
                                              bool lower)
{
    using Mtx = matrix::Csr<ValueType, IndexType>;

    auto exec = mtx->get_executor();
    auto master = exec->get_master();
    auto host_mtx = gko::copy_and_convert_to<Mtx>(master, mtx);
    const auto mtx_dim = host_mtx->get_size()[0];
    const auto row_ptrs = host_mtx->get_const_row_ptrs();
    const auto col_idxs = host_mtx->get_const_col_idxs();
    const auto values = host_mtx->get_const_values();

    for (size_type row = 0; row < mtx_dim; row++) {
        bool diagonal_found = false;

        for (size_type j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
            const auto col = col_idxs[j];
            const auto val = values[j];

            if (col == row) {
                if (gko::is_zero(val)) {
                    return {false, "zero diagonal."};
                }
                diagonal_found = true;
            } else if (lower && col > row) {
                return {false, "Not lower triangular."};
            } else if (!lower && col < row) {
                return {false, "Not upper triangular."};
            }
        }
        if (!diagonal_found) {
            return {false, "Missing diagonal."};
        }
    }
    return {true, ""};
}


template <typename ValueType, typename IndexType>
validation_result has_all_non_zero_diagonal(
    const matrix::Csr<ValueType, IndexType>* mtx)
{
    const auto exec = mtx->get_executor();
    const auto master = exec->get_master();
    auto host_mtx = make_temporary_clone(master, mtx);
    const auto row_ptrs = host_mtx->get_const_row_ptrs();
    const auto col_idxs = host_mtx->get_const_col_idxs();
    const auto values = host_mtx->get_const_values();
    const auto num_rows = host_mtx->get_size()[0];

    for (size_type row = 0; row < num_rows; row++) {
        bool diagonal_found = false;
        for (size_type j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
            if (col_idxs[j] == row) {
                if (gko::is_zero(values[j])) {
                    return {false,
                            "zero diagonal at row " + std::to_string(row)};
                }
                diagonal_found = true;
                break;
            }
        }
        if (!diagonal_found) {
            return {false, "Missing diagonal at row " + std::to_string(row)};
        }
    }
    return {true, ""};
}


template <typename Pointer>
validation_result not_nullptr(const Pointer& ptr)
{
    if (!ptr) {
        return {false, "pointer must not be null"};
    }
    return {true, ""};
}


}  // namespace validation
}  // namespace gko


#endif  // GKO_CORE_BASE_UTILS_HPP_
