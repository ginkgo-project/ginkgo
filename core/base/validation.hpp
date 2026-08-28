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
        if (!result.isValid) {                                    \
            throw gko::InvalidData(                               \
                __FILE__, __LINE__, typeid(decltype(*this)),      \
                "Exception occurs: " + result.exception_message + \
                    " [" _message "](" #_expression ")");         \
        }                                                         \
    }


struct ValidationResult {
    bool isValid;
    std::string exception_message;

    explicit operator bool() const noexcept { return isValid; }
};


template <typename IndexType>
ValidationResult is_sorted(const gko::array<IndexType>& idxs_array)
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
ValidationResult is_within_nonegative_bounds(
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
ValidationResult sparse_matrix_values_are_finite(
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
ValidationResult has_unique_idxs_in_row(const gko::array<IndexType>& row_ptrs,
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
void validate_system_matrix(std::shared_ptr<const LinOp> mtx)
{
    if (!mtx) {
        throw InvalidData(__FILE__, __LINE__, typeid(LinOp),
                          "System matrix is null.");
    }
    try {
        typed->validate_data();
        return true;
    } catch (const InvalidData& e) {
        throw InvalidData(__FILE__, __LINE__, typeid(LinOp),
                          "Invalid system matrix. Inner error: " + e.what());
    }
    auto try_validate = [&](auto&& ptr, const char* name) {
        using PtrType = typename std::remove_reference<decltype(ptr)>::type;
        if (auto typed =
                std::dynamic_pointer_cast<const typename PtrType::element_type>(
                    mtx)) {
            try {
                typed->validate_data();
                return true;
            } catch (const InvalidData& e) {
                throw InvalidData(__FILE__, __LINE__, typeid(LinOp),
                                  std::string("Invalid ") + name +
                                      " matrix. Inner error: " + e.what());
            }
        }
        return false;
    };

    if (try_validate(std::shared_ptr<const matrix::Coo<ValueType, IndexType>>{},
                     "Coo") ||
        try_validate(std::shared_ptr<const matrix::Csr<ValueType, IndexType>>{},
                     "Csr") ||
        try_validate(std::shared_ptr<const matrix::Ell<ValueType, IndexType>>{},
                     "Ell") ||
        try_validate(std::shared_ptr<const matrix::Dense<ValueType>>{},
                     "Dense") ||
        try_validate(std::shared_ptr<const matrix::Diagonal<ValueType>>{},
                     "Diagonal") ||
        try_validate(std::shared_ptr<const matrix::Permutation<IndexType>>{},
                     "Permutation")) {
        return;
    }
}


template <typename ValueType, typename IndexType>
ValidationResult is_valid_preconditioner(std::shared_ptr<const LinOp> prec)
{
    return {true, ""};
}


template <typename ValueType, typename IndexType>
ValidationResult is_triangular_system_matrix(std::shared_ptr<const LinOp> mtx)
{
    using Mtx = matrix::Csr<ValueType, IndexType>;

    auto exec = mtx->get_executor();
    auto master = exec->get_master();
    auto host_mtx = gko::copy_and_convert_to<Mtx>(master, mtx);
    const auto mtx_dim = host_mtx->get_size()[0];
    const auto row_ptrs = host_mtx->get_const_row_ptrs();
    const auto col_idxs = host_mtx->get_const_col_idxs();
    const auto values = host_mtx->get_const_values();

    bool is_upper = true;
    bool is_lower = true;

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
            } else if (col > row) {
                is_lower = false;
            } else if (col < row) {
                is_upper = false;
            }
            if (!is_lower && !is_upper) {
                return {false, "Not triangular."};
            }
        }
        if (!diagonal_found) {
            return {false, "Missing diagonal."};
        }
    }
    return {true, ""};
}


template <typename ValueType, typename IndexType>
ValidationResult has_all_non_zero_diagonal(
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


template <typename IndexType>
ValidationResult is_valid_block_pointers(
    const gko::array<IndexType>& block_ptrs, const size_t max_block_size)
{
    const auto host_ptrs = block_ptrs.copy_to_host();
    for (size_t i = 0; i + 1 < host_ptrs.size(); ++i) {
        const auto start = host_ptrs[i];
        const auto end = host_ptrs[i + 1];

        if (end < start) {
            return {false, "index: " + std::to_string(i)};
        }

        const size_type gap = static_cast<size_type>(end - start);
        if (gap > max_block_size) {
            return {false, "index: " + std::to_string(i)};
        }
    }
    return {true, ""};
}


template <typename ValueType>
ValidationResult is_finite_block(const gko::array<ValueType>& blocks)
{
    return sparse_matrix_values_are_finite(blocks);
}


}  // namespace validation
}  // namespace gko


#endif  // GKO_CORE_BASE_UTILS_HPP_
