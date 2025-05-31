// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_VALIDATION_HPP_
#define GKO_CORE_BASE_VALIDATION_HPP_


#include <cmath>
#include <unordered_set>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>


namespace gko {
namespace validation {


#define GKO_VALIDATE(_expression, _message)                                 \
    if (!(_expression)) {                                                   \
        throw gko::InvalidData(__FILE__, __LINE__, typeid(decltype(*this)), \
                               _message " (" #_expression ")");             \
    }


template <typename ValueType>
bool is_finite_scalar(const ValueType& value)
{
    return std::isfinite(static_cast<double>(value));
}

template <typename ValueType>
bool is_finite_scalar(const std::complex<ValueType>& value)
{
    return is_finite_scalar(value.real()) && is_finite_scalar(value.imag());
}


template <typename IndexType>
bool is_sorted(const gko::array<IndexType>& row_ptrs)
{
    const auto host_row_ptrs = row_ptrs.copy_to_host();
    return std::is_sorted(host_row_ptrs.begin(), host_row_ptrs.end());
}


template <typename IndexType>
bool is_within_bounds(const gko::array<IndexType>& col_idxs,
                      const IndexType upper_bound)
{
    const auto host_col_idxs = col_idxs.copy_to_host();
    const auto [min, max] =
        std::minmax_element(host_col_idxs.begin(), host_col_idxs.end());

    return *min >= 0 && *max < upper_bound;
}


template <typename IndexType>
bool ell_has_unique_idxs(const gko::array<IndexType>& col_idxs,
                         const IndexType num_rows, const size_type nnz,
                         const size_type stride)
{
    const auto host_col_idxs = col_idxs.copy_to_host();
    for (size_type i = 0; i < num_rows; ++i) {
        std::unordered_set<IndexType> unique_idxs;
        for (size_type j = 0; j < nnz; ++j) {
            const auto idx = host_col_idxs[i + stride * j];
            if (idx == -1) {
                continue;
            }
            if (idx != -1 && idx < num_rows) {
                if (!unique_idxs.insert(idx).second) {
                    return false;
                }
            }
        }
    }
    return true;
}


template <typename IndexType>
bool ell_is_within_bounds(const gko::array<IndexType>& col_idxs,
                          const IndexType num_rows, const size_type nnz,
                          const size_type stride)
{
    const auto host_col_idxs = col_idxs.copy_to_host();
    for (size_type i = 0; i < num_rows; ++i) {
        bool padding = false;
        for (size_type j = 0; j < nnz; ++j) {
            const auto idx = host_col_idxs[i + stride * j];
            if (idx == -1) {
                padding = true;
            } else if (padding) {
                return false;
            }
        }
    }
    return true;
}


template <typename ValueType>
bool is_finite(const gko::array<ValueType>& values)
{
    const auto host_values = values.copy_to_host();
    for (size_t i = 0; i < host_values.size(); ++i) {
        if (!is_finite_scalar(host_values[i])) {
            return false;
        }
    }
    return true;
}


template <typename IndexType>
bool has_unique_columns(const gko::array<IndexType>& row_ptrs,
                        const gko::array<IndexType>& col_idxs)
{
    const auto host_row_ptrs = row_ptrs.copy_to_host();
    const auto host_col_idxs = col_idxs.copy_to_host();

    const auto num_rows_ = host_row_ptrs.size() - 1;
    bool result = true;

    for (IndexType row = 0; row < num_rows_; row++) {
        const auto begin = host_row_ptrs[row];
        const auto end = host_row_ptrs[row + 1];
        const auto size = end - begin;
        std::unordered_set<IndexType> unique_ptrs(host_col_idxs.begin() + begin,
                                                  host_col_idxs.begin() + end);

        if (unique_ptrs.size() < size) {
            return false;
        }
    }
    return result;
}

template <typename IndexType>
bool has_unique_perm_idxs(const gko::array<IndexType>& permutation_)
{
    const auto host_perm_idxs = permutation_.copy_to_host();
    const auto size = host_perm_idxs.size();
    std::unordered_set<IndexType> unique_ptrs(host_perm_idxs.begin(),
                                              host_perm_idxs.end());

    if (unique_ptrs.size() < size) {
        return false;
    }

    for (IndexType i = 0; i < static_cast<IndexType>(size); ++i) {
        if (unique_ptrs.find(i) == unique_ptrs.end()) {
            return false;
        }
    }

    return true;
}

}  // namespace validation
}  // namespace gko


#endif  // GKO_CORE_BASE_UTILS_HPP_
