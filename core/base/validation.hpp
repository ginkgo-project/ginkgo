// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_VALIDATION_HPP_
#define GKO_CORE_BASE_VALIDATION_HPP_


#include <cmath>
#include <unordered_set>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>


namespace gko {
namespace validation {


#define GKO_VALIDATE(_expression, _message)                                 \
    if (!(_expression)) {                                                   \
        throw gko::InvalidData(__FILE__, __LINE__, typeid(decltype(*this)), \
                               _message " (" #_expression ")");             \
    }


template <typename IndexType>
void is_sorted(const gko::array<IndexType>& idxs_array)
{
    const auto host_idxs_array = idxs_array.copy_to_host();
    for (size_t i = 0; i + 1 < host_idxs_array.size(); ++i) {
        if (host_idxs_array[i] > host_idxs_array[i + 1]) {
            throw gko::InvalidData(
                __FILE__, __LINE__, typeid(gko::array<IndexType>),
                "array not sorted at index " + std::to_string(i));
        }
    }
}


template <typename IndexType>
void is_within_bounds(const gko::array<IndexType>& idxs_array,
                      const IndexType upper_bound)
{
    const auto host_idxs_array = idxs_array.copy_to_host();
    auto min_pos = 0;
    auto max_pos = 0;

    for (size_t i = 1; i < host_idxs_array.size(); ++i) {
        if (host_idxs_array[i] < host_idxs_array[min_pos]) {
            min_pos = i;
        }
        if (host_idxs_array[i] > host_idxs_array[max_pos]) {
            max_pos = i;
        }
    }
    if (host_idxs_array[min_pos] < 0) {
        throw gko::InvalidData(
            __FILE__, __LINE__, typeid(gko::array<IndexType>),
            "Index " + std::to_string(min_pos) + " is lower than lower bound.");
    }
    if (host_idxs_array[max_pos] >= upper_bound) {
        throw gko::InvalidData(__FILE__, __LINE__,
                               typeid(gko::array<IndexType>),
                               "Index " + std::to_string(min_pos) +
                                   " is larger than upper bound.");
    } else if (host_idxs_array[max_pos] == upper_bound) {
        throw gko::InvalidData(
            __FILE__, __LINE__, typeid(gko::array<IndexType>),
            "Index " + std::to_string(min_pos) + " is equal to upper bound.");
    }
}


template <typename ValueType>
void assert_array_is_finite(const gko::array<ValueType>& values)
{
    const auto host_values = values.copy_to_host();
    for (size_t i = 0; i < host_values.size(); ++i) {
        if (!gko::is_finite(host_values[i])) {
            throw gko::InvalidData(
                __FILE__, __LINE__, typeid(gko::array<ValueType>),
                "matrix contains infinite value at index " + std::to_string(i));
        }
    }
}


template <typename IndexType>
void has_unique_idxs(const gko::array<IndexType>& row_ptrs,
                     const gko::array<IndexType>& col_idxs)
{
    const auto host_row_ptrs = row_ptrs.copy_to_host();
    const auto host_col_idxs = col_idxs.copy_to_host();

    const auto num_rows_ = host_row_ptrs.size() - 1;
    // bool result = true;

    for (IndexType row = 0; row < num_rows_; row++) {
        const auto begin = host_row_ptrs[row];
        const auto end = host_row_ptrs[row + 1];
        const auto size = end - begin;
        std::unordered_set<IndexType> unique_ptrs(host_col_idxs.begin() + begin,
                                                  host_col_idxs.begin() + end);

        if (unique_ptrs.size() < size) {
            throw gko::InvalidData(__FILE__, __LINE__,
                                   typeid(gko::array<IndexType>),
                                   "Column index array has duplicated index.");
        }
    }
}

}  // namespace validation
}  // namespace gko


#endif  // GKO_CORE_BASE_UTILS_HPP_
