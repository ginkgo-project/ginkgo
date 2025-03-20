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
                               _message);                                   \
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
    const auto host_row_ptrs =
        make_temporary_clone(row_ptrs.get_executor()->get_master(), &row_ptrs);
    return std::is_sorted(
        host_row_ptrs->get_const_data(),
        host_row_ptrs->get_const_data() + host_row_ptrs->get_size());
}


template <typename IndexType>
bool is_within_bounds(const gko::array<IndexType>& col_idxs,
                      const std::int64_t& lower_bound,
                      const size_t& upper_bound)
{
    const auto host_col_idxs =
        make_temporary_clone(col_idxs.get_executor()->get_master(), &col_idxs);
    const auto [min, max] = std::minmax_element(
        host_col_idxs->get_const_data(),
        host_col_idxs->get_const_data() + host_col_idxs->get_size());
    return *min >= lower_bound && *max < upper_bound;
}


template <typename IndexType>
bool ell_is_within_bounds(const gko::array<IndexType>& col_idxs,
                          const std::int64_t& lower_bound,
                          const size_t& upper_bound)
{
    const auto host_col_idxs =
        make_temporary_clone(col_idxs.get_executor()->get_master(), &col_idxs);
    const auto [min, max] = std::minmax_element(
        host_col_idxs->get_const_data(),
        host_col_idxs->get_const_data() + host_col_idxs->get_size());
    return (*min >= lower_bound || *min == -1) &&
           *max < upper_bound;  //-1 for padding? Probably.
}


template <typename ValueType>
bool is_finite(const gko::array<ValueType>& values)
{
    const auto host_values =
        make_temporary_clone(values.get_executor()->get_master(), &values);

    for (size_t i = 0; i < host_values->get_size(); ++i) {
        if (!is_finite_scalar(host_values->get_const_data()[i])) {
            return false;
        }
    }

    return true;
}


template <typename IndexType>
bool has_unique_ptrs(const gko::array<IndexType>& row_ptrs,
                     const gko::array<IndexType>& col_idxs)
{
    const auto host_row_ptrs =
        make_temporary_clone(row_ptrs.get_executor()->get_master(), &row_ptrs);
    const auto host_col_idxs =
        make_temporary_clone(col_idxs.get_executor()->get_master(), &col_idxs);

    const auto num_rows_ = host_row_ptrs->get_size() - 1;
    const auto row_ptrs_ = host_row_ptrs->get_const_data();
    const auto col_idxs_ = host_col_idxs->get_const_data();

    bool result = true;

    for (IndexType row = 0; row < num_rows_; row++) {
        const auto begin = row_ptrs_[row];
        const auto end = row_ptrs_[row + 1];
        const auto size = end - begin;
        std::unordered_set<IndexType> unique_ptrs(col_idxs_ + begin,
                                                  col_idxs_ + end);

        result = result && unique_ptrs.size() == size;
    }
    return result;
}

// template <typename ValueType, typename IndexType>
// bool has_non_zero_diagonal(const gko::array<IndexType>& row_ptrs,
//                            const gko::array<IndexType> col_idxs,
//                            const gko::array<ValueType>& values)
// {
//     const auto host_row_ptrs =
//         make_temporary_clone(row_ptrs.get_executor()->get_master(),
//         &row_ptrs);
//     const auto host_col_idxs =
//         make_temporary_clone(col_idxs.get_executor()->get_master(),
//         &col_idxs);
//     const auto host_values =
//         make_temporary_clone(values.get_executor()->get_master(), &values);

//     const auto num_rows = host_row_ptrs->get_size() - 1;
//     const auto row_ptrs = host_row_ptrs->get_const_data();
//     const auto col_idxs = host_col_idxs->get_const_data();
//     const auto values = host_values->get_const_data();

//     if(row_ptrs->get_size() != col_idxs->get_size())
//     {
//         return false;
//     }

//     if (values->get_size() > num_rows * (num_rows -1))  //pigeonhole
//     principle
//     {
//         return true;
//     }

//     for (IndexType i = 0; i < num_rows; i++)
//     {
//         if (row_ptrs[i] == row_ptrs[i + 1])
//         {
//             return false;
//         }

//     }

//     return ;
// }

}  // namespace validation
}  // namespace gko


#endif  // GKO_CORE_BASE_UTILS_HPP_
