// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_CSR_ACCESSOR_HELPER_HPP_
#define GKO_CORE_MATRIX_CSR_ACCESSOR_HELPER_HPP_


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "accessor/index_limit_checks.hpp"
#include "accessor/index_span.hpp"
#include "accessor/reduced_row_major.hpp"


namespace gko {
namespace acc {
namespace helper {


template <typename ArithmeticType, typename IndexType, typename ValueType>
auto build_rrm_accessor(matrix::view::dense<ValueType> input)
{
    using accessor =
        gko::acc::reduced_row_major<2, ArithmeticType, ValueType, IndexType>;
    GKO_ASSERT(dense_accessor_fits<accessor>(input.size[0], input.size[1],
                                             input.stride));
    return range<accessor>(
        typename accessor::dim_type{
            {static_cast<typename accessor::size_type>(input.size[0]),
             static_cast<typename accessor::size_type>(input.size[1])}},
        input.values,
        typename accessor::storage_stride_type{
            {static_cast<typename accessor::size_type>(input.stride)}});
}

template <typename ArithmeticType, typename IndexType, typename ValueType>
auto build_rrm_accessor(matrix::view::dense<ValueType> input,
                        index_span column_span)
{
    using accessor =
        gko::acc::reduced_row_major<2, ArithmeticType, ValueType, IndexType>;
    assert(column_span.is_valid());
    GKO_ASSERT(dense_accessor_fits<accessor>(
        input.size[0], column_span.end - column_span.begin, input.stride));
    return range<accessor>(
        typename accessor::dim_type{
            {static_cast<typename accessor::size_type>(input.size[0]),
             static_cast<typename accessor::size_type>(column_span.end -
                                                       column_span.begin)}},
        input.values + column_span.begin,
        typename accessor::storage_stride_type{
            {static_cast<typename accessor::size_type>(input.stride)}});
}


// use a different name for const to allow the non-const to create const
// accessor
template <typename ArithmeticType, typename IndexType, typename ValueType>
auto build_const_rrm_accessor(matrix::view::dense<const ValueType> input)
{
    using accessor = gko::acc::reduced_row_major<2, ArithmeticType,
                                                 const ValueType, IndexType>;
    GKO_ASSERT(dense_accessor_fits<accessor>(input.size[0], input.size[1],
                                             input.stride));
    return range<accessor>(
        typename accessor::dim_type{
            {static_cast<typename accessor::size_type>(input.size[0]),
             static_cast<typename accessor::size_type>(input.size[1])}},
        input.values,
        typename accessor::storage_stride_type{
            {static_cast<typename accessor::size_type>(input.stride)}});
}

template <typename ArithmeticType, typename IndexType, typename ValueType>
auto build_const_rrm_accessor(matrix::view::dense<const ValueType> input,
                              index_span column_span)
{
    using accessor = gko::acc::reduced_row_major<2, ArithmeticType,
                                                 const ValueType, IndexType>;
    assert(column_span.is_valid());
    GKO_ASSERT(dense_accessor_fits<accessor>(
        input.size[0], column_span.end - column_span.begin, input.stride));
    return range<accessor>(
        typename accessor::dim_type{
            {static_cast<typename accessor::size_type>(input.size[0]),
             static_cast<typename accessor::size_type>(column_span.end -
                                                       column_span.begin)}},
        input.values + column_span.begin,
        typename accessor::storage_stride_type{
            {static_cast<typename accessor::size_type>(input.stride)}});
}


template <typename ArithmeticType, typename ValueType, typename IndexType>
auto build_rrm_accessor(matrix::view::csr<ValueType, IndexType> input)
{
    using accessor =
        gko::acc::reduced_row_major<1, ArithmeticType, ValueType, IndexType>;
    return gko::acc::range<accessor>(
        typename accessor::dim_type{{static_cast<typename accessor::size_type>(
            input.num_stored_elements)}},
        input.values);
}


template <typename ArithmeticType, typename ValueType, typename IndexType>
auto build_const_rrm_accessor(
    matrix::view::csr<const ValueType, const IndexType> input)
{
    using accessor = gko::acc::reduced_row_major<1, ArithmeticType,
                                                 const ValueType, IndexType>;
    return gko::acc::range<accessor>(
        typename accessor::dim_type{{static_cast<typename accessor::size_type>(
            input.num_stored_elements)}},
        input.values);
}


}  // namespace helper
}  // namespace acc
}  // namespace gko


#endif  // GKO_CORE_MATRIX_CSR_ACCESSOR_HELPER_HPP_
