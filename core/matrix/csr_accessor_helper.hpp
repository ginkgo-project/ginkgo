// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_CSR_ACCESSOR_HELPER_HPP_
#define GKO_CORE_MATRIX_CSR_ACCESSOR_HELPER_HPP_


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "accessor/index_span.hpp"
#include "accessor/reduced_row_major.hpp"
#include "core/base/utils.hpp"


namespace gko {
namespace acc {
namespace helper {


template <typename ArthType, typename IndexType, typename ValueType>
auto build_rrm_accessor(matrix::view::dense<ValueType> input)
{
    using accessor =
        gko::acc::reduced_row_major<2, ArthType, ValueType, IndexType>;
    GKO_ASSERT(fits_index_type<IndexType>(input.size[0] * input.stride));
    return range<accessor>(
        typename accessor::dim_type{{static_cast<IndexType>(input.size[0]),
                                     static_cast<IndexType>(input.size[1])}},
        input.values,
        typename accessor::storage_stride_type{
            {static_cast<IndexType>(input.stride)}});
}

template <typename ArthType, typename IndexType, typename ValueType>
auto build_rrm_accessor(matrix::view::dense<ValueType> input,
                        index_span column_span)
{
    using accessor =
        gko::acc::reduced_row_major<2, ArthType, ValueType, IndexType>;
    assert(column_span.is_valid());
    GKO_ASSERT(fits_index_type<IndexType>(input.size[0] * input.stride));
    return range<accessor>(
        typename accessor::dim_type{
            {static_cast<IndexType>(input.size[0]),
             static_cast<IndexType>(column_span.end - column_span.begin)}},
        input.values + column_span.begin,
        typename accessor::storage_stride_type{
            {static_cast<IndexType>(input.stride)}});
}


// use a different name for const to allow the non-const to create const
// accessor
template <typename ArthType, typename IndexType, typename ValueType>
auto build_const_rrm_accessor(matrix::view::dense<const ValueType> input)
{
    using accessor =
        gko::acc::reduced_row_major<2, ArthType, const ValueType, IndexType>;
    GKO_ASSERT(fits_index_type<IndexType>(input.size[0] * input.stride));
    return range<accessor>(
        typename accessor::dim_type{{static_cast<IndexType>(input.size[0]),
                                     static_cast<IndexType>(input.size[1])}},
        input.values,
        typename accessor::storage_stride_type{
            {static_cast<IndexType>(input.stride)}});
}

template <typename ArthType, typename IndexType, typename ValueType>
auto build_const_rrm_accessor(matrix::view::dense<const ValueType> input,
                              index_span column_span)
{
    using accessor =
        gko::acc::reduced_row_major<2, ArthType, const ValueType, IndexType>;
    assert(column_span.is_valid());
    GKO_ASSERT(fits_index_type<IndexType>(input.size[0] * input.stride));
    return range<accessor>(
        typename accessor::dim_type{
            {static_cast<IndexType>(input.size[0]),
             static_cast<IndexType>(column_span.end - column_span.begin)}},
        input.values + column_span.begin,
        typename accessor::storage_stride_type{
            {static_cast<IndexType>(input.stride)}});
}


template <typename ArthType, typename ValueType, typename IndexType>
auto build_rrm_accessor(matrix::Csr<ValueType, IndexType>* input)
{
    using accessor =
        gko::acc::reduced_row_major<1, ArthType, ValueType, IndexType>;
    return gko::acc::range<accessor>(
        typename accessor::dim_type{
            {static_cast<IndexType>(input->get_num_stored_elements())}},
        input->get_values());
}


template <typename ArthType, typename ValueType, typename IndexType>
auto build_const_rrm_accessor(const matrix::Csr<ValueType, IndexType>* input)
{
    using accessor =
        gko::acc::reduced_row_major<1, ArthType, const ValueType, IndexType>;
    return gko::acc::range<accessor>(
        typename accessor::dim_type{
            {static_cast<IndexType>(input->get_num_stored_elements())}},
        input->get_const_values());
}


}  // namespace helper
}  // namespace acc
}  // namespace gko


#endif  // GKO_CORE_MATRIX_CSR_ACCESSOR_HELPER_HPP_
