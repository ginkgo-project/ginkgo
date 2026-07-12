// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_CSR_ACCESSOR_HELPER_HPP_
#define GKO_CORE_MATRIX_CSR_ACCESSOR_HELPER_HPP_


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "accessor/index_span.hpp"
#include "accessor/reduced_row_major.hpp"
#include "accessor/utils.hpp"


namespace gko {
namespace acc_helper {


template <typename ArthType, typename ValueType>
auto build_rrm_accessor(matrix::view::dense<ValueType> input)
{
    using accessor = acc::reduced_row_major<2, ArthType, ValueType>;
    return acc::range<accessor>(
        std::array<acc::size_type, 2>{
            {static_cast<acc::size_type>(input.size[0]),
             static_cast<acc::size_type>(input.size[1])}},
        input.values,
        std::array<acc::size_type, 1>{
            {static_cast<acc::size_type>(input.stride)}});
}

template <typename ArthType, typename ValueType>
auto build_rrm_accessor(matrix::view::dense<ValueType> input,
                        acc::index_span column_span)
{
    using accessor = acc::reduced_row_major<2, ArthType, ValueType>;
    assert(column_span.is_valid());
    return acc::range<accessor>(
        std::array<acc::size_type, 2>{
            {static_cast<acc::size_type>(input.size[0]),
             static_cast<acc::size_type>(column_span.end - column_span.begin)}},
        input.values + column_span.begin,
        std::array<acc::size_type, 1>{
            {static_cast<acc::size_type>(input.stride)}});
}


// use a different name for const to allow the non-const to create const
// accessor
template <typename ArthType, typename ValueType>
auto build_const_rrm_accessor(matrix::view::dense<const ValueType> input)
{
    using accessor = acc::reduced_row_major<2, ArthType, const ValueType>;
    return acc::range<accessor>(
        std::array<acc::size_type, 2>{
            {static_cast<acc::size_type>(input.size[0]),
             static_cast<acc::size_type>(input.size[1])}},
        input.values,
        std::array<acc::size_type, 1>{
            {static_cast<acc::size_type>(input.stride)}});
}

template <typename ArthType, typename ValueType>
auto build_const_rrm_accessor(matrix::view::dense<const ValueType> input,
                              acc::index_span column_span)
{
    using accessor = acc::reduced_row_major<2, ArthType, const ValueType>;
    assert(column_span.is_valid());
    return acc::range<accessor>(
        std::array<acc::size_type, 2>{
            {static_cast<acc::size_type>(input.size[0]),
             static_cast<acc::size_type>(column_span.end - column_span.begin)}},
        input.values + column_span.begin,
        std::array<acc::size_type, 1>{
            {static_cast<acc::size_type>(input.stride)}});
}


template <typename ArthType, typename ValueType, typename IndexType>
auto build_rrm_accessor(matrix::Csr<ValueType, IndexType>* input)
{
    using accessor = acc::reduced_row_major<1, ArthType, ValueType, IndexType>;
    return acc::range<accessor>(
        std::array<acc::size_type, 1>{
            {static_cast<acc::size_type>(input->get_num_stored_elements())}},
        input->get_values());
}


template <typename ArthType, typename ValueType, typename IndexType>
auto build_const_rrm_accessor(const matrix::Csr<ValueType, IndexType>* input)
{
    using accessor =
        acc::reduced_row_major<1, ArthType, const ValueType, IndexType>;
    return acc::range<accessor>(
        std::array<acc::size_type, 1>{
            {static_cast<acc::size_type>(input->get_num_stored_elements())}},
        input->get_const_values());
}


}  // namespace acc_helper
}  // namespace gko


#endif  // GKO_CORE_MATRIX_CSR_ACCESSOR_HELPER_HPP_
