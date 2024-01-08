// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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
namespace acc {
namespace helper {


template <typename ArthType, typename ValueType>
auto build_rrm_accessor(matrix::Dense<ValueType>* input)
{
    using accessor = gko::acc::reduced_row_major<2, ArthType, ValueType>;
    return range<accessor>(
        std::array<acc::size_type, 2>{
            {static_cast<acc::size_type>(input->get_size()[0]),
             static_cast<acc::size_type>(input->get_size()[1])}},
        input->get_values(),
        std::array<acc::size_type, 1>{
            {static_cast<acc::size_type>(input->get_stride())}});
}

template <typename ArthType, typename ValueType>
auto build_rrm_accessor(matrix::Dense<ValueType>* input, index_span column_span)
{
    using accessor = gko::acc::reduced_row_major<2, ArthType, ValueType>;
    assert(column_span.is_valid());
    return range<accessor>(
        std::array<acc::size_type, 2>{
            {static_cast<acc::size_type>(input->get_size()[0]),
             static_cast<acc::size_type>(column_span.end - column_span.begin)}},
        input->get_values() + column_span.begin,
        std::array<acc::size_type, 1>{
            {static_cast<acc::size_type>(input->get_stride())}});
}


// use a different name for const to allow the non-const to create const
// accessor
template <typename ArthType, typename ValueType>
auto build_const_rrm_accessor(const matrix::Dense<ValueType>* input)
{
    using accessor = gko::acc::reduced_row_major<2, ArthType, const ValueType>;
    return range<accessor>(
        std::array<acc::size_type, 2>{
            {static_cast<acc::size_type>(input->get_size()[0]),
             static_cast<acc::size_type>(input->get_size()[1])}},
        input->get_const_values(),
        std::array<acc::size_type, 1>{
            {static_cast<acc::size_type>(input->get_stride())}});
}

template <typename ArthType, typename ValueType>
auto build_const_rrm_accessor(const matrix::Dense<ValueType>* input,
                              index_span column_span)
{
    using accessor = gko::acc::reduced_row_major<2, ArthType, const ValueType>;
    assert(column_span.is_valid());
    return range<accessor>(
        std::array<acc::size_type, 2>{
            {static_cast<acc::size_type>(input->get_size()[0]),
             static_cast<acc::size_type>(column_span.end - column_span.begin)}},
        input->get_const_values() + column_span.begin,
        std::array<acc::size_type, 1>{
            {static_cast<acc::size_type>(input->get_stride())}});
}


template <typename ArthType, typename ValueType, typename IndexType>
auto build_rrm_accessor(matrix::Csr<ValueType, IndexType>* input)
{
    using accessor = gko::acc::reduced_row_major<1, ArthType, ValueType>;
    return gko::acc::range<accessor>(
        std::array<acc::size_type, 1>{
            {static_cast<acc::size_type>(input->get_num_stored_elements())}},
        input->get_values());
}


template <typename ArthType, typename ValueType, typename IndexType>
auto build_const_rrm_accessor(const matrix::Csr<ValueType, IndexType>* input)
{
    using accessor = gko::acc::reduced_row_major<1, ArthType, const ValueType>;
    return gko::acc::range<accessor>(
        std::array<acc::size_type, 1>{
            {static_cast<acc::size_type>(input->get_num_stored_elements())}},
        input->get_const_values());
}


}  // namespace helper
}  // namespace acc
}  // namespace gko


#endif  // GKO_CORE_MATRIX_CSR_ACCESSOR_HELPER_HPP_
