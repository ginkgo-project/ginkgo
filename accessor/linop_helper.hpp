/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_ACCESSOR_LINOP_HELPER_HPP_
#define GKO_ACCESSOR_LINOP_HELPER_HPP_


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "index_span.hpp"
#include "reduced_row_major.hpp"
#include "utils.hpp"


namespace gko {
namespace acc {
namespace helper {


template <typename ArthType, typename ValueType>
auto build_accessor(matrix::Dense<ValueType>* input)
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
auto build_accessor(matrix::Dense<ValueType>* input, index_span column_span)
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
auto build_const_accessor(const matrix::Dense<ValueType>* input)
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
auto build_const_accessor(const matrix::Dense<ValueType>* input,
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
auto build_accessor(matrix::Csr<ValueType, IndexType>* input)
{
    using accessor = gko::acc::reduced_row_major<1, ArthType, ValueType>;
    return gko::acc::range<accessor>(
        std::array<acc::size_type, 1>{
            {static_cast<acc::size_type>(input->get_num_stored_elements())}},
        input->get_values());
}


template <typename ArthType, typename ValueType, typename IndexType>
auto build_const_accessor(const matrix::Csr<ValueType, IndexType>* input)
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


#endif  // GKO_ACCESSOR_LINOP_HELPER_HPP_
