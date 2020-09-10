/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_CORE_SYNTHESIZER_EXPRESSION_OPERATORS_HPP_
#define GKO_CORE_SYNTHESIZER_EXPRESSION_OPERATORS_HPP_

#include <ginkgo/core/matrix/dense.hpp>
#include "expression_builders.hpp"
#include "expression_types.hpp"


#include <type_traits>


namespace gko {
namespace expression {
namespace detail {

// build scalar from value
template <typename ValueType, typename ScalarType>
scalar_expression<ValueType> create_scalar(std::shared_ptr<const Executor> exec,
                                           ScalarType scalar)
{
    return {share(initialize<matrix::Dense<ValueType>>(
        {static_cast<ValueType>(scalar)}, exec))};
}

}  // namespace detail


// A * B = Product
template <typename ValueType>
product_expression<ValueType, linop_expression<ValueType>,
                   linop_expression<ValueType>>
operator*(linop_expression<ValueType> e1, linop_expression<ValueType> e2)
{
    return {std::make_tuple(e1, e2)};
}


// Sum * A = Product
template <typename ValueType, typename... Summands1>
product_expression<ValueType, sum_expression<ValueType, Summands1...>,
                   linop_expression<ValueType>>
operator*(sum_expression<ValueType, Summands1...> e1,
          linop_expression<ValueType> e2)
{
    return {std::make_tuple(e1, e2)};
}


// A * Sum = Product
template <typename ValueType, typename... Summands2>
product_expression<ValueType, linop_expression<ValueType>,
                   sum_expression<ValueType, Summands2...>>
operator*(linop_expression<ValueType> e1,
          sum_expression<ValueType, Summands2...> e2)
{
    return {std::make_tuple(e1, e2)};
}


// Sum * Sum = Product
template <typename ValueType, typename... Summands1, typename... Summands2>
product_expression<ValueType, sum_expression<ValueType, Summands1...>,
                   sum_expression<ValueType, Summands2...>>
operator*(sum_expression<ValueType, Summands1...> e1,
          sum_expression<ValueType, Summands2...> e2)
{
    return {std::make_tuple(e1, e2)};
}


// Sum * Product = Product
template <typename ValueType, typename... Summands1, typename... Factors2>
product_expression<ValueType, sum_expression<ValueType, Summands1...>,
                   Factors2...>
operator*(sum_expression<ValueType, Summands1...> e1,
          product_expression<ValueType, Factors2...> e2)
{
    return {std::tuple_cat(std::make_tuple(e1), e2.factors)};
}


// Product * Sum = Product
template <typename ValueType, typename... Factors1, typename... Summands2>
product_expression<ValueType, Factors1...,
                   sum_expression<ValueType, Summands2...>>
operator*(product_expression<ValueType, Factors1...> e1,
          sum_expression<ValueType, Summands2...> e2)
{
    return {std::tuple_cat(e1.factors, std::make_tuple(e2))};
}


// Product * Product = Product
template <typename ValueType, typename... Factors1, typename... Factors2>
product_expression<ValueType, Factors1..., Factors2...> operator*(
    product_expression<ValueType, Factors1...> e1,
    product_expression<ValueType, Factors2...> e2)
{
    return {std::tuple_cat(e1.factors, e2.factors)};
}


// s * A = sA
template <typename ValueType>
scaled_linop_expression<ValueType> operator*(scalar_expression<ValueType> e1,
                                             linop_expression<ValueType> e2)
{
    return {e1, e2};
}

template <typename ValueType, typename ScalarType,
          typename = std::enable_if_t<
              std::is_convertible<ScalarType, ValueType>::value>>
scaled_linop_expression<ValueType> operator*(ScalarType s,
                                             linop_expression<ValueType> e2)
{
    return {detail::create_scalar<ValueType>(get_executor(e2), s), e2};
}


// A * s = sA
template <typename ValueType>
scaled_linop_expression<ValueType> operator*(linop_expression<ValueType> e1,
                                             scalar_expression<ValueType> e2)
{
    return {e2, e1};
}

template <typename ValueType, typename ScalarType,
          typename = std::enable_if_t<
              std::is_convertible<ScalarType, ValueType>::value>>
scaled_linop_expression<ValueType> operator*(linop_expression<ValueType> e1,
                                             ScalarType s)
{
    return {detail::create_scalar<ValueType>(get_executor(e1), s), e1};
}


// -A = sA
template <typename ValueType>
scaled_linop_expression<ValueType> operator-(linop_expression<ValueType> e1)
{
    return -1 * e1;
}


// s * Sum = sSum
template <typename ValueType, typename... Summands2>
scaled_sum_expression<ValueType, Summands2...> operator*(
    scalar_expression<ValueType> e1, sum_expression<ValueType, Summands2...> e2)
{
    return {e1, e2};
}

template <typename ValueType, typename ScalarType,
          typename = std::enable_if_t<
              std::is_convertible<ScalarType, ValueType>::value>,
          typename... Summands2>
scaled_sum_expression<ValueType, Summands2...> operator*(
    ScalarType s, sum_expression<ValueType, Summands2...> e2)
{
    return {detail::create_scalar<ValueType>(get_executor(e2), s), e2};
}


// Sum * s = sSum
template <typename ValueType, typename... Summands1>
scaled_sum_expression<ValueType, Summands1...> operator*(
    sum_expression<ValueType, Summands1...> e1, scalar_expression<ValueType> e2)
{
    return {e2, e1};
}

template <typename ValueType, typename ScalarType,
          typename = std::enable_if_t<
              std::is_convertible<ScalarType, ValueType>::value>,
          typename... Summands1>
scaled_sum_expression<ValueType, Summands1...> operator*(
    sum_expression<ValueType, Summands1...> e1, ScalarType s)
{
    return {detail::create_scalar<ValueType>(get_executor(e1), s), e1};
}


// -Sum = sSum
template <typename ValueType, typename... Summands1>
scaled_sum_expression<ValueType, Summands1...> operator-(
    sum_expression<ValueType, Summands1...> e1)
{
    return -1 * e1;
}


// s * Product = sProduct
template <typename ValueType, typename... Factors2>
scaled_product_expression<ValueType, Factors2...> operator*(
    scalar_expression<ValueType> e1,
    product_expression<ValueType, Factors2...> e2)
{
    return {e1, e2};
}

template <typename ValueType, typename ScalarType,
          typename = std::enable_if_t<
              std::is_convertible<ScalarType, ValueType>::value>,
          typename... Factors2>
scaled_product_expression<ValueType, Factors2...> operator*(
    ScalarType s, product_expression<ValueType, Factors2...> e2)
{
    return {detail::create_scalar<ValueType>(get_executor(e2), s), e2};
}


// Product * s = sProduct
template <typename ValueType, typename... Factors1>
scaled_product_expression<ValueType, Factors1...> operator*(
    product_expression<ValueType, Factors1...> e1,
    scalar_expression<ValueType> e2)
{
    return {e2, e1};
}

template <typename ValueType, typename ScalarType,
          typename = std::enable_if_t<
              std::is_convertible<ScalarType, ValueType>::value>,
          typename... Factors2>
scaled_product_expression<ValueType, Factors2...> operator*(
    product_expression<ValueType, Factors2...> e1, ScalarType s)
{
    return {detail::create_scalar<ValueType>(get_executor(e1), s), e1};
}


// -Product = sProduct
template <typename ValueType, typename... Factors1>
scaled_product_expression<ValueType, Factors1...> operator-(
    product_expression<ValueType, Factors1...> e1)
{
    return -1 * e1;
}


// sA * B = sProduct
template <typename ValueType>
scaled_product_expression<ValueType, linop_expression<ValueType>,
                          linop_expression<ValueType>>
operator*(scaled_linop_expression<ValueType> e1, linop_expression<ValueType> e2)
{
    return {e1.scale, {std::make_tuple(e1.op, e2)}};
}


// A * sB = sProduct
template <typename ValueType>
scaled_product_expression<ValueType, linop_expression<ValueType>,
                          linop_expression<ValueType>>
operator*(linop_expression<ValueType> e1, scaled_linop_expression<ValueType> e2)
{
    return {e2.scale, {std::make_tuple(e1, e2.op)}};
}


// sSum * A = sProduct
template <typename ValueType, typename... Summands1>
scaled_product_expression<ValueType, sum_expression<ValueType, Summands1...>,
                          linop_expression<ValueType>>
operator*(scaled_sum_expression<ValueType, Summands1...> e1,
          linop_expression<ValueType> e2)
{
    return {e1.scale, {std::make_tuple(e1.sum, e2)}};
}


// Sum * sA = sProduct
template <typename ValueType, typename... Summands1>
scaled_product_expression<ValueType, sum_expression<ValueType, Summands1...>,
                          linop_expression<ValueType>>
operator*(sum_expression<ValueType, Summands1...> e1,
          scaled_linop_expression<ValueType> e2)
{
    return {e2.scale, {std::make_tuple(e1, e2.op)}};
}


// sA * Sum = sProduct
template <typename ValueType, typename... Summands2>
scaled_product_expression<ValueType, linop_expression<ValueType>,
                          sum_expression<ValueType, Summands2...>>
operator*(scaled_linop_expression<ValueType> e1,
          sum_expression<ValueType, Summands2...> e2)
{
    return {e1.scale, std::make_tuple(e1.op, e2)};
}


// A * sSum = sProduct
template <typename ValueType, typename... Summands2>
scaled_product_expression<ValueType, linop_expression<ValueType>,
                          sum_expression<ValueType, Summands2...>>
operator*(linop_expression<ValueType> e1,
          scaled_sum_expression<ValueType, Summands2...> e2)
{
    return {e2.scale, std::make_tuple(e1, e2.sum)};
}


// sSum * Sum = sProduct
template <typename ValueType, typename... Summands1, typename... Summands2>
scaled_product_expression<ValueType, sum_expression<ValueType, Summands1...>,
                          sum_expression<ValueType, Summands2...>>
operator*(scaled_sum_expression<ValueType, Summands1...> e1,
          sum_expression<ValueType, Summands2...> e2)
{
    return {e1.scale, {std::make_tuple(e1.sum, e2)}};
}


// sSum * Product = sProduct
template <typename ValueType, typename... Summands1, typename... Factors2>
scaled_product_expression<ValueType, sum_expression<ValueType, Summands1...>,
                          Factors2...>
operator*(scaled_sum_expression<ValueType, Summands1...> e1,
          product_expression<ValueType, Factors2...> e2)
{
    return {e1.scale, {std::tuple_cat(std::make_tuple(e1.sum), e2.factors)}};
}


// Sum * sProduct = sProduct
template <typename ValueType, typename... Summands1, typename... Factors2>
scaled_product_expression<ValueType, sum_expression<ValueType, Summands1...>,
                          Factors2...>
operator*(sum_expression<ValueType, Summands1...> e1,
          scaled_product_expression<ValueType, Factors2...> e2)
{
    return {e2.scale,
            {std::tuple_cat(std::make_tuple(e1), e2.product.factors)}};
}


// sProduct * Sum = sProduct
template <typename ValueType, typename... Factors1, typename... Summands2>
scaled_product_expression<ValueType, Factors1...,
                          sum_expression<ValueType, Summands2...>>
operator*(scaled_product_expression<ValueType, Factors1...> e1,
          sum_expression<ValueType, Summands2...> e2)
{
    return {e1.scale,
            {std::tuple_cat(e1.product.factors, std::make_tuple(e2))}};
}


// Product * sSum = sProduct
template <typename ValueType, typename... Factors1, typename... Summands2>
scaled_product_expression<ValueType, Factors1...,
                          sum_expression<ValueType, Summands2...>>
operator*(product_expression<ValueType, Factors1...> e1,
          scaled_sum_expression<ValueType, Summands2...> e2)
{
    return {e2.scale, {std::tuple_cat(e1.factors, std::make_tuple(e2.sum))}};
}


// sProduct * Product = sProduct
template <typename ValueType, typename... Factors1, typename... Factors2>
scaled_product_expression<ValueType, Factors1..., Factors2...> operator*(
    scaled_product_expression<ValueType, Factors1...> e1,
    product_expression<ValueType, Factors2...> e2)
{
    return {e1.scale, {std::tuple_cat(e1.product.factors, e2.factors)}};
}


// Product * sProduct = sProduct
template <typename ValueType, typename... Factors1, typename... Factors2>
scaled_product_expression<ValueType, Factors1..., Factors2...> operator*(
    product_expression<ValueType, Factors1...> e1,
    scaled_product_expression<ValueType, Factors2...> e2)
{
    return {e2.scale, {std::tuple_cat(e1.factors, e2.product.factors)}};
}


// sA + sB = Sum
template <typename ValueType>
sum_expression<ValueType, scaled_linop_expression<ValueType>,
               scaled_linop_expression<ValueType>>
operator+(scaled_linop_expression<ValueType> e1,
          scaled_linop_expression<ValueType> e2)
{
    return {std::make_tuple(e1, e2)};
}


// A + B = Sum
template <typename ValueType>
sum_expression<ValueType, scaled_linop_expression<ValueType>,
               scaled_linop_expression<ValueType>>
operator+(linop_expression<ValueType> e1, linop_expression<ValueType> e2)
{
    return 1 * e1 + 1 * e2;
}


// sA + B = Sum
template <typename ValueType>
sum_expression<ValueType, scaled_linop_expression<ValueType>,
               scaled_linop_expression<ValueType>>
operator+(scaled_linop_expression<ValueType> e1, linop_expression<ValueType> e2)
{
    return e1 + 1 * e2;
}


// A + sB = Sum
template <typename ValueType>
sum_expression<ValueType, scaled_linop_expression<ValueType>,
               scaled_linop_expression<ValueType>>
operator+(linop_expression<ValueType> e1, scaled_linop_expression<ValueType> e2)
{
    return 1 * e1 + e2;
}


// sProduct + sProduct = Sum
template <typename ValueType, typename... Factors1, typename... Factors2>
sum_expression<ValueType, scaled_product_expression<ValueType, Factors1...>,
               scaled_product_expression<ValueType, Factors2...>>
operator+(scaled_product_expression<ValueType, Factors1...> e1,
          scaled_product_expression<ValueType, Factors2...> e2)
{
    return {std::make_tuple(e1, e2)};
}


// Product + Product = Sum
template <typename ValueType, typename... Factors1, typename... Factors2>
sum_expression<ValueType, scaled_product_expression<ValueType, Factors1...>,
               scaled_product_expression<ValueType, Factors2...>>
operator+(product_expression<ValueType, Factors1...> e1,
          product_expression<ValueType, Factors2...> e2)
{
    return 1 * e1 + 1 * e2;
}


// sProduct + Product = Sum
template <typename ValueType, typename... Factors1, typename... Factors2>
sum_expression<ValueType, scaled_product_expression<ValueType, Factors1...>,
               scaled_product_expression<ValueType, Factors2...>>
operator+(scaled_product_expression<ValueType, Factors1...> e1,
          product_expression<ValueType, Factors2...> e2)
{
    return e1 + 1 * e2;
}


// Product + sProduct = Sum
template <typename ValueType, typename... Factors1, typename... Factors2>
sum_expression<ValueType, scaled_product_expression<ValueType, Factors1...>,
               scaled_product_expression<ValueType, Factors2...>>
operator+(product_expression<ValueType, Factors1...> e1,
          scaled_product_expression<ValueType, Factors2...> e2)
{
    return 1 * e1 + e2;
}


// sA + Sum = Sum
template <typename ValueType, typename... Summands2>
sum_expression<ValueType, scaled_linop_expression<ValueType>, Summands2...>
operator+(scaled_linop_expression<ValueType> e1,
          sum_expression<ValueType, Summands2...> e2)
{
    return {std::tuple_cat(std::make_tuple(e1), e2.summands)};
}


// A + Sum = Sum
template <typename ValueType, typename... Summands2>
sum_expression<ValueType, scaled_linop_expression<ValueType>, Summands2...>
operator+(linop_expression<ValueType> e1,
          sum_expression<ValueType, Summands2...> e2)
{
    return 1 * e1 + e2;
}


// Sum + sA = Sum
template <typename ValueType, typename... Summands1>
sum_expression<ValueType, Summands1..., scaled_linop_expression<ValueType>>
operator+(sum_expression<ValueType, Summands1...> e1,
          scaled_linop_expression<ValueType> e2)
{
    return {std::tuple_cat(e1.summands, std::make_tuple(e2))};
}


// Sum + A = Sum
template <typename ValueType, typename... Summands1>
sum_expression<ValueType, scaled_linop_expression<ValueType>, Summands1...>
operator+(sum_expression<ValueType, Summands1...> e1,
          linop_expression<ValueType> e2)
{
    return e1 + 1 * e2;
}


// sProduct + sA = Sum
template <typename ValueType, typename... Factors1>
sum_expression<ValueType, scaled_product_expression<ValueType, Factors1...>,
               scaled_linop_expression<ValueType>>
operator+(scaled_product_expression<ValueType, Factors1...> e1,
          scaled_linop_expression<ValueType> e2)
{
    return {std::make_tuple(e1, e2)};
}


// Product + A = Sum
template <typename ValueType, typename... Factors1>
sum_expression<ValueType, scaled_product_expression<ValueType, Factors1...>,
               scaled_linop_expression<ValueType>>
operator+(product_expression<ValueType, Factors1...> e1,
          linop_expression<ValueType> e2)
{
    return 1 * e1 + 1 * e2;
}


// Product + sA = Sum
template <typename ValueType, typename... Factors1>
sum_expression<ValueType, scaled_product_expression<ValueType, Factors1...>,
               scaled_linop_expression<ValueType>>
operator+(product_expression<ValueType, Factors1...> e1,
          scaled_linop_expression<ValueType> e2)
{
    return 1 * e1 + e2;
}


// sProduct + A = Sum
template <typename ValueType, typename... Factors1>
sum_expression<ValueType, scaled_product_expression<ValueType, Factors1...>,
               scaled_linop_expression<ValueType>>
operator+(scaled_product_expression<ValueType, Factors1...> e1,
          linop_expression<ValueType> e2)
{
    return e1 + 1 * e2;
}


// sA + sProduct = Sum
template <typename ValueType, typename... Factors2>
sum_expression<ValueType, scaled_linop_expression<ValueType>,
               scaled_product_expression<ValueType, Factors2...>>
operator+(scaled_linop_expression<ValueType> e1,
          scaled_product_expression<ValueType, Factors2...> e2)
{
    return {std::make_tuple(e1, e2)};
}


// A + Product = Sum
template <typename ValueType, typename... Factors2>
sum_expression<ValueType, scaled_linop_expression<ValueType>,
               scaled_product_expression<ValueType, Factors2...>>
operator+(linop_expression<ValueType> e1,
          product_expression<ValueType, Factors2...> e2)
{
    return 1 * e1 + 1 * e2;
}


// A + sProduct = Sum
template <typename ValueType, typename... Factors2>
sum_expression<ValueType, scaled_linop_expression<ValueType>,
               scaled_product_expression<ValueType, Factors2...>>
operator+(linop_expression<ValueType> e1,
          scaled_product_expression<ValueType, Factors2...> e2)
{
    return 1 * e1 + e2;
}


// sA + Product = Sum
template <typename ValueType, typename... Factors2>
sum_expression<ValueType, scaled_linop_expression<ValueType>,
               scaled_product_expression<ValueType, Factors2...>>
operator+(scaled_linop_expression<ValueType> e1,
          product_expression<ValueType, Factors2...> e2)
{
    return e1 + 1 * e2;
}


// Sum + Sum = Sum
template <typename ValueType, typename... Factors1, typename... Factors2>
sum_expression<ValueType, Factors1..., Factors2...> operator+(
    sum_expression<ValueType, Factors1...> e1,
    sum_expression<ValueType, Factors2...> e2)
{
    return {std::tuple_cat(e1.summands, e2.summands)};
}


}  // namespace expression
}  // namespace gko

#endif  // GKO_CORE_SYNTHESIZER_EXPRESSION_OPERATORS_HPP_
