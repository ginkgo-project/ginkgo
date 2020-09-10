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

#ifndef GKO_CORE_SYNTHESIZER_EXPRESSION_BUILDERS_HPP_
#define GKO_CORE_SYNTHESIZER_EXPRESSION_BUILDERS_HPP_

#include "expression_types.hpp"

namespace gko {
namespace expression {


// pick the first executor we can find in a LinOp (not scalar)
template <typename ValueType>
std::shared_ptr<const Executor> get_executor(linop_expression<ValueType> expr)
{
    return expr.op->get_executor();
}


template <typename ValueType, typename... Summands>
std::shared_ptr<const Executor> get_executor(
    sum_expression<ValueType, Summands...> expr)
{
    return get_executor(std::get<0>(expr.summands));
}


template <typename ValueType, typename... Factors>
std::shared_ptr<const Executor> get_executor(
    product_expression<ValueType, Factors...> expr)
{
    return get_executor(std::get<0>(expr.factors));
}


template <typename ValueType>
std::shared_ptr<const Executor> get_executor(
    scaled_linop_expression<ValueType> expr)
{
    return get_executor(expr.op);
}


template <typename ValueType, typename... Summands>
std::shared_ptr<const Executor> get_executor(
    scaled_sum_expression<ValueType, Summands...> expr)
{
    return get_executor(expr.sum);
}


template <typename ValueType, typename... Factors>
std::shared_ptr<const Executor> get_executor(
    scaled_product_expression<ValueType, Factors...> expr)
{
    return get_executor(expr.product);
}


// build the expression into a combined LinOp
template <typename ValueType>
std::shared_ptr<LinOp> build(linop_expression<ValueType> expr)
{
    return expr.op;
}


template <typename ValueType, typename Tuple, std::size_t... Idxs>
std::shared_ptr<LinOp> build_composition_impl(Tuple factors,
                                              std::index_sequence<Idxs...>)
{
    return Composition<ValueType>::create(build(std::get<Idxs>(factors))...);
}


template <typename ValueType, typename... Factors>
std::shared_ptr<LinOp> build(product_expression<ValueType, Factors...> expr)
{
    return build_composition_impl(expr.factors,
                                  std::index_sequence_for<Factors...>{});
}


template <typename ValueType, typename Tuple, std::size_t... Idxs>
std::shared_ptr<LinOp> build_combination_impl(Tuple parameters,
                                              std::index_sequence<Idxs...>)
{
    return Combination<ValueType>(std::get<Idxs>(parameters)...);
}


template <typename ValueType>
std::shared_ptr<LinOp> build_combination_scalar_impl(
    std::shared_ptr<const Executor>, scaled_linop_expression<ValueType> expr)
{
    return expr.scale.op;
}


template <typename ValueType, typename... Factors>
std::shared_ptr<LinOp> build_combination_scalar_impl(
    std::shared_ptr<const Executor>,
    scaled_product_expression<ValueType, Factors...> expr)
{
    return expr.scale.op;
}


template <typename ValueType>
std::shared_ptr<LinOp> build_combination_operator_impl(
    linop_expression<ValueType> expr)
{
    return build(expr);
}


template <typename ValueType, typename... Factors>
std::shared_ptr<LinOp> build_combination_operator_impl(
    product_expression<ValueType, Factors...> expr)
{
    return build(expr);
}


template <typename ValueType>
std::shared_ptr<LinOp> build_combination_operator_impl(
    std::shared_ptr<const Executor>, scaled_linop_expression<ValueType> expr)
{
    return build(expr.op);
}


template <typename ValueType, typename... Factors>
std::shared_ptr<LinOp> build_combination_operator_impl(
    std::shared_ptr<const Executor>,
    scaled_product_expression<ValueType, Factors...> expr)
{
    return build(expr.product);
}


template <typename ValueType, typename Tuple, std::size_t... Idxs>
std::shared_ptr<LinOp> build_combination_interleave_impl(
    Tuple factors, std::index_sequence<Idxs...>)
{
    return build_combination_impl<ValueType>(std::tuple_cat(std::make_tuple(
        build_combination_scalar_impl<ValueType>(
            get_executor(std::get<0>(factors)), std::get<Idxs>(factors))...,
        build_combination_operator_impl<ValueType>(
            std::get<Idxs>(factors))...)));
}


template <typename ValueType, typename... Factors>
std::shared_ptr<LinOp> build(sum_expression<ValueType, Factors...> expr)
{
    return build_combination_interleave_impl<ValueType>(
        expr.summands, std::index_sequence_for<Factors...>{});
}


}  // namespace expression
}  // namespace gko

#endif  // GKO_CORE_SYNTHESIZER_EXPRESSION_BUILDERS_HPP_
