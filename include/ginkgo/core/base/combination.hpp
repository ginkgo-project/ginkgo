/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_CORE_BASE_COMBINATION_HPP_
#define GKO_CORE_BASE_COMBINATION_HPP_


#include <vector>


#include <ginkgo/core/base/lin_op.hpp>


namespace gko {


/**
 * The Combination class can be used to construct a linear combination of
 * multiple linear operators `c1 * op1 + c2 * op2 + ... + ck * opk`.
 *
 * @tparam ValueType  precision of input and result vectors
 *
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Combination : public EnableLinOp<Combination<ValueType>>,
                    public EnableCreateMethod<Combination<ValueType>> {
    friend class EnablePolymorphicObject<Combination, LinOp>;
    friend class EnableCreateMethod<Combination>;

public:
    using value_type = ValueType;

    /**
     * Returns a list of coefficients of the combination.
     *
     * @return a list of coefficients
     */
    const std::vector<std::shared_ptr<const LinOp>> &get_coefficients() const
        noexcept
    {
        return coefficients_;
    }

    /**
     * Returns a list of operators of the combination.
     *
     * @return a list of operators
     */
    const std::vector<std::shared_ptr<const LinOp>> &get_operators() const
        noexcept
    {
        return operators_;
    }

protected:
    /**
     * Creates an empty linear combination (0x0 operator).
     *
     * @param exec  Executor associated to the linear combination
     */
    explicit Combination(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Combination>(exec)
    {}

    /**
     * Creates a linear combination of operators using the specified list of
     * coefficients and operators.
     *
     * @tparam CoefficientIterator  a class representing iterators over the
     *                              coefficients of the linear combination
     * @tparam OperatorIterator  a class representing iterators over the
     *                           operators of the linear combination
     *
     * @param coefficient_begin  iterator pointing to the first coefficient
     * @param coefficient_end  iterator pointing behind the last coefficient
     * @param operator_begin  iterator pointing to the first operator
     * @param operator_end  iterator pointing behind the last operator
     */
    template <
        typename CoefficientIterator, typename OperatorIterator,
        typename = xstd::void_t<
            typename std::iterator_traits<
                CoefficientIterator>::iterator_category,
            typename std::iterator_traits<OperatorIterator>::iterator_category>>
    explicit Combination(CoefficientIterator coefficient_begin,
                         CoefficientIterator coefficient_end,
                         OperatorIterator operator_begin,
                         OperatorIterator operator_end)
        : EnableLinOp<Combination>([&] {
              if (operator_begin == operator_end) {
                  throw OutOfBoundsError(__FILE__, __LINE__, 1, 0);
              }
              return (*operator_begin)->get_executor();
          }()),
          coefficients_(coefficient_begin, coefficient_end),
          operators_(operator_begin, operator_end)
    {
        for (const auto &c : coefficients_) {
            GKO_ASSERT_EQUAL_DIMENSIONS(c, dim<2>(1, 1));
        }
        this->set_size(operators_[0]->get_size());
        for (const auto &o : operators_) {
            GKO_ASSERT_EQUAL_DIMENSIONS(o, this->get_size());
        }
    }

    /**
     * Creates a linear combination of operators using the specified list of
     * coefficients and operators.
     *
     * @tparam Rest  types of trailing parameters
     *
     * @param coef  the first coefficient
     * @param oper  the first operator
     * @param rest  other coefficient and operators (interleaved)
     */
    template <typename... Rest>
    explicit Combination(std::shared_ptr<const LinOp> coef,
                         std::shared_ptr<const LinOp> oper, Rest &&... rest)
        : Combination(std::forward<Rest>(rest)...)
    {
        GKO_ASSERT_EQUAL_DIMENSIONS(coef, dim<2>(1, 1));
        GKO_ASSERT_EQUAL_DIMENSIONS(oper, this->get_size());
        coefficients_.insert(begin(coefficients_), coef);
        operators_.insert(begin(operators_), oper);
    }

    /**
     * Creates a linear combination of operators using the specified list of
     * coefficients and operators.
     *
     * @tparam Rest  types of trailing parameters
     *
     * @param coef  the first coefficient
     * @param oper  the first operator
     *
     * @note this is the base case of the template constructor
     *       Combination(std::shared_ptr<const LinOp>, std::shared_ptr<const
     *       LinOp>, Rest &&...)
     */
    explicit Combination(std::shared_ptr<const LinOp> coef,
                         std::shared_ptr<const LinOp> oper)
        : EnableLinOp<Combination>(oper->get_executor(), oper->get_size()),
          coefficients_{coef},
          operators_{oper}
    {}

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

private:
    std::vector<std::shared_ptr<const LinOp>> coefficients_;
    std::vector<std::shared_ptr<const LinOp>> operators_;

    // TODO: solve race conditions when multithreading
    mutable struct cache_struct {
        cache_struct() = default;
        ~cache_struct() = default;
        cache_struct(const cache_struct &other) {}
        cache_struct &operator=(const cache_struct &other) { return *this; }

        std::unique_ptr<LinOp> zero;
        std::unique_ptr<LinOp> one;
        std::unique_ptr<LinOp> intermediate_x;
    } cache_;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_COMBINATION_HPP_
