/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_BASE_COMPOSITION_HPP_
#define GKO_PUBLIC_CORE_BASE_COMPOSITION_HPP_


#include <vector>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {


/**
 * The Composition class can be used to compose linear operators `op1, op2, ...,
 * opn` and obtain the operator `op1 * op2 * ... * opn`.
 *
 * All LinOps of the Composition must operate on Dense inputs.
 * For an operator `op_k` that require an initial guess for their `apply`,
 * Composition provides either
 * * the output of the previous `op_{k+1}->apply` if `op_k` has square dimension
 * * zero if `op_k` is rectangular
 * as an initial guess.
 *
 * @tparam ValueType  precision of input and result vectors
 *
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Composition : public EnableLinOp<Composition<ValueType>>,
                    public EnableCreateMethod<Composition<ValueType>>,
                    public Transposable {
    friend class EnablePolymorphicObject<Composition, LinOp>;
    friend class EnableCreateMethod<Composition>;

public:
    using value_type = ValueType;
    using transposed_type = Composition<ValueType>;

    /**
     * Returns a list of operators of the composition.
     *
     * @return a list of operators
     */
    const std::vector<std::shared_ptr<const LinOp>> &get_operators() const
        noexcept
    {
        return operators_;
    }

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

protected:
    /**
     * Creates an empty operator composition (0x0 operator).
     *
     * @param exec  Executor associated to the composition
     */
    explicit Composition(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Composition>(exec), storage_{exec}
    {}

    /**
     * Creates a composition of operators using the operators in a range.
     *
     * @tparam Iterator  a class representing iterators over the
     *                  perators of the linear combination
     *
     * @param begin  iterator pointing to the first operator
     * @param end  iterator pointing behind the last operator
     */
    template <typename Iterator,
              typename = xstd::void_t<
                  typename std::iterator_traits<Iterator>::iterator_category>>
    explicit Composition(Iterator begin, Iterator end)
        : EnableLinOp<Composition>([&] {
              if (begin == end) {
                  throw OutOfBoundsError(__FILE__, __LINE__, 1, 0);
              }
              return (*begin)->get_executor();
          }()),
          storage_{(*begin)->get_executor()},
          operators_(begin, end)
    {
        this->set_size(gko::dim<2>{operators_.front()->get_size()[0],
                                   operators_.back()->get_size()[1]});
        for (size_type i = 1; i < operators_.size(); ++i) {
            GKO_ASSERT_CONFORMANT(operators_[i - 1], operators_[i]);
        }
    }

    /**
     * Creates a composition of operators using the specified list of operators.
     *
     * @tparam Rest  types of trailing parameters
     *
     * @param oper  the first operator
     * @param rest  remainging operators
     */
    template <typename... Rest>
    explicit Composition(std::shared_ptr<const LinOp> oper, Rest &&... rest)
        : Composition(std::forward<Rest>(rest)...)
    {
        GKO_ASSERT_CONFORMANT(oper, operators_[0]);
        operators_.insert(begin(operators_), oper);
        this->set_size(gko::dim<2>{operators_.front()->get_size()[0],
                                   operators_.back()->get_size()[1]});
    }

    /**
     * Creates a composition of operators using the specified list of operators.
     *
     * @param oper  the first operator
     *
     * @note this is the base case of the template constructor
     *       Composition(std::shared_ptr<const LinOp>, Rest &&...)
     */
    explicit Composition(std::shared_ptr<const LinOp> oper)
        : EnableLinOp<Composition>(oper->get_executor(), oper->get_size()),
          operators_{oper},
          storage_{oper->get_executor()}
    {}

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

private:
    std::vector<std::shared_ptr<const LinOp>> operators_;
    mutable Array<ValueType> storage_;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_COMPOSITION_HPP_
