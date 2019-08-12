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

#ifndef GKO_CORE_BASE_REFLECTION_HPP_
#define GKO_CORE_BASE_REFLECTION_HPP_


#include <vector>


#include <ginkgo/core/base/lin_op.hpp>


namespace gko {


/**
 * The Reflection class can be used to be as (I+coef*UV)
 *
 * @tparam ValueType  precision of input and result vectors
 *
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Reflection : public EnableLinOp<Reflection<ValueType>>,
                   public EnableCreateMethod<Reflection<ValueType>> {
    friend class EnablePolymorphicObject<Reflection, LinOp>;
    friend class EnableCreateMethod<Reflection>;

public:
    using value_type = ValueType;

    /**
     * Returns the operator U of the reflection.
     *
     * @return the operator U
     */
    const std::shared_ptr<const LinOp> get_u_operator() const noexcept
    {
        return U_;
    }

    /**
     * Returns the operator V of the reflection.
     *
     * @return the operator V
     */
    const std::shared_ptr<const LinOp> get_v_operator() const noexcept
    {
        return V_;
    }

    /**
     * Returns the coefficient of the reflection.
     *
     * @return the coefficent coef
     */
    const std::shared_ptr<const LinOp> get_coefficient() const noexcept
    {
        return coef_;
    }

protected:
    /**
     * Creates an empty operator reflection (0x0 operator).
     *
     * @param exec  Executor associated to the reflection
     */
    explicit Reflection(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Reflection>(std::move(exec))
    {}

    /**
     * Creates a reflection of operators with only one operator U.
     *    (V is set to the conjugate transpose of U.)
     *
     * @param coef  the coefficient
     * @param U  the operator U
     *
     */
    explicit Reflection(std::shared_ptr<const LinOp> coef,
                        std::shared_ptr<const LinOp> U)
        : Reflection(std::move(coef), std::move(U),
                     std::move(as<Transposable>(lend(U))->conj_transpose()))
    {}

    /**
     * Creates a reflection of operators.
     *
     * @param coef  the coefficient
     * @param U  the operator U
     * @param V  the operator V
     *
     */
    explicit Reflection(std::shared_ptr<const LinOp> coef,
                        std::shared_ptr<const LinOp> U,
                        std::shared_ptr<const LinOp> V)
        : EnableLinOp<Reflection>(U->get_executor(),
                                  gko::dim<2>{U->get_size()[0]}),
          coef_{std::move(coef)},
          U_{std::move(U)},
          V_{std::move(V)}
    {
        this->validate_reflection();
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    /**
     * Validates the dimension of coef, U, V.
     * coef should be 1 by 1.
     * The dimension of U should be same as the dimension of conjugate transpose
     * of V.
     */
    void validate_reflection()
    {
        GKO_ASSERT_CONFORMANT(U_, V_);
        GKO_ASSERT_CONFORMANT(V_, U_);
        GKO_ASSERT_EQUAL_DIMENSIONS(coef_, dim<2>(1, 1));
    }

private:
    std::shared_ptr<const LinOp> U_;
    std::shared_ptr<const LinOp> V_;
    std::shared_ptr<const LinOp> coef_;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_REFLECTION_HPP_
