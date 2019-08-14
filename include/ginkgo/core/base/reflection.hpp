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


#include <memory>


#include <ginkgo/core/base/lin_op.hpp>


namespace gko {


/**
 * The Reflection class can be used to construct a LinOp to represent the
 * `(identity + scaler * basis * projector)` This operator adds a movement along
 * a direction construted by `basis` and `projector` on the LinOp. `projector`
 * gives the coefficient of `basis` to decide the direction.
 * For example, Householder matrix can be represented in Reflection.
 * Householder matrix = (I - 2 u u*), u is the housholder factor
 * scaler = -2, basis = u, and projector = u*
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
     * Returns the basis of the reflection.
     *
     * @return the basis
     */
    const std::shared_ptr<const LinOp> get_basis() const noexcept
    {
        return basis_;
    }

    /**
     * Returns the projector of the reflection.
     *
     * @return the projector
     */
    const std::shared_ptr<const LinOp> get_projector() const noexcept
    {
        return projector_;
    }

    /**
     * Returns the scaler of the reflection.
     *
     * @return the scaler
     */
    const std::shared_ptr<const LinOp> get_scaler() const noexcept
    {
        return scaler_;
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
     * Creates a reflection with scaler and basis by setting projector to the
     * conjugate transpose of basis. Basis must be transposable. Reflection will
     * throw GKO_NOT_SUPPORT if basis is not transposable.
     *
     * @param scaler  scaling of the movement
     * @param basis  the direction basis
     */
    explicit Reflection(std::shared_ptr<const LinOp> scaler,
                        std::shared_ptr<const LinOp> basis)
        : Reflection(
              std::move(scaler),
              // basis can not be std::move(basis). Otherwise, Program deletes
              // basis before applying conjugate transpose
              basis,
              std::move((as<gko::Transposable>(lend(basis)))->conj_transpose()))
    {}

    /**
     * Creates a reflection of scaler, basis and projector.
     *
     * @param scaler  scaling of the movement
     * @param basis  the direction basis
     * @param projector  decides the coefficient of basis
     */
    explicit Reflection(std::shared_ptr<const LinOp> scaler,
                        std::shared_ptr<const LinOp> basis,
                        std::shared_ptr<const LinOp> projector)
        : EnableLinOp<Reflection>(basis->get_executor(),
                                  gko::dim<2>{basis->get_size()[0]}),
          scaler_{std::move(scaler)},
          basis_{std::move(basis)},
          projector_{std::move(projector)}
    {
        this->validate_reflection();
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    /**
     * validate_reflection check the dimension of scaler, basis, projector.
     * scaler must be 1 by 1.
     * The dimension of basis should be same as the dimension of conjugate
     * transpose of projector.
     */
    void validate_reflection()
    {
        GKO_ASSERT_CONFORMANT(basis_, projector_);
        GKO_ASSERT_CONFORMANT(projector_, basis_);
        GKO_ASSERT_EQUAL_DIMENSIONS(scaler_, dim<2>(1, 1));
    }

private:
    std::shared_ptr<const LinOp> basis_;
    std::shared_ptr<const LinOp> projector_;
    std::shared_ptr<const LinOp> scaler_;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_REFLECTION_HPP_
