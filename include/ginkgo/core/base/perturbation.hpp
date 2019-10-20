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

#ifndef GKO_CORE_BASE_PERTURBATION_HPP_
#define GKO_CORE_BASE_PERTURBATION_HPP_


#include <memory>


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {


/**
 * The Perturbation class can be used to construct a LinOp to represent the
 * operation `(identity + scalar * basis * projector)`. This operator adds a
 * movement along a direction constructed by `basis` and `projector` on the
 * LinOp. `projector` gives the coefficient of `basis` to decide the direction.
 *
 * For example, the Householder matrix can be represented with the Perturbation
 * operator as follows.
 * If u is the Householder factor then we can generate the [Householder
 * transformation](https://en.wikipedia.org/wiki/Householder_transformation),
 * H = (I - 2 u u*). In this case, the parameters of Perturbation class are
 * scalar = -2, basis = u, and projector = u*.
 *
 * @tparam ValueType  precision of input and result vectors
 *
 * @note the apply operations of Perturbation class are not thread safe
 *
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Perturbation : public EnableLinOp<Perturbation<ValueType>>,
                     public EnableCreateMethod<Perturbation<ValueType>> {
    friend class EnablePolymorphicObject<Perturbation, LinOp>;
    friend class EnableCreateMethod<Perturbation>;

public:
    using value_type = ValueType;

    /**
     * Returns the basis of the perturbation.
     *
     * @return the basis of the perturbation
     */
    const std::shared_ptr<const LinOp> get_basis() const noexcept
    {
        return basis_;
    }

    /**
     * Returns the projector of the perturbation.
     *
     * @return the projector of the perturbation
     */
    const std::shared_ptr<const LinOp> get_projector() const noexcept
    {
        return projector_;
    }

    /**
     * Returns the scalar of the perturbation.
     *
     * @return the scalar of the perturbation
     */
    const std::shared_ptr<const LinOp> get_scalar() const noexcept
    {
        return scalar_;
    }

protected:
    /**
     * Creates an empty perturbation operator (0x0 operator).
     *
     * @param exec  Executor associated to the perturbation
     */
    explicit Perturbation(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Perturbation>(std::move(exec))
    {}

    /**
     * Creates a perturbation with scalar and basis by setting projector to the
     * conjugate transpose of basis. Basis must be transposable. Perturbation
     * will throw gko::NotSupported if basis is not transposable.
     *
     * @param scalar  scaling of the movement
     * @param basis  the direction basis
     */
    explicit Perturbation(std::shared_ptr<const LinOp> scalar,
                          std::shared_ptr<const LinOp> basis)
        : Perturbation(
              std::move(scalar),
              // basis can not be std::move(basis). Otherwise, Program deletes
              // basis before applying conjugate transpose
              basis,
              std::move((as<gko::Transposable>(lend(basis)))->conj_transpose()))
    {}

    /**
     * Creates a perturbation of scalar, basis and projector.
     *
     * @param scalar  scaling of the movement
     * @param basis  the direction basis
     * @param projector  decides the coefficient of basis
     */
    explicit Perturbation(std::shared_ptr<const LinOp> scalar,
                          std::shared_ptr<const LinOp> basis,
                          std::shared_ptr<const LinOp> projector)
        : EnableLinOp<Perturbation>(basis->get_executor(),
                                    gko::dim<2>{basis->get_size()[0]}),
          scalar_{std::move(scalar)},
          basis_{std::move(basis)},
          projector_{std::move(projector)}
    {
        this->validate_perturbation();
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    /**
     * Validates the dimensions of the `scalar`, `basis` and `projector`
     * parameters for the `apply`. scalar must be 1 by 1. The dimension of basis
     * should be same as the dimension of conjugate transpose of projector.
     */
    void validate_perturbation()
    {
        GKO_ASSERT_CONFORMANT(basis_, projector_);
        GKO_ASSERT_CONFORMANT(projector_, basis_);
        GKO_ASSERT_EQUAL_DIMENSIONS(scalar_, dim<2>(1, 1));
    }

private:
    std::shared_ptr<const LinOp> basis_;
    std::shared_ptr<const LinOp> projector_;
    std::shared_ptr<const LinOp> scalar_;

    // TODO: solve race conditions when multithreading
    mutable struct cache_struct {
        cache_struct() = default;
        ~cache_struct() = default;
        cache_struct(const cache_struct &other) {}
        cache_struct &operator=(const cache_struct &other) { return *this; }

        // allocate linops of cache. The dimenstion of `intermediate` is
        // (the number of rows of projector, the number of columns of b). Others
        // are 1x1 scalar.
        void allocate(std::shared_ptr<const Executor> exec, dim<2> size)
        {
            using vec = gko::matrix::Dense<ValueType>;
            if (one == nullptr) {
                one = initialize<vec>({gko::one<ValueType>()}, exec);
            }
            if (alpha_scalar == nullptr) {
                alpha_scalar = vec::create(exec, gko::dim<2>(1));
            }
            if (intermediate == nullptr || intermediate->get_size() != size) {
                intermediate = vec::create(exec, size);
            }
        }

        std::unique_ptr<LinOp> intermediate;
        std::unique_ptr<LinOp> one;
        std::unique_ptr<LinOp> alpha_scalar;
    } cache_;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_PERTURBATION_HPP_
