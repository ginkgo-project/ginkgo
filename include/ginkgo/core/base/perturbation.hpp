// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_PERTURBATION_HPP_
#define GKO_PUBLIC_CORE_BASE_PERTURBATION_HPP_


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

    Perturbation& operator=(const Perturbation& other);

    Perturbation& operator=(Perturbation&& other);

    Perturbation(const Perturbation& other);

    Perturbation(Perturbation&& other);

    /**
     * Creates an empty perturbation operator (0x0 operator).
     *
     * @param exec  Executor associated to the perturbation
     *
     * @return A smart pointer to the newly created perturbation.
     */
    static std::unique_ptr<Perturbation> create(
        std::shared_ptr<const Executor> exec);

    /**
     * Creates a perturbation with scalar and basis by setting projector to the
     * conjugate transpose of basis. Basis must be transposable. Perturbation
     * will throw gko::NotSupported if basis is not transposable.
     *
     * @param scalar  scaling of the movement
     * @param basis  the direction basis
     *
     * @return A smart pointer to the newly created perturbation.
     */
    static std::unique_ptr<Perturbation> create(
        std::shared_ptr<const LinOp> scalar,
        std::shared_ptr<const LinOp> basis);

    /**
     * Creates a perturbation of scalar, basis and projector.
     *
     * @param scalar  scaling of the movement
     * @param basis  the direction basis
     * @param projector  decides the coefficient of basis
     *
     * @return A smart pointer to the newly created perturbation.
     */
    static std::unique_ptr<Perturbation> create(
        std::shared_ptr<const LinOp> scalar, std::shared_ptr<const LinOp> basis,
        std::shared_ptr<const LinOp> projector);

protected:
    explicit Perturbation(std::shared_ptr<const Executor> exec);

    explicit Perturbation(std::shared_ptr<const LinOp> scalar,
                          std::shared_ptr<const LinOp> basis);

    explicit Perturbation(std::shared_ptr<const LinOp> scalar,
                          std::shared_ptr<const LinOp> basis,
                          std::shared_ptr<const LinOp> projector);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    /**
     * Validates the dimensions of the `scalar`, `basis` and `projector`
     * parameters for the `apply`. scalar must be 1 by 1. The dimension of basis
     * should be same as the dimension of conjugate transpose of projector.
     */
    void validate_perturbation();

private:
    std::shared_ptr<const LinOp> basis_;
    std::shared_ptr<const LinOp> projector_;
    std::shared_ptr<const LinOp> scalar_;

    // TODO: solve race conditions when multithreading
    mutable struct cache_struct {
        cache_struct() = default;
        ~cache_struct() = default;
        cache_struct(const cache_struct& other) {}
        cache_struct& operator=(const cache_struct& other) { return *this; }

        // allocate linops of cache. The dimension of `intermediate` is
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


#endif  // GKO_PUBLIC_CORE_BASE_PERTURBATION_HPP_
