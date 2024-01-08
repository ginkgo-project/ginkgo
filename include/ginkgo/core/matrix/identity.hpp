// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_IDENTITY_HPP_
#define GKO_PUBLIC_CORE_MATRIX_IDENTITY_HPP_


#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace matrix {


/**
 * This class is a utility which efficiently implements the identity matrix (a
 * linear operator which maps each vector to itself).
 *
 * Thus, objects of the Identity class always represent a square matrix, and
 * don't require any storage for their values. The apply method is implemented
 * as a simple copy (or a linear combination).
 *
 * @note This class is useful when composing it with other operators. For
 *       example, it can be used instead of a preconditioner in Krylov solvers,
 *       if one wants to run a "plain" solver, without using a preconditioner.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup identity
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Identity : public EnableLinOp<Identity<ValueType>>,
                 public EnableCreateMethod<Identity<ValueType>>,
                 public Transposable {
    friend class EnablePolymorphicObject<Identity, LinOp>;
    friend class EnableCreateMethod<Identity>;

public:
    using EnableLinOp<Identity>::convert_to;
    using EnableLinOp<Identity>::move_to;

    using value_type = ValueType;
    using transposed_type = Identity<ValueType>;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;


protected:
    /**
     * Creates an empty Identity matrix.
     *
     * @param exec  Executor associated to the matrix
     */
    explicit Identity(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Identity>(exec)
    {}

    /**
     * Creates an Identity matrix of the specified size.
     *
     * @param size  size of the matrix (must be square)
     */
    Identity(std::shared_ptr<const Executor> exec, dim<2> size)
        : EnableLinOp<Identity>(exec, size)
    {
        GKO_ASSERT_IS_SQUARE_MATRIX(this);
    }

    /**
     * Creates an Identity matrix of the specified size.
     *
     * @param size  size of the matrix
     */
    Identity(std::shared_ptr<const Executor> exec, size_type size)
        : EnableLinOp<Identity>(exec, dim<2>{size})
    {}

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;
};


/**
 * This factory is a utility which can be used to generate Identity operators.
 *
 * The factory will generate the Identity matrix with the same dimension as
 * the passed in operator. It will throw an exception if the operator is not
 * square.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class IdentityFactory
    : public EnablePolymorphicObject<IdentityFactory<ValueType>, LinOpFactory> {
    friend class EnablePolymorphicObject<IdentityFactory, LinOpFactory>;

public:
    using value_type = ValueType;

    /**
     * Creates a new Identity factory.
     *
     * @param exec  the executor where the Identity operator will be stored
     *
     * @return a unique pointer to the newly created factory
     */
    static std::unique_ptr<IdentityFactory> create(
        std::shared_ptr<const Executor> exec)
    {
        return std::unique_ptr<IdentityFactory>(
            new IdentityFactory(std::move(exec)));
    }

protected:
    std::unique_ptr<LinOp> generate_impl(
        std::shared_ptr<const LinOp> base) const override;

    IdentityFactory(std::shared_ptr<const Executor> exec)
        : EnablePolymorphicObject<IdentityFactory, LinOpFactory>(exec)
    {}
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_IDENTITY_HPP_
