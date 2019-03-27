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

#ifndef GKO_CORE_MATRIX_IDENTITY_HPP_
#define GKO_CORE_MATRIX_IDENTITY_HPP_


#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
/**
 * @brief The matrix namespace.
 *
 * \ingroup matrix
 */
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
 * \ingroup identity
 * \ingroup mat_formats
 * \ingroup LinOp
 */
template <typename ValueType = default_precision>
class Identity : public EnableLinOp<Identity<ValueType>>,
                 public EnableCreateMethod<Identity<ValueType>> {
    friend class EnablePolymorphicObject<Identity, LinOp>;
    friend class EnableCreateMethod<Identity>;

public:
    using EnableLinOp<Identity>::convert_to;
    using EnableLinOp<Identity>::move_to;

    using value_type = ValueType;

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
     * @param size  size of the matrix
     */
    Identity(std::shared_ptr<const Executor> exec, size_type size)
        : EnableLinOp<Identity>(exec, dim<2>{size})
    {}

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;
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
 * \ingroup mat_formats
 * \ingroup LinOp
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


#endif  // GKO_CORE_MATRIX_IDENTITY_HPP_
