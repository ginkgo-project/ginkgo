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

#ifndef GKO_CORE_PRECONDITIONER_ISAI_HPP_
#define GKO_CORE_PRECONDITIONER_ISAI_HPP_


#include <memory>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
/**
 * @brief The Preconditioner namespace.
 *
 * @ingroup precond
 */
namespace preconditioner {


/**
 * The Incomplete Sparse Approximate Inverse (ISAI) Preconditioner generates
 * approximate inverse matrices for a given lower triangular matrix L and upper
 * triangular matrix U. Using the precionditioner computes $aiU * aiL * x$
 * for a given vector x (may have multiple right hand sides). aiU and aiL
 * are the approximate inverses for U and L respectively.
 *
 * The sparsity pattern used for the approximate inverses is the same as
 * the sparsity pattern of the respective triangular matrix.
 * The L and U matrices need to be in a Composition<ValueType> object in the
 * order: L, U.
 *
 * For more details on the algorithm, see the paper
 * <a href="https://doi.org/10.1016/j.parco.2017.10.003">
 * Incomplete Sparse Approximate Inverses for Parallel Preconditioning</a>,
 * which is the basis for this work.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup isai
 * @ingroup precond
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Isai : public EnableLinOp<Isai<ValueType, IndexType>> {
    friend class EnableLinOp<Isai>;
    friend class EnablePolymorphicObject<Isai, LinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using Csr = matrix::Csr<ValueType, IndexType>;

    /**
     * Returns the approximate inverse of L.
     *
     * @returns the approximate inverse of L
     */
    std::shared_ptr<const Csr> get_approx_inverse_l() const { return l_inv_; }

    /**
     * Returns the approximate inverse of U.
     *
     * @returns the approximate inverse of U
     */
    std::shared_ptr<const Csr> get_approx_inverse_u() const { return u_inv_; }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory){};

    GKO_ENABLE_LIN_OP_FACTORY(Isai, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit Isai(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Isai>(std::move(exec))
    {}

    /**
     * Creates an Isai preconditioner from a matrix using an Isai::Factory.
     *
     * @param factory  the factory to use to create the preconditoner
     * @param factors  Composition<ValueType> of a lower triangular and an
     *                 upper triangular matrix (L and U)
     */
    explicit Isai(const Factory *factory, std::shared_ptr<const LinOp> factors)
        : EnableLinOp<Isai>(factory->get_executor(), factors->get_size()),
          parameters_{factory->get_parameters()}
    {
        auto comp = dynamic_cast<const Composition<ValueType> *>(factors.get());
        if (!comp) {
            GKO_NOT_SUPPORTED(factors);
        }
        const auto num_operators = comp->get_operators().size();
        if (num_operators != 2) {
            GKO_NOT_SUPPORTED(comp);
        }
        const auto l_factor = comp->get_operators()[0];
        const auto u_factor = comp->get_operators()[1];

        GKO_ASSERT_IS_SQUARE_MATRIX(l_factor);
        GKO_ASSERT_IS_SQUARE_MATRIX(u_factor);
        GKO_ASSERT_EQUAL_DIMENSIONS(l_factor, u_factor);

        l_inv_ = this->generate_l(l_factor.get());
        u_inv_ = this->generate_u(u_factor.get());
    }

    void apply_impl(const LinOp *b, LinOp *x) const override
    {
        cache_.prepare(l_inv_.get(), b);
        l_inv_->apply(b, cache_.intermediate.get());
        u_inv_->apply(cache_.intermediate.get(), x);
    }

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override
    {
        cache_.prepare(l_inv_.get(), b);
        l_inv_->apply(b, cache_.intermediate.get());
        u_inv_->apply(alpha, cache_.intermediate.get(), beta, x);
    }

    /**
     * Generates the approximate inverse of a lower triangular matrix
     *
     * @param to_invert_l  the source lower triangular matrix used to generate
     *                     the approximate inverse
     */
    std::shared_ptr<Csr> generate_l(const LinOp *to_invert_l);

    /**
     * Generates the approximate inverse.
     *
     * @param to_invert_u  the source upper triangular matrix used to generate
     *                     the approximate inverse
     */
    std::shared_ptr<Csr> generate_u(const LinOp *to_invert_u);

private:
    // shared_ptr, so it is easily copyable
    std::shared_ptr<Csr> l_inv_;
    std::shared_ptr<Csr> u_inv_;

    mutable struct cache_struct {
        using Dense = matrix::Dense<ValueType>;
        cache_struct() = default;
        ~cache_struct() = default;
        cache_struct(const cache_struct &) {}
        cache_struct &operator=(const cache_struct &) { return *this; }

        void prepare(const LinOp *left_factor, const LinOp *right_factor)
        {
            auto new_size =
                dim<2>{left_factor->get_size()[0], right_factor->get_size()[1]};
            if (intermediate == nullptr ||
                intermediate->get_size() != new_size) {
                intermediate =
                    Dense::create(left_factor->get_executor(), new_size);
            }
        }
        std::unique_ptr<Dense> intermediate;
    } cache_;
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_ISAI_HPP_
