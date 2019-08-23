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

#ifndef GKO_CORE_SOLVER_UPPER_TRS_HPP_
#define GKO_CORE_SOLVER_UPPER_TRS_HPP_


#include <memory>
#include <utility>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/identity.hpp>


namespace gko {
namespace solver {


/**
 * UpperTrs is the triangular solver which solves the system U x = b, when U is
 * an upper triangular matrix. It works best when passing in a matrix in CSR
 * format. If the matrix is not in CSR, then the generate step converts it into
 * a CSR matrix. The generation fails if the matrix is not convertible to CSR.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indices
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class UpperTrs : public EnableLinOp<UpperTrs<ValueType, IndexType>>,
                 public Preconditionable {
    friend class EnableLinOp<UpperTrs>;
    friend class EnablePolymorphicObject<UpperTrs, LinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;

    /**
     * Gets the system operator (CSR matrix) of the linear system.
     *
     * @return the system operator (CSR matrix)
     */
    std::shared_ptr<const matrix::Csr<ValueType, IndexType>> get_system_matrix()
        const
    {
        return system_matrix_;
    }

    /**
     * Returns the preconditioner operator used by the solver.
     *
     * @return the preconditioner operator used by the solver
     */
    std::shared_ptr<const LinOp> get_preconditioner() const override
    {
        return preconditioner_;
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Preconditioner factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER(
            preconditioner, nullptr);

        /**
         * Number of right hand sides.
         *
         * @note: This value must be same as to be passed to the b vector of
         * apply.
         */
        gko::size_type GKO_FACTORY_PARAMETER(num_rhs, 1u);
    };
    GKO_ENABLE_LIN_OP_FACTORY(UpperTrs, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    /**
     * Generates the analysis structure from the system matrix and the right
     * hand side(only dimensional info needed) needed for the level solver.
     */
    void generate();

    explicit UpperTrs(std::shared_ptr<const Executor> exec)
        : EnableLinOp<UpperTrs>(std::move(exec))
    {}

    explicit UpperTrs(const Factory *factory,
                      std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<UpperTrs>(factory->get_executor(),
                                transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()},
          system_matrix_{}
    {
        using CsrMatrix = matrix::Csr<ValueType, IndexType>;

        GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
        // This is needed because it does not make sense to call the copy and
        // convert if the existing matrix is empty.
        const auto exec = this->get_executor();
        if (!system_matrix->get_size()) {
            system_matrix_ = CsrMatrix::create(exec);
        } else {
            system_matrix_ =
                copy_and_convert_to<CsrMatrix>(exec, system_matrix);
        }
        if (parameters_.preconditioner) {
            preconditioner_ =
                parameters_.preconditioner->generate(system_matrix_);
        } else {
            preconditioner_ = matrix::Identity<ValueType>::create(
                this->get_executor(), this->get_size()[0]);
        }
        this->generate();
    }

private:
    std::shared_ptr<const matrix::Csr<ValueType, IndexType>> system_matrix_{};
    std::shared_ptr<const LinOp> preconditioner_{};
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_CORE_SOLVER_UPPER_TRS_HPP
