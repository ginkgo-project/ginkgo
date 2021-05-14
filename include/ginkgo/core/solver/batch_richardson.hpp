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

#ifndef GKO_PUBLIC_CORE_SOLVER_BATCH_RICHARDSON_HPP_
#define GKO_PUBLIC_CORE_SOLVER_BATCH_RICHARDSON_HPP_


#include <vector>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/preconditioner/batch_preconditioner_types.hpp>


namespace gko {
namespace solver {


/**
 * The (preconditioned) Richardson solver is an iterative method that uses
 * a preconditioner to approximate the error of the current solution via the
 * current (preconditioned) residual.
 * This solver applies the Richardson iteration to a batch of linear systems.
 *
 * ```
 * solution = initial_guess
 * while not converged:
 *     residual = b - A solution
 *     error = preconditioner(A, residual)
 *     solution = solution + relaxation_factor * error
 * ```
 *
 * Unless otherwise specified via the `preconditioner` factory parameter, this
 * implementation uses the Jacobi preconditioner as the default preconditioner.
 * The only stopping criterion currently available is controlled by the
 * `max_iterations` and `rel_residual_tol` factory parameters. The solver is
 * stopped whrn the maximum iterations are reached, or the relative residual
 * is smaller than the specified tolerance.
 *
 * @sa Ir
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision>
class BatchRichardson : public EnableBatchLinOp<BatchRichardson<ValueType>>,
                        public BatchTransposable,
                        public EnableBatchScaledSolver<ValueType> {
    friend class EnableBatchLinOp<BatchRichardson>;
    friend class EnablePolymorphicObject<BatchRichardson, BatchLinOp>;

public:
    using value_type = ValueType;
    using real_type = gko::remove_complex<ValueType>;
    using transposed_type = BatchRichardson<ValueType>;

    /**
     * Returns the system operator (matrix) of the linear system.
     *
     * @return the system operator (matrix)
     */
    std::shared_ptr<const BatchLinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    std::unique_ptr<BatchLinOp> transpose() const override;

    std::unique_ptr<BatchLinOp> conj_transpose() const override;

    /**
     * Return true as iterative solvers use the data in x as an initial guess.
     *
     * @return true as iterative solvers use the data in x as an initial guess.
     */
    bool apply_uses_initial_guess() const override { return true; }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Inner preconditioner descriptor.
         */
        preconditioner::batch::type GKO_FACTORY_PARAMETER_SCALAR(
            preconditioner, preconditioner::batch::type::jacobi);

        /**
         * Maximum number iterations allowed.
         */
        int GKO_FACTORY_PARAMETER_SCALAR(max_iterations, 100);

        /**
         * Relative residual tolerance.
         */
        real_type GKO_FACTORY_PARAMETER_SCALAR(rel_residual_tol, 1e-6);

        /**
         * Relaxation factor for Richardson iteration.
         */
        real_type GKO_FACTORY_PARAMETER_SCALAR(relaxation_factor, real_type{1});
    };
    GKO_ENABLE_BATCH_LIN_OP_FACTORY(BatchRichardson, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const BatchLinOp *b, BatchLinOp *x) const override;

    void apply_impl(const BatchLinOp *alpha, const BatchLinOp *b,
                    const BatchLinOp *beta, BatchLinOp *x) const override;

    explicit BatchRichardson(std::shared_ptr<const Executor> exec)
        : EnableBatchLinOp<BatchRichardson>(std::move(exec))
    {}

    explicit BatchRichardson(const Factory *factory,
                             std::shared_ptr<const BatchLinOp> system_matrix)
        : EnableBatchLinOp<BatchRichardson>(factory->get_executor(),
                                            system_matrix->get_size()),
          parameters_{factory->get_parameters()},
          system_matrix_{std::move(system_matrix)}
    {
        GKO_ASSERT_BATCH_HAS_SQUARE_MATRICES(system_matrix_);
    }

private:
    std::shared_ptr<const BatchLinOp> system_matrix_{};
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_BATCH_RICHARDSON_HPP_
