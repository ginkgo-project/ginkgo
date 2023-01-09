/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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
#include <ginkgo/core/solver/batch_solver.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>


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
class BatchRichardson : public EnableBatchSolver<BatchRichardson<ValueType>>,
                        public BatchTransposable {
    friend class EnableBatchLinOp<BatchRichardson>;
    friend class EnablePolymorphicObject<BatchRichardson, BatchLinOp>;

public:
    using value_type = ValueType;
    using real_type = gko::remove_complex<ValueType>;
    using transposed_type = BatchRichardson<ValueType>;

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
         * Preconditioner factory.
         */
        std::shared_ptr<const BatchLinOpFactory> GKO_FACTORY_PARAMETER_SCALAR(
            preconditioner, nullptr);

        /**
         * Already generated preconditioner. If one is provided, the factory
         * `preconditioner` will be ignored.
         *
         * Note that if scaling is requested (@sa left_scaling_op,
         * @sa right_scaling_op), this is assumed to be generated from the
         * scaled matrix.
         */
        std::shared_ptr<const BatchLinOp> GKO_FACTORY_PARAMETER_SCALAR(
            generated_preconditioner, nullptr);

        std::shared_ptr<const BatchLinOp> GKO_FACTORY_PARAMETER_SCALAR(
            left_scaling_op, nullptr);

        std::shared_ptr<const BatchLinOp> GKO_FACTORY_PARAMETER_SCALAR(
            right_scaling_op, nullptr);

        /**
         * Maximum number iterations allowed.
         */
        int GKO_FACTORY_PARAMETER_SCALAR(default_max_iterations, 100);

        /**
         * Relative residual tolerance.
         */
        real_type GKO_FACTORY_PARAMETER_SCALAR(default_residual_tol, 1e-6);

        /**
         * Tolerance type.
         */
        gko::stop::batch::ToleranceType GKO_FACTORY_PARAMETER_SCALAR(
            tolerance_type, gko::stop::batch::ToleranceType::relative);

        /**
         * Relaxation factor for Richardson iteration.
         */
        real_type GKO_FACTORY_PARAMETER_SCALAR(relaxation_factor, real_type{1});
    };
    GKO_ENABLE_BATCH_LIN_OP_FACTORY(BatchRichardson, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit BatchRichardson(std::shared_ptr<const Executor> exec)
        : EnableBatchSolver<BatchRichardson>(std::move(exec))
    {}

    explicit BatchRichardson(const Factory* factory,
                             std::shared_ptr<const BatchLinOp> system_matrix)
        : EnableBatchSolver<BatchRichardson>(
              factory->get_executor(), std::move(system_matrix),
              detail::extract_common_batch_params(factory->get_parameters())),
          parameters_{factory->get_parameters()}
    {}

private:
    void solver_apply(const BatchLinOp* b, BatchLinOp* x,
                      BatchInfo* const info) const override;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_BATCH_RICHARDSON_HPP_
