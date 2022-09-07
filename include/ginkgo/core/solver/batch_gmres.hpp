/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_SOLVER_BATCH_GMRES_HPP_
#define GKO_PUBLIC_CORE_SOLVER_BATCH_GMRES_HPP_


#include <vector>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/solver/batch_solver.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>


namespace gko {
namespace solver {

/**
 * GMRES or the generalized minimal residual method is an iterative type Krylov
 * subspace method which is suitable for nonsymmetric linear systems.
 *
 * This solver solves a batch of linear systems using GMRES algorithm.
 *
 * Unless otherwise specified via the `preconditioner` factory parameter, this
 * implementation does not use any preconditioner by default.
 * The type of tolerance( absolute or relative ) and the maximum number of
 * iterations to be used in the stopping criterion can be set via the factory
 * parameters.
 *
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision>
class BatchGmres : public EnableBatchSolver<BatchGmres<ValueType>>,
                   public BatchTransposable {
    friend class EnableBatchLinOp<BatchGmres>;
    friend class EnablePolymorphicObject<BatchGmres, BatchLinOp>;

public:
    using value_type = ValueType;
    using real_type = gko::remove_complex<ValueType>;
    using transposed_type = BatchGmres<ValueType>;

    std::unique_ptr<BatchLinOp> transpose() const override;

    std::unique_ptr<BatchLinOp> conj_transpose() const override;

    /**
     * Return true as iterative solvers use the data in x as an initial guess.
     *
     * @return true as iterative solvers use the data in x as an initial guess.
     */
    bool apply_uses_initial_guess() const override { return true; }

    /**
     * Gets the restart parameter of the solver
     *
     * @return the restart number
     */
    int get_restart_number() const { return parameters_.restart; }

    /**
     * Sets the restart paramter
     *
     * @param other  the new restart number
     */
    void set_restart_number(const int& other) { parameters_.restart = other; }


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
         * Residual tolerance.
         */
        real_type GKO_FACTORY_PARAMETER_SCALAR(default_residual_tol, 1e-6);

        /**
         * Restart parameter for Gmres
         */
        int GKO_FACTORY_PARAMETER_SCALAR(restart, 10);

        /**
         * Specify type of tolerance to be considered.
         */
        ::gko::stop::batch::ToleranceType GKO_FACTORY_PARAMETER_SCALAR(
            tolerance_type, ::gko::stop::batch::ToleranceType::absolute);
    };
    GKO_ENABLE_BATCH_LIN_OP_FACTORY(BatchGmres, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit BatchGmres(std::shared_ptr<const Executor> exec)
        : EnableBatchSolver<BatchGmres>(std::move(exec))
    {}

    explicit BatchGmres(const Factory* factory,
                        std::shared_ptr<const BatchLinOp> system_matrix)
        : EnableBatchSolver<BatchGmres>(
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


#endif  // GKO_PUBLIC_CORE_SOLVER_BATCH_GMRES_HPP_
