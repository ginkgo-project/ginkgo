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

#ifndef GKO_PUBLIC_CORE_SOLVER_BATCH_IDR_HPP_
#define GKO_PUBLIC_CORE_SOLVER_BATCH_IDR_HPP_


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
 *
 * IDR(s) is an efficient method for solving large nonsymmetric systems of
 * linear equations. The implemented version is the one presented in the
 * paper "Algorithm 913: An elegant IDR(s) variant that efficiently exploits
 * biorthogonality properties" by M. B. Van Gijzen and P. Sonneveld.
 *
 * The method is based on the induced dimension reduction theorem which
 * provides a way to construct subsequent residuals that lie in a sequence
 * of shrinking subspaces. These subspaces are spanned by s vectors which are
 * first generated randomly and then orthonormalized. They are stored in
 * a dense matrix.
 *
 * This solver solves a batch of linear systems using the IDR(s) algorithm.
 *
 * Unless otherwise specified via the `preconditioner` factory parameter, this
 * implementation does not use any preconditioner by default.
 * The type of tolerance( absolute or relative ) and the maximum number of
 * iterations to be used in the stopping criterion can be set via the factory
 * parameters.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision>
class BatchIdr : public EnableBatchSolver<BatchIdr<ValueType>>,
                 public BatchTransposable {
    friend class EnableBatchLinOp<BatchIdr>;
    friend class EnablePolymorphicObject<BatchIdr, BatchLinOp>;

public:
    using value_type = ValueType;
    using real_type = gko::remove_complex<ValueType>;
    using transposed_type = BatchIdr<ValueType>;

    std::unique_ptr<BatchLinOp> transpose() const override;

    std::unique_ptr<BatchLinOp> conj_transpose() const override;

    /**
     * Return true as iterative solvers use the data in x as an initial guess.
     *
     * @return true as iterative solvers use the data in x as an initial guess.
     */
    bool apply_uses_initial_guess() const override { return true; }

    /**
     * Gets the subspace dimension of the solver.
     *
     * @return the subspace Dimension*/
    size_type get_subspace_dim() const { return parameters_.subspace_dim; }

    /**
     * Sets the subspace dimension of the solver.
     *
     * @param other  the new subspace Dimension*/
    void set_subspace_dim(const size_type other)
    {
        parameters_.subspace_dim = other;
    }

    /**
     * Gets the kappa parameter of the solver.
     *
     * @return the kappa parameter
     */
    real_type get_kappa() const { return parameters_.kappa; }

    /**
     * Sets the kappa parameter of the solver.
     *
     * @param other  the new kappa parameter
     */
    void set_kappa(const real_type other) { parameters_.kappa = other; }


    /**
     * Gets the complex_subspace parameter of the solver.
     *
     * @return the complex_subspace parameter
     */
    bool get_complex_subspace() const { return parameters_.complex_subspace; }


    /**
     * Sets the complex_subspace parameter of the solver.
     *
     * @param other  the new complex_subspace parameter
     */
    void set_complex_subspace(const bool other)
    {
        parameters_.complex_subspace = other;
    }

    /**
     * Gets the deterministic parameter of the solver.
     *
     * @return the deterministic parameter
     */
    bool get_deterministic() const { return parameters_.deterministic; }

    /**
     * Sets the deterministic parameter of the solver.
     *
     * @param other  the new deterministic parameter
     */
    void set_deterministic(const bool other)
    {
        parameters_.deterministic = other;
    }


    /**
     * Gets the smoothing paramter of solver
     *
     *
     * @return the smoothing paramter
     */
    bool get_smoothing() const { return parameters_.smoothing; }


    /**
     * Sets the smoothing paramter of solver
     *
     * @param other the new smoothing paramter
     */
    void set_smoothing(const bool other) { parameters_.smoothing = other; }


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
         *
         * @sa tolerance_type
         */
        real_type GKO_FACTORY_PARAMETER_SCALAR(default_residual_tol, 1e-8);

        /**
         * Subspace Dimension
         */
        size_type GKO_FACTORY_PARAMETER_SCALAR(subspace_dim,
                                               static_cast<size_type>(2));

        /**
         * kappa value
         */
        real_type GKO_FACTORY_PARAMETER_SCALAR(kappa, 0.7);

        /**
         * If set to true, IDR is supposed to use a complex subspace S also for
         * real problems, allowing for faster convergence and better results by
         * acknowledging the influence of complex eigenvectors.
         *
         * The default is false.
         * Currently, the option of having complex subspace for real matrices
         * is not supported.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(complex_subspace, false);

        /**
         * To enable smoothing
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(smoothing, true);

        /**
         * If set to true, the vectors spanning the subspace S are chosen
         * deterministically. This is mostly needed for testing purposes.
         *
         * Note: If 'deterministic' is set to true, the subspace vectors are
         * generated in serial on the CPU, which can be very slow.
         *
         * The default behaviour is to choose the subspace vectors randomly.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(deterministic, false);

        /**
         * To specify which tolerance is to be considered.
         *
         */
        ::gko::stop::batch::ToleranceType GKO_FACTORY_PARAMETER_SCALAR(
            tolerance_type, ::gko::stop::batch::ToleranceType::absolute);
    };
    GKO_ENABLE_BATCH_LIN_OP_FACTORY(BatchIdr, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit BatchIdr(std::shared_ptr<const Executor> exec)
        : EnableBatchSolver<BatchIdr>(std::move(exec))
    {}

    explicit BatchIdr(const Factory* factory,
                      std::shared_ptr<const BatchLinOp> system_matrix)
        : EnableBatchSolver<BatchIdr>(
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


#endif  // GKO_PUBLIC_CORE_SOLVER_BATCH_IDR_HPP_
