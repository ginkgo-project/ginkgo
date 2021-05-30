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

#ifndef GKO_PUBLIC_CORE_SOLVER_BATCH_IDR_HPP_
#define GKO_PUBLIC_CORE_SOLVER_BATCH_IDR_HPP_


#include <vector>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/preconditioner/batch_preconditioner_types.hpp>
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
class BatchIdr : public EnableBatchLinOp<BatchIdr<ValueType>>,
                 public BatchTransposable,
                 public EnableBatchScaledSolver<ValueType> {
    friend class EnableBatchLinOp<BatchIdr>;
    friend class EnablePolymorphicObject<BatchIdr, BatchLinOp>;

public:
    using value_type = ValueType;
    using real_type = gko::remove_complex<ValueType>;
    using transposed_type = BatchIdr<ValueType>;

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

    /**
     * Gets the subspace dimension of the solver.
     *
     * @return the subspace Dimension*/
    size_type get_subspace_dim() const { return subspace_dim_; }

    /**
     * Sets the subspace dimension of the solver.
     *
     * @param other  the new subspace Dimension*/
    void set_subspace_dim(const size_type other) { subspace_dim_ = other; }

    /**
     * Gets the kappa parameter of the solver.
     *
     * @return the kappa parameter
     */
    real_type get_kappa() const { return kappa_; }

    /**
     * Sets the kappa parameter of the solver.
     *
     * @param other  the new kappa parameter
     */
    void set_kappa(const real_type other) { kappa_ = other; }


    /**
     * Gets the complex_subspace parameter of the solver.
     *
     * @return the complex_subspace parameter
     */
    bool get_complex_subspace() const { return complex_subspace_; }


    /**
     * Sets the complex_subspace parameter of the solver.
     *
     * @param other  the new complex_subspace parameter
     */
    void set_complex_subpsace(const bool other) { complex_subspace_ = other; }

    /**
     * Gets the deterministic parameter of the solver.
     *
     * @return the deterministic parameter
     */
    bool get_deterministic() const { return deterministic_; }

    /**
     * Sets the deterministic parameter of the solver.
     *
     * @param other  the new deterministic parameter
     */
    void set_deterministic(const bool other) { deterministic_ = other; }


    /**
     * Gets the smoothing paramter of solver
     *
     *
     * @return the smoothing paramter
     */
    bool get_smoothing() const { return smoothing_; }


    /**
     * Sets the smoothing paramter of solver
     *
     * @param other the new smoothing paramter
     */
    void set_smoothing(const bool other) { smoothing_ = other; }


    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Inner preconditioner descriptor.
         */
        preconditioner::batch::type GKO_FACTORY_PARAMETER_SCALAR(
            preconditioner, preconditioner::batch::type::none);

        /**
         * Maximum number iterations allowed.
         */
        int GKO_FACTORY_PARAMETER_SCALAR(max_iterations, 100);

        /**
         * Relative residual tolerance.
         */
        real_type GKO_FACTORY_PARAMETER_SCALAR(rel_residual_tol, 1e-6);

        /**
         * Absolute residual tolerance.
         */
        real_type GKO_FACTORY_PARAMETER_SCALAR(abs_residual_tol, 1e-11);

        /**
         * Subspace Dimension
         */
        size_type GKO_FACTORY_PARAMETER_SCALAR(subspace_dim,
                                               static_cast<size_type>(2));


        /**
         * kappa value
         *
         */
        real_type GKO_FACTORY_PARAMETER_SCALAR(kappa, 0.7);


        /**
         * If set to true, IDR will use a complex subspace S also for real
         * problems, allowing for faster convergence and better results by
         * acknowledging the influence of complex eigenvectors.
         *
         * The default is false.
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
    void apply_impl(const BatchLinOp *b, BatchLinOp *x) const override;

    void apply_impl(const BatchLinOp *alpha, const BatchLinOp *b,
                    const BatchLinOp *beta, BatchLinOp *x) const override;

    explicit BatchIdr(std::shared_ptr<const Executor> exec)
        : EnableBatchLinOp<BatchIdr>(std::move(exec))
    {}

    explicit BatchIdr(const Factory *factory,
                      std::shared_ptr<const BatchLinOp> system_matrix)
        : EnableBatchLinOp<BatchIdr>(factory->get_executor(),
                                     gko::transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()},
          system_matrix_{std::move(system_matrix)}
    {
        GKO_ASSERT_BATCH_HAS_SQUARE_MATRICES(system_matrix_);
        complex_subspace_ = parameters_.complex_subspace;
        subspace_dim_ = parameters_.subspace_dim;
        kappa_ = parameters_.kappa;
        deterministic_ = parameters_.deterministic;
        smoothing_ = parameters_.smoothing;
    }

private:
    std::shared_ptr<const BatchLinOp> system_matrix_{};
    bool complex_subspace_;
    size_type subspace_dim_;
    real_type kappa_;
    bool deterministic_;
    bool smoothing_;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_BATCH_IDR_HPP_
