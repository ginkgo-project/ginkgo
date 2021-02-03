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

#ifndef GKO_PUBLIC_CORE_SOLVER_IDR_HPP_
#define GKO_PUBLIC_CORE_SOLVER_IDR_HPP_


#include <random>
#include <typeinfo>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
/**
 * @brief The ginkgo Solver namespace.
 *
 * @ingroup solvers
 */
namespace solver {


/**
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
 * @tparam ValueType  precision of the elements of the system matrix.
 *
 * @ingroup idr
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Idr : public EnableLinOp<Idr<ValueType>>,
            public Preconditionable,
            public Transposable {
    friend class EnableLinOp<Idr>;
    friend class EnablePolymorphicObject<Idr, LinOp>;

public:
    using value_type = ValueType;
    using transposed_type = Idr<ValueType>;

    /**
     * Gets the system operator (matrix) of the linear system.
     *
     * @return the system operator (matrix)
     */
    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Return true as iterative solvers use the data in x as an initial guess.
     *
     * @return true as iterative solvers use the data in x as an initial guess.
     */
    bool apply_uses_initial_guess() const override { return true; }

    /**
     * Gets the stopping criterion factory of the solver.
     *
     * @return the stopping criterion factory
     */
    std::shared_ptr<const stop::CriterionFactory> get_stop_criterion_factory()
        const
    {
        return stop_criterion_factory_;
    }

    /**
     * Sets the stopping criterion of the solver.
     *
     * @param other  the new stopping criterion factory
     */
    void set_stop_criterion_factory(
        std::shared_ptr<const stop::CriterionFactory> other)
    {
        stop_criterion_factory_ = std::move(other);
    }

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
    remove_complex<ValueType> get_kappa() const { return kappa_; }

    /**
     * Sets the kappa parameter of the solver.
     *
     * @param other  the new kappa parameter
     */
    void set_kappa(const remove_complex<ValueType> other) { kappa_ = other; }

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

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Criterion factories.
         */
        std::vector<std::shared_ptr<const stop::CriterionFactory>>
            GKO_FACTORY_PARAMETER_VECTOR(criteria, nullptr);

        /**
         * Preconditioner factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER_SCALAR(
            preconditioner, nullptr);

        /**
         * Already generated preconditioner. If one is provided, the factory
         * `preconditioner` will be ignored.
         */
        std::shared_ptr<const LinOp> GKO_FACTORY_PARAMETER_SCALAR(
            generated_preconditioner, nullptr);

        /**
         * Dimension of the subspace S. Determines how many intermediate
         * residuals are computed in each iteration.
         */
        size_type GKO_FACTORY_PARAMETER_SCALAR(subspace_dim, 2u);

        /**
         * Threshold to determine if Av_n and v_n are too close to being
         * perpendicular.
         * This is considered to be the case if
         * $|(Av_n)^H * v_n / (norm(Av_n) * norm(v_n))| < kappa$
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(kappa, 0.7);

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
         * If set to true, IDR will use a complex subspace S also for real
         * problems, allowing for faster convergence and better results by
         * acknowledging the influence of complex eigenvectors.
         *
         * The default is false.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(complex_subspace, false);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Idr, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    template <typename SubspaceType>
    void iterate(const LinOp *b, LinOp *x) const;

    explicit Idr(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Idr>(std::move(exec))
    {}

    explicit Idr(const Factory *factory,
                 std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Idr>(factory->get_executor(),
                           gko::transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()},
          system_matrix_{std::move(system_matrix)}
    {
        if (parameters_.generated_preconditioner) {
            GKO_ASSERT_EQUAL_DIMENSIONS(parameters_.generated_preconditioner,
                                        this);
            set_preconditioner(parameters_.generated_preconditioner);
        } else if (parameters_.preconditioner) {
            set_preconditioner(
                parameters_.preconditioner->generate(system_matrix_));
        } else {
            set_preconditioner(matrix::Identity<ValueType>::create(
                this->get_executor(), this->get_size()[0]));
        }
        stop_criterion_factory_ =
            stop::combine(std::move(parameters_.criteria));
        subspace_dim_ = parameters_.subspace_dim;
        kappa_ = parameters_.kappa;
        deterministic_ = parameters_.deterministic;
        complex_subspace_ = parameters_.complex_subspace;
    }

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    std::shared_ptr<const stop::CriterionFactory> stop_criterion_factory_{};
    size_type subspace_dim_;
    remove_complex<ValueType> kappa_;
    bool deterministic_;
    bool complex_subspace_;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_IDR_HPP_
