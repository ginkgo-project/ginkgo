// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_HYPRE_BOOMERAMG_HPP_
#define GKO_PUBLIC_CORE_SOLVER_HYPRE_BOOMERAMG_HPP_


#include <ginkgo/config.hpp>


#if GKO_HAVE_HYPRE


#include <memory>

#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace experimental {
namespace solver {


/**
 * Provides a wrapper around hypre's BoomerAMG preconditioner.
 *
 * This wraps a single AMG cycle (V, W, or F) as a Ginkgo LinOp, suitable
 * for use as a preconditioner. The setup phase (coarsening, interpolation,
 * smoother construction) happens during generation, and each apply()
 * performs one AMG cycle.
 *
 * This wrapper runs sequentially using MPI_COMM_SELF. It does not support
 * distributed matrices.
 *
 * @note This solver only runs on the host (CPU). If the executor is not
 *       a ReferenceExecutor or OmpExecutor, data will be copied to the
 *       host for the hypre calls.
 *
 * @tparam ValueType  the type used to store values of the system matrix.
 *                    Only double is natively supported; float inputs are
 *                    converted to double internally.
 */
template <typename ValueType = double>
class HypreBoomerAmg : public EnableLinOp<HypreBoomerAmg<ValueType>> {
    friend class EnableLinOp<HypreBoomerAmg>;
    friend class EnablePolymorphicObject<HypreBoomerAmg, LinOp>;

    static_assert(!is_complex_s<ValueType>::value,
                  "HypreBoomerAmg does not support complex value types. "
                  "Use float or double.");

public:
    using value_type = ValueType;
    using matrix_type = matrix::Csr<ValueType, int32>;

    class Factory;

    struct parameters_type
        : public enable_parameters_type<parameters_type, Factory> {
        /**
         * AMG cycle type.
         *   1 = V-cycle (default)
         *   2 = W-cycle
         */
        int GKO_FACTORY_PARAMETER_SCALAR(cycle_type, 1);

        /**
         * Coarsening algorithm.
         *   0  = CLJP
         *   3  = classical Ruge-Stueben
         *   6  = Falgout (default in older hypre)
         *   8  = PMIS
         *   10 = HMIS (default)
         *   21 = CGC
         */
        int GKO_FACTORY_PARAMETER_SCALAR(coarsening_type, 10);

        /**
         * Strength-of-connection threshold (theta).
         * A connection a_ij is strong if -a_ij >= theta * max_{k!=i}(-a_ik).
         * Default 0.25 matches hypre's BoomerAMG default.
         */
        double GKO_FACTORY_PARAMETER_SCALAR(strength_threshold, 0.25);

        /**
         * Relaxation (smoother) type.
         *   0  = Jacobi
         *   3  = hybrid forward Gauss-Seidel
         *   4  = hybrid backward Gauss-Seidel
         *   6  = hybrid symmetric Gauss-Seidel (default)
         *   8  = hybrid symmetric l1-Gauss-Seidel
         *   13 = forward l1-Gauss-Seidel
         *   14 = backward l1-Gauss-Seidel
         *   16 = Chebyshev
         *   18 = l1-Jacobi
         */
        int GKO_FACTORY_PARAMETER_SCALAR(smoother_type, 6);

        /**
         * Number of smoother sweeps per level.
         */
        int GKO_FACTORY_PARAMETER_SCALAR(num_sweeps, 1);

        /**
         * Interpolation type.
         *   0  = classical modified interpolation (default)
         *   3  = direct interpolation
         *   4  = multipass interpolation
         *   6  = extended+i interpolation
         *   13 = FF1 interpolation
         *   14 = extended interpolation
         */
        int GKO_FACTORY_PARAMETER_SCALAR(interpolation_type, 0);

        /**
         * Maximum number of AMG levels.
         */
        int GKO_FACTORY_PARAMETER_SCALAR(max_levels, 25);
    };
    GKO_ENABLE_LIN_OP_FACTORY(HypreBoomerAmg, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    HypreBoomerAmg(const HypreBoomerAmg&);

    HypreBoomerAmg(HypreBoomerAmg&&) noexcept;

    HypreBoomerAmg& operator=(const HypreBoomerAmg&);

    HypreBoomerAmg& operator=(HypreBoomerAmg&&) noexcept;

    ~HypreBoomerAmg() override;

protected:
    explicit HypreBoomerAmg(std::shared_ptr<const Executor> exec);

    HypreBoomerAmg(const Factory* factory,
                   std::shared_ptr<const LinOp> system_matrix);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    struct hypre_state;

    void setup_hypre();

    std::unique_ptr<hypre_state> state_;
    std::shared_ptr<const matrix::Csr<double, int32>> system_matrix_;

    mutable detail::DenseCache<ValueType> host_buffer_;
    mutable detail::DenseCache<ValueType> host_x_buffer_;
};


}  // namespace solver
}  // namespace experimental
}  // namespace gko


#endif  // GKO_HAVE_HYPRE


#endif  // GKO_PUBLIC_CORE_SOLVER_HYPRE_BOOMERAMG_HPP_
