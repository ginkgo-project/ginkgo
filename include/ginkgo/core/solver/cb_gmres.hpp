// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_CB_GMRES_HPP_
#define GKO_PUBLIC_CORE_SOLVER_CB_GMRES_HPP_


#include <memory>
#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace solver {


namespace cb_gmres {


/**
 * Describes the storage precision that is used in CB-GMRES.
 *
 * The storage precision is described relative to the ValueType:
 * - keep: The storage precision is the same as the ValueType.
 * - reduce1: The storage type is the ValueType reduced in precision once,
 *            for example, ValueType == double -> storage precision == float
 * - reduce2: ValueType precision is reduced twice
 * - integer: The storage precision is an integer of the same size of
 *            ValueType. Note that complex values are not supported.
 * - ireduce1: The storage precision is an integer of the same size as
 *             a reduced ValueType.
 * - ireduce2: The storage precision is an integer of the same size as
 *             a twice reduced ValueType.
 *
 * Precision reduction works as follows:
 * - double -> float -> half -> half -> ... (half is the lowest supported
 *   precision)
 * - std::complex<double> -> std::complex<float> -> std::complex<half>
 *   -> std::complex<half> ... (std::complex<half> is the lowest supported
 *   precision)
 *
 * To integer conversions:
 * - double -> int64
 * - float -> int32
 * - half -> int16
 */
enum class storage_precision {
    keep,
    reduce1,
    reduce2,
    integer,
    ireduce1,
    ireduce2
};


}  // namespace cb_gmres


/**
 * CB-GMRES or the compressed basis generalized minimal residual method is an
 * iterative type Krylov subspace method which is suitable for nonsymmetric
 * linear systems.
 *
 * The implementation in Ginkgo makes use of the merged kernel to make the best
 * use of data locality. The inner operations in one iteration of CB-GMRES
 * are merged into 2 separate steps. Classical Gram-Schmidt with
 * reorthogonalization is used.
 *
 * The Krylov basis can be stored in reduced precision (compressed) to reduce
 * memory accesses, while all computations (including Krylov basis operations)
 * are performed in the same arithmetic precision ValueType. By default, the
 * Krylov basis are stored in one precision lower than ValueType.
 *
 * Mathematically CB-GMRES is the same algorithm as standard
 * @ref gko::solver::Gmres "GMRES" — the Arnoldi process produces an
 * orthonormal basis \f$ V_m \f$ and an upper Hessenberg
 * \f$ \bar H_m \f$ with
 * \f$ A V_m = V_{m+1} \bar H_m \f$, and the iterate is chosen as
 * \f[
 *   x_m = x_0 + V_m y_m,
 *   \qquad
 *   y_m = \arg\min_{y} \| \beta e_1 - \bar H_m y \|_2.
 * \f]
 * The "compressed basis" variant stores the columns of \f$ V_m \f$ in a
 * precision lower than ValueType: each basis vector is cast down to the
 * storage type when written and cast back up when read. The Hessenberg
 * matrix, the Givens rotations, and the working residual all stay in
 * full ValueType precision, so the final approximation has the same
 * accuracy as standard GMRES while the basis storage and the bandwidth
 * needed to apply it are reduced. See the `storage_precision` enum for
 * the supported compression options.
 *
 * @tparam ValueType  the arithmetic precision and the precision of matrix
 *                    elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class CbGmres : public LinOp,
                public EnablePreconditionedIterativeSolver<ValueType,
                                                           CbGmres<ValueType>> {
    GKO_ASSERT_SUPPORTED_VALUE_TYPE;

public:
    using value_type = ValueType;

    /**
     * Returns the Krylov dimension.
     *
     * @return the Krylov dimension
     */
    size_type get_krylov_dim() const { return parameters_.krylov_dim; }

    /**
     * Sets the Krylov dimension
     *
     * @param other  the new Krylov dimension
     */
    void set_krylov_dim(size_type other) { parameters_.krylov_dim = other; }

    /**
     * Returns the restart ratio.
     *
     * @return the restart ratio
     */
    remove_complex<ValueType> get_restart_ratio() const
    {
        return parameters_.restart_ratio;
    }

    /**
     * Sets the restart ratio.
     *
     * @param other  the new restart ratio
     */
    void set_restart_ratio(remove_complex<ValueType> other)
    {
        parameters_.restart_ratio = other;
    }

    /**
     * Returns the maximum number of re-orthogonalization passes.
     *
     * @return the maximum number of re-orthogonalization passes
     */
    size_type get_max_reorth_steps() const
    {
        return parameters_.max_reorth_steps;
    }

    /**
     * Sets the maximum number of re-orthogonalization passes.
     *
     * @param other  the new maximum
     */
    void set_max_reorth_steps(size_type other)
    {
        parameters_.max_reorth_steps = other;
    }

    /**
     * Returns the storage precision used internally.
     *
     * @return the storage precision used internally
     */
    cb_gmres::storage_precision get_storage_precision() const
    {
        return parameters_.storage_precision;
    }

    class Factory;

    struct parameters_type
        : enable_preconditioned_iterative_solver_factory_parameters<
              parameters_type, Factory> {
        /**
         * Determines which storage type is used.
         */
        cb_gmres::storage_precision GKO_FACTORY_PARAMETER_SCALAR(
            storage_precision, cb_gmres::storage_precision::reduce1);

        /**
         * Krylov dimension factory.
         *
         * When restart_ratio is zero this is the restart length: a cycle always
         * ends after krylov_dim steps. When restart_ratio is positive this acts
         * instead as a cap on the size of the Krylov basis, and a cycle ends at
         * whichever of the two criteria is met first.
         */
        size_type GKO_FACTORY_PARAMETER_SCALAR(krylov_dim, 100u);

        /**
         * Restart ratio.
         *
         * A cycle ends once the residual has fallen to this fraction of its
         * value at the start of the cycle, that is, once
         * \f$ \|r_i - A d_i\| \leq \text{restart\_ratio} \cdot \|r_i\| \f$
         * holds for every right-hand side still running, with \f$r_i\f$ the
         * residual at the start of the cycle and \f$d_i\f$ the correction
         * built so far. This bounds the normwise backward error of the
         * correction; a cycle cut short by krylov_dim carries no such bound.
         *
         * It is a per-cycle reduction, not a convergence tolerance, so useful
         * values are far larger than a stopping criterion's. It must lie in
         * [0, 1) and must exceed the reduction a krylov_dim-step cycle
         * achieves, or it never fires: use 1e-3 to 1e-1 when a cycle gains
         * orders of magnitude, and around 0.9 when it gains only a few
         * percent. Zero, the default, disables the criterion.
         *
         * For the criterion and its analysis, see sec. 4.2 of
         * <a href="https://doi.org/10.1093/imanum/draf049">A Modular Framework
         * for the Backward Error Analysis of GMRES</a>.
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(restart_ratio,
                                                               0.0);

        /**
         * Maximum number of re-orthogonalization passes per Arnoldi step.
         *
         * The Arnoldi step uses classical Gram-Schmidt and re-orthogonalizes
         * only when the norm of the new vector has dropped by more than a
         * factor \f$\sqrt{2}\f$, which indicates cancellation. This bounds how
         * many such passes are taken. Zero leaves plain classical
         * Gram-Schmidt, and the default of two reproduces the published
         * algorithm.
         */
        size_type GKO_FACTORY_PARAMETER_SCALAR(max_reorth_steps, 2u);
    };

    GKO_ENABLE_LIN_OP_FACTORY(CbGmres, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /**
     * Create the parameters from the property_tree.
     * Because this is directly tied to the specific type, the value/index type
     * settings within config are ignored and type_descriptor is only used
     * for children configs.
     *
     * @param config  the property tree for setting
     * @param context  the registry
     * @param td_for_child  the type descriptor for children configs. The
     *                      default uses the value type of this class.
     *
     * @return parameters
     */
    static parameters_type parse(const config::pnode& config,
                                 const config::registry& context,
                                 const config::type_descriptor& td_for_child =
                                     config::make_type_descriptor<ValueType>());

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_dense_impl(const matrix::Dense<ValueType>* b,
                          matrix::Dense<ValueType>* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    explicit CbGmres(std::shared_ptr<const Executor> exec)
        : LinOp(std::move(exec))
    {}

    explicit CbGmres(const Factory* factory,
                     std::shared_ptr<const LinOp> system_matrix)
        : LinOp(factory->get_executor(), transpose(system_matrix->get_size())),
          EnablePreconditionedIterativeSolver<ValueType, CbGmres<ValueType>>{
              std::move(system_matrix), factory->get_parameters()},
          parameters_{factory->get_parameters()}
    {
        GKO_THROW_IF_INVALID(
            parameters_.restart_ratio >= zero<remove_complex<ValueType>>() &&
                parameters_.restart_ratio < one<remove_complex<ValueType>>(),
            "restart_ratio must lie in [0, 1); zero disables the criterion");
    }
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_CB_GMRES_HPP_
