// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_NULLSPACE_PROJECTION_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_NULLSPACE_PROJECTION_HPP_


#include <memory>

#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>
#include <ginkgo/core/solver/nullspace.hpp>


namespace gko {
namespace preconditioner {


/**
 * NullspaceProjection wraps a preconditioner \f$ M \f$ with projections onto
 * the complement of a nullspace, \f$ z = P M Q r \f$. \f$ P \f$ removes the
 * nullspace from the preconditioned vector, which keeps the search
 * directions of a Krylov solver, and thus its solution, orthogonal to the
 * nullspace. The optional \f$ Q \f$ removes the left nullspace from the input.
 * With both, \f$ P M P \f$ is symmetric for a symmetric \f$ M \f$, which
 * solvers like Minres need on ill-conditioned problems or when iterated past
 * convergence.
 *
 * Iterative solvers that support nullspaces (see solver::nullspace_traits)
 * wrap their preconditioner in a NullspaceProjection when their `nullspace`
 * parameter is set. It can also be used as the preconditioner of any solver
 * directly; then the right-hand side and the initial guess need to be
 * projected by the user, see solver::Nullspace::project.
 *
 * A constant-only nullspace is adapted to the size of the system matrix, so
 * the factory can be used for matrices of any size.
 *
 * @tparam ValueType  the value type of the vectors
 *
 * @ingroup precond
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class NullspaceProjection
    : public LinOp,
      public EnableCloneable<NullspaceProjection<ValueType>>,
      public Transposable {
    friend class EnableCloneable<NullspaceProjection>;
    GKO_ASSERT_SUPPORTED_VALUE_TYPE;

public:
    using EnableCloneable<NullspaceProjection>::convert_to;
    using EnableCloneable<NullspaceProjection>::move_to;

    using value_type = ValueType;

    /** @return the wrapped preconditioner */
    std::shared_ptr<const LinOp> get_preconditioner() const;

    /**
     * @return the nullspace removed after the preconditioner, or nullptr if
     *         there is none
     */
    std::shared_ptr<const solver::Nullspace<ValueType>> get_nullspace() const;

    /**
     * @return the left nullspace removed before the preconditioner, or nullptr
     *         if there is none
     */
    std::shared_ptr<const solver::Nullspace<ValueType>> get_left_nullspace()
        const;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    NullspaceProjection& operator=(const NullspaceProjection& other);

    NullspaceProjection& operator=(NullspaceProjection&& other);

    NullspaceProjection(const NullspaceProjection& other);

    NullspaceProjection(NullspaceProjection&& other);

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * The preconditioner \f$ M \f$ to wrap. By default, the identity.
         */
        std::shared_ptr<const LinOpFactory> GKO_DEFERRED_FACTORY_PARAMETER(
            preconditioner);

        /**
         * Already generated preconditioner. If one is provided, the factory
         * `preconditioner` will be ignored.
         */
        std::shared_ptr<const LinOp> GKO_FACTORY_PARAMETER_SCALAR(
            generated_preconditioner, nullptr);

        /**
         * The nullspace removed from the preconditioned vector, given as a
         * solver::Nullspace<ValueType>. By default, none.
         */
        std::shared_ptr<const LinOp> GKO_FACTORY_PARAMETER_SCALAR(nullspace,
                                                                  nullptr);

        /**
         * The left nullspace removed from the input before the
         * preconditioner, given as a solver::Nullspace<ValueType>. For
         * symmetric problems, pass the same object as `nullspace`. By default,
         * none, so only the output is projected.
         */
        std::shared_ptr<const LinOp> GKO_FACTORY_PARAMETER_SCALAR(
            left_nullspace, nullptr);
    };
    GKO_ENABLE_LIN_OP_FACTORY(NullspaceProjection, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /**
     * Create the parameters from the property_tree. Because this is directly
     * tied to the specific type, the value type in the property tree is
     * ignored. The nullspaces and the generated preconditioner are read from
     * the registry.
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
    explicit NullspaceProjection(std::shared_ptr<const Executor> exec);

    NullspaceProjection(const Factory* factory,
                        std::shared_ptr<const LinOp> system_matrix);

    NullspaceProjection(
        std::shared_ptr<const Executor> exec, dim<2> size,
        std::shared_ptr<const LinOp> preconditioner,
        std::shared_ptr<const solver::Nullspace<ValueType>> nullspace,
        std::shared_ptr<const solver::Nullspace<ValueType>> left_nullspace);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    // a vector of the same kind as the input, which is not copied along with
    // the object
    struct vector_cache {
        vector_cache() = default;
        vector_cache(const vector_cache&) {}
        vector_cache(vector_cache&&) noexcept {}
        vector_cache& operator=(const vector_cache&) { return *this; }
        vector_cache& operator=(vector_cache&&) noexcept { return *this; }

        mutable std::unique_ptr<LinOp> vec;
    };

    std::shared_ptr<const LinOp> preconditioner_;
    std::shared_ptr<const solver::Nullspace<ValueType>> nullspace_;
    std::shared_ptr<const solver::Nullspace<ValueType>> left_nullspace_;
    vector_cache projected_input_;
    vector_cache result_;
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_NULLSPACE_PROJECTION_HPP_
