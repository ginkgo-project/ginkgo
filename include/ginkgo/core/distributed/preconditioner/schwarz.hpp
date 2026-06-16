// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_SCHWARZ_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_SCHWARZ_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/distributed/vector_cache.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/solver/workspace.hpp>


namespace gko {
namespace experimental {
namespace distributed {
/**
 * @brief The Preconditioner namespace.
 *
 * @ingroup precond
 */
namespace preconditioner {


/**
 * A Schwarz preconditioner is a simple domain decomposition preconditioner that
 * generalizes the Block Jacobi preconditioner, incorporating options for
 * different local subdomain solvers and overlaps between the subdomains.
 *
 * A L1 smoother variant is also available, which updates the diagonal block
 * with the sums of the off-diagonal block row sums.
 *
 * See Iterative Methods for Sparse Linear Systems (Y. Saad) for a general
 * treatment and variations of the method.
 *
 * A Two-level variant is also available. To enable two-level preconditioning,
 * you need to specify a LinOpFactory that can generate a
 * multigrid::MultigridLevel and a solver for the coarse level solution.
 * Currently, only additive coarse correction is supported with an optional
 * weighting between the local and the coarse solutions, for cases when the
 * coarse solutions might tend to overcorrect.
 * - See Smith, Bjorstad, Gropp, Domain Decomposition, 1996, Cambridge
 * University Press.
 *
 * @note Currently overlap is not supported (TODO).
 *
 * @tparam ValueType  precision of matrix element
 * @tparam LocalIndexType  local integer type of the matrix
 * @tparam GlobalIndexType  global integer type of the matrix
 *
 * @ingroup schwarz
 * @ingroup precond
 * @ingroup LinOp
 */
template <typename ValueType = default_precision,
          typename LocalIndexType = int32, typename GlobalIndexType = int64>
class Schwarz
    : public EnableLinOp<Schwarz<ValueType, LocalIndexType, GlobalIndexType>> {
    friend class EnableLinOp<Schwarz>;
    friend class EnablePolymorphicObject<Schwarz, LinOp>;
    GKO_ASSERT_SUPPORTED_VALUE_AND_DIST_INDEX_TYPE;

public:
    using EnableLinOp<Schwarz>::convert_to;
    using EnableLinOp<Schwarz>::move_to;
    using value_type = ValueType;
    using index_type = GlobalIndexType;
    using local_index_type = LocalIndexType;
    using global_index_type = GlobalIndexType;

    Schwarz(const Schwarz& other) : EnableLinOp<Schwarz>(other)
    {
        *this = other;
    }
    Schwarz(Schwarz&& other) noexcept : EnableLinOp<Schwarz>(std::move(other))
    {
        *this = std::move(other);
    }
    Schwarz& operator=(const Schwarz& other)
    {
        if (this != &other) {
            EnableLinOp<Schwarz>::operator=(other);
            local_solver_ = other.local_solver_;
            system_matrix_ = other.system_matrix_;
            parameters_ = other.parameters_;
            coarse_level_ = other.coarse_level_;
            coarse_solver_ = other.coarse_solver_;
            coarse_weight_ = other.coarse_weight_;
            local_weight_ = other.local_weight_;
        }
        return *this;
    }
    Schwarz& operator=(Schwarz&& other) noexcept
    {
        if (this != &other) {
            EnableLinOp<Schwarz>::operator=(std::move(other));
            local_solver_ = std::move(other.local_solver_);
            system_matrix_ = std::move(other.system_matrix_);
            parameters_ = std::move(other.parameters_);
            coarse_level_ = std::move(other.coarse_level_);
            coarse_solver_ = std::move(other.coarse_solver_);
            coarse_weight_ = std::move(other.coarse_weight_);
            local_weight_ = std::move(other.local_weight_);
            owned_workspace_ = std::move(other.owned_workspace_);
            workspace_view_ = std::exchange(other.workspace_view_, nullptr);
        }
        return *this;
    }

    /**
     * Return whether the local solvers use the data in x as an initial guess.
     *
     * @return true when the local solvers use the data in x as an initial
     * guess. otherwise, false.
     *
     * @note TODO: after adding refining step, need to revisit this.
     */
    bool apply_uses_initial_guess() const override;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Local solver factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_DEFERRED_FACTORY_PARAMETER(
            local_solver);

        /**
         * Generated Inner solvers.
         */
        std::shared_ptr<const LinOp> GKO_FACTORY_PARAMETER_SCALAR(
            generated_local_solver, nullptr);

        /**
         * Enable l1 smoother.
         *
         * This creates a diagonal matrix from the row-wise absolute
         * sum of the off-diagonal matrix entries. The diagonal matrix
         * is then added to the system matrix when generating the
         * local solver.
         *
         * Note: The L1 smoother will not be used for the coarse level matrix
         * generation, and the original system matrix will still be used.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(l1_smoother, false);

        /**
         * Coarse weighting.
         *
         * By default the coarse and the local solutions are added together
         * (when the coarse weight is < 0 or > 1). A weighting can instead be
         * provided if the coarse solution tends to over-correct.
         */
        ValueType GKO_FACTORY_PARAMETER_SCALAR(coarse_weight, ValueType{-1.0});

        /**
         * Operator factory list to generate the triplet (prolong_op, coarse_op,
         * restrict_op), `A_c = R * A * P`
         *
         * Note: The linop factory must generate the triplet (R, A_c, P). For
         * example, any coarse level generator from multigrid::MultigridLevel
         * can be used.
         */
        std::shared_ptr<const LinOpFactory> GKO_DEFERRED_FACTORY_PARAMETER(
            coarse_level);

        /**
         * Coarse solver factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_DEFERRED_FACTORY_PARAMETER(
            coarse_solver);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Schwarz, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /**
     * Create the parameters from the property_tree.
     * Because this is directly tied to the specific type, the value/index type
     * settings within config are ignored and type_descriptor is only used
     * for children objects.
     *
     * @param config  the property tree for setting
     * @param context  the registry
     * @param td_for_child  the type descriptor for children objects. The
     *                      default uses the value/local/global index type of
     *                      this class.
     *
     * @return parameters
     */
    static parameters_type parse(
        const config::pnode& config, const config::registry& context,
        const config::type_descriptor& td_for_child =
            config::make_type_descriptor<ValueType, LocalIndexType,
                                         GlobalIndexType>());

protected:
    /**
     * Creates an empty Schwarz preconditioner.
     *
     * @param exec  the executor this object is assigned to
     */
    explicit Schwarz(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Schwarz>(std::move(exec))
    {}

    /**
     * Creates a Schwarz preconditioner from a matrix using a Schwarz::Factory.
     *
     * @param factory  the factory to use to create the preconditioner
     * @param system_matrix  the matrix this preconditioner should be created
     *                       from
     */
    explicit Schwarz(const Factory* factory, LinOpGenerateComponents components)
        : EnableLinOp<Schwarz>(
              factory->get_executor(),
              gko::transpose(components.system_matrix->get_size())),
          parameters_{factory->get_parameters()},
          system_matrix_{components.system_matrix}
    {
        if (components.has_owned_workspace()) {
            owned_workspace_ = components.take_owned_workspace();
            workspace_view_ = owned_workspace_.get();
        } else if (components.has_view_workspace()) {
            workspace_view_ = components.get_view_workspace();
        }
        this->generate(std::move(components.system_matrix));
    }

    /**
     * Generates the preconditioner.
     */
    void generate(std::shared_ptr<const LinOp> system_matrix);

    /**
     * Generate an inner LinOp, threading the workspace view through to the
     * factory when one is available. Otherwise falls back to the workspace-
     * less generate overload. `tag` names the child node created under the
     * Schwarz workspace.
     */
    std::unique_ptr<LinOp> generate_inner(const LinOpFactory* factory,
                                          std::shared_ptr<const LinOp> matrix,
                                          const std::string& tag);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    template <typename VectorType>
    void apply_dense_impl(const VectorType* b, VectorType* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    /**
     * Sets the solver operator used as the local solver.
     *
     * @param new_solver  the new local solver
     */
    void set_solver(std::shared_ptr<const LinOp> new_solver);

    std::shared_ptr<const LinOp> local_solver_;
    std::shared_ptr<const LinOp> system_matrix_;

    std::unique_ptr<solver::Workspace> owned_workspace_;
    solver::Workspace* workspace_view_ = nullptr;

    // Used for advanced apply
    detail::VectorCache<ValueType> cache_;
    // Used in apply for two-level method
    detail::VectorCache<ValueType> csol_cache_;
    detail::VectorCache<ValueType> crhs_cache_;

    std::shared_ptr<const LinOp> coarse_level_;
    std::shared_ptr<const LinOp> coarse_solver_;
    std::shared_ptr<const matrix::Dense<ValueType>> coarse_weight_;
    std::shared_ptr<const matrix::Dense<ValueType>> local_weight_;
};


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GINKGO_BUILD_MPI
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_SCHWARZ_HPP_
