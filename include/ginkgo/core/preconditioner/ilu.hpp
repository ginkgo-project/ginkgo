// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_ILU_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_ILU_HPP_


#include <memory>
#include <type_traits>

#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/type_traits.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/factorization/par_ilu.hpp>
#include <ginkgo/core/solver/solver_traits.hpp>
#include <ginkgo/core/solver/triangular.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


namespace gko {
namespace preconditioner {


/**
 * The Incomplete LU (ILU) preconditioner solves the equation \f$LUx = b\f$ for
 * a given lower triangular matrix \f$L\f$, an upper triangular matrix
 * \f$U\f$ and the right hand side \f$b\f$ (can contain multiple right hand
 * sides).
 *
 * It allows to set both the solver for \f$L\f$ and the solver for \f$U\f$
 * independently, while providing the defaults solver::LowerTrs and
 * solver::UpperTrs, which are direct triangular solvers.
 * For these solvers, a factory can be provided (with `with_l_solver` and
 * `with_u_solver`) to have more control over their behavior. In particular, it
 * is possible to use an iterative method for solving the triangular systems.
 *
 * An object of this class can be created with a matrix or a gko::Composition
 * containing two matrices. If created with a matrix, it is factorized before
 * creating the solver. If a gko::Composition (containing two matrices) is
 * used, the first operand will be taken as the \f$L\f$ matrix, the second
 * will be considered the \f$U\f$ matrix. ParIlu can be directly used, since
 * it orders the factors in the correct way.
 *
 * @note When providing a gko::Composition, the first matrix must be the lower
 *       matrix (\f$L\f$), and the second matrix must be the upper matrix
 *       (\f$U\f$). If they are swapped, solving might crash or return the
 *       wrong result.
 *
 * @note Do not use symmetric solvers (like CG) for \f$L\f$ or \f$U\f$
 *       solvers since both matrices (\f$L\f$ and \f$U\f$) are, by design,
 *       not symmetric.
 *
 * @note This class is not thread safe (even a const object is not) because it
 *       uses an internal cache to accelerate multiple (sequential) applies.
 *       Using it in parallel can lead to segmentation faults, wrong results
 *       and other unwanted behavior.
 *
 * @note The default template during parse is <ValueType, IndexType> not
 *       <LowerTrs, IndexType>. Only the variants with ValueType are supported
 *       in parse.
 *
 * @tparam ValueType  the value type used for the \f$L\f$ and \f$U\f$
 *                    matrices.
 * @tparam ReverseApply  default behavior (ReverseApply = false) is first to
 *                       solve with \f$L\f$ (\f$Ly = b\f$) and then with
 *                       \f$U\f$ (\f$Ux = y\f$). When set to true, it will
 *                       solve first with \f$U\f$, and then with \f$L\f$.
 * @tparam IndexType  Type of the indices when ParIlu is used to generate
 *                    both \f$L\f$ and \f$U\f$ factors. Irrelevant
 *                    otherwise.
 *
 * @ingroup precond
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, bool ReverseApply = false,
          typename IndexType = int32>
class Ilu : public LinOp, public Transposable {
public:
    using value_type = ValueType;
    static constexpr bool performs_reverse_apply = ReverseApply;
    using index_type = IndexType;
    using transposed_type = Ilu;

    class Factory;

    struct parameters_type
        : public enable_parameters_type<parameters_type, Factory> {
        /**
         * Factory for the \f$L\f$ solver
         */
        std::shared_ptr<const LinOpFactory> l_solver_factory{};

        /**
         * Factory for the \f$U\f$ solver
         */
        std::shared_ptr<const LinOpFactory> u_solver_factory{};

        /**
         * Factory for the factorization
         */
        std::shared_ptr<const LinOpFactory> factorization_factory{};

        /**
         * When LSolverTypeOrValueType is a concrete solver type, this only
         * accepts the factory from the same concrete solver type. When
         * LSolverTypeOrValueType is a value type, it accepts any LinOpFactory.
         */
        parameters_type& with_l_solver(
            deferred_factory_parameter<const LinOpFactory> solver);

        /**
         * When USolverTypeOrValueType is a concrete solver type, this only
         * accepts the factory from the same concrete solver type. When
         * USolverTypeOrValueType is a value type, it accepts any LinOpFactory.
         */
        parameters_type& with_u_solver(
            deferred_factory_parameter<const LinOpFactory> solver)
        {
            this->u_solver_generator = std::move(solver);
            this->deferred_factories["u_solver"] = [](const auto& exec,
                                                      auto& params) {
                if (!params.u_solver_generator.is_empty()) {
                    params.u_solver_factory =
                        params.u_solver_generator.on(exec);
                }
            };
            return *this;
        }

        parameters_type& with_factorization(
            deferred_factory_parameter<const LinOpFactory> factorization)
        {
            this->factorization_generator = std::move(factorization);
            this->deferred_factories["factorization"] = [](const auto& exec,
                                                           auto& params) {
                if (!params.factorization_generator.is_empty()) {
                    params.factorization_factory =
                        params.factorization_generator.on(exec);
                }
            };
            return *this;
        }

    private:
        deferred_factory_parameter<const LinOpFactory> l_solver_generator;
        deferred_factory_parameter<const LinOpFactory> u_solver_generator;
        deferred_factory_parameter<const LinOpFactory> factorization_generator;
    };

    GKO_ENABLE_LIN_OP_FACTORY(Ilu, parameters, Factory);
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
     *                      default uses the value/index type of this class.
     *
     * @return parameters
     *
     * @note only support the following when using <ValueType, ValueType,
     *       ReverseApply, IndexType> not <LSolverType, USolverType,
     *       ReverseApply, IndexType> variants
     */
    static parameters_type parse(
        const config::pnode& config, const config::registry& context,
        const config::type_descriptor& td_for_child =
            config::make_type_descriptor<value_type, index_type>());

    /**
     * Returns the solver which is used for the provided \f$L\f$ matrix.
     *
     * @returns  the solver which is used for the provided \f$L\f$ matrix
     */
    std::shared_ptr<const LinOp> get_l_solver() const { return l_solver_; }

    /**
     * Returns the solver which is used for the provided \f$U\f$ matrix.
     *
     * @returns  the solver which is used for the provided \f$U\f$ matrix
     */
    std::shared_ptr<const LinOp> get_u_solver() const { return u_solver_; }

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Copy-assigns an ILU preconditioner. Preserves the executor,
     * shallow-copies the solvers and parameters. Creates a clone of the solvers
     * if they are on the wrong executor.
     */
    Ilu& operator=(const Ilu& other);

    /**
     * Move-assigns an ILU preconditioner. Preserves the executor,
     * moves the solvers and parameters. Creates a clone of the solvers
     * if they are on the wrong executor. The moved-from object is empty (0x0
     * with nullptr solvers and default parameters)
     */
    Ilu& operator=(Ilu&& other);

    /**
     * Copy-constructs an ILU preconditioner. Inherits the executor,
     * shallow-copies the solvers and parameters.
     */
    Ilu(const Ilu& other);

    /**
     * Move-constructs an ILU preconditioner. Inherits the executor,
     * moves the solvers and parameters. The moved-from object is empty (0x0
     * with nullptr solvers and default parameters)
     */
    Ilu(Ilu&& other);

protected:
    void apply_impl(const AbstractMultiVector* b,
                    AbstractMultiVector* x) const override;

    void apply_impl(const AbstractMultiVector* alpha,
                    const AbstractMultiVector* b,
                    const AbstractMultiVector* beta,
                    AbstractMultiVector* x) const override;

    explicit Ilu(std::shared_ptr<const Executor> exec);

    explicit Ilu(const Factory* factory, std::shared_ptr<const LinOp> lin_op);

    /**
     * Prepares the intermediate vector for the solve by creating it and
     * by copying the values from `b`, so `b` acts as the initial guess.
     *
     * @param b  Right hand side of the first solve. Also acts as the initial
     *           guess, meaning the intermediate value will be a copy of b
     */
    void set_cache_to(const AbstractMultiVector* b) const;

private:
    std::shared_ptr<const LinOp> l_solver_{};
    std::shared_ptr<const LinOp> u_solver_{};
    /**
     * Manages a vector as a cache, so there is no need to allocate one every
     * time an intermediate vector is required.
     * Copying an instance will only yield an empty object since copying the
     * cached vector would not make sense.
     *
     * @internal  The struct is present so the whole class can be copyable
     *            (could also be done with writing `operator=` and copy
     *            constructor of the enclosing class by hand)
     */
    mutable struct cache_struct {
        cache_struct() = default;
        ~cache_struct() = default;
        cache_struct(const cache_struct&) {}
        cache_struct(cache_struct&&) {}
        cache_struct& operator=(const cache_struct&) { return *this; }
        cache_struct& operator=(cache_struct&&) { return *this; }
        std::unique_ptr<matrix::MultiVector<ValueType>> intermediate{};
    } cache_;
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_ILU_HPP_
