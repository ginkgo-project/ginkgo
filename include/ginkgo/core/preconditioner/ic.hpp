// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_IC_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_IC_HPP_


#include <memory>
#include <type_traits>

#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/type_traits.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/factorization/par_ic.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/solver_traits.hpp>
#include <ginkgo/core/solver/triangular.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


namespace gko {
namespace preconditioner {


/**
 * The Incomplete Cholesky (IC) preconditioner solves the equation \f$LL^H*x = b\f$
 * for a given lower triangular matrix L and the right hand side b (can contain
 * multiple right hand sides).
 *
 * It allows setting the solver for L, defaulting to solver::LowerTrs, which is
 * a direct triangular solvers. The solver for L^H is the
 * conjugate-transposed solver for L, ensuring that the preconditioner is
 * symmetric and positive-definite. For this L solver, a factory can be provided
 * (using `with_l_solver`) to have more control over their behavior. In
 * particular, it is possible to use an iterative method for solving the
 * triangular systems.
 *
 * An object of this class can be created with a matrix or a gko::Composition
 * containing two matrices. If created with a matrix, it is factorized before
 * creating the solver. If a gko::Composition (containing two matrices) is
 * used, the first operand will be taken as the L matrix, the second will be
 * considered the L^H matrix, which helps to avoid the otherwise necessary
 * transposition of L inside the solver. ParIc can be directly used, since it
 * orders the factors in the correct way.
 *
 * @note When providing a gko::Composition, the first matrix must be the lower
 *       matrix (\f$L\f$), and the second matrix must be its conjugate-transpose
 * (\f$L^H\f$). If they are swapped, solving might crash or return the wrong result.
 *
 * @note Do not use symmetric solvers (like CG) for the L solver since both
 *       matrices (L and L^H) are, by design, not symmetric.
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
 * @tparam ValueType  the value type used for the L matrix.
 * @tparam IndexType  type of the indices when ParIc is used to generate
 *                    the L and L^H factors. Irrelevant otherwise.
 *
 * @ingroup precond
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Ic : public LinOp, public Transposable {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using transposed_type = Ic;

    class Factory;

    struct parameters_type
        : public enable_parameters_type<parameters_type, Factory> {
        /**
         * Factory for the L solver
         */
        std::shared_ptr<const LinOpFactory> l_solver_factory{};

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
            deferred_factory_parameter<const LinOpFactory> solver)
        {
            this->l_solver_generator = std::move(solver);
            this->deferred_factories["l_solver"] = [](const auto& exec,
                                                      auto& params) {
                if (!params.l_solver_generator.is_empty()) {
                    params.l_solver_factory =
                        params.l_solver_generator.on(exec);
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

        deferred_factory_parameter<const LinOpFactory> factorization_generator;
    };

    GKO_ENABLE_LIN_OP_FACTORY(Ic, parameters, Factory);
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
     *                      When l_solver_type uses LinOp not concrete type, it
     *                      will use the default_precision in ginkgo.
     *
     * @return parameters
     *
     * @note only support the following when using <ValueType, IndexType> not
     *       <LSolverType, IndexType> variants
     */
    static parameters_type parse(
        const config::pnode& config, const config::registry& context,
        const config::type_descriptor& td_for_child =
            config::make_type_descriptor<value_type, index_type>());

    /**
     * Returns the solver which is used for the provided L matrix.
     *
     * @returns  the solver which is used for the provided L matrix
     */
    std::shared_ptr<const LinOp> get_l_solver() const { return l_solver_; }

    /**
     * Returns the solver which is used for the L^H matrix.
     *
     * @returns  the solver which is used for the L^H matrix
     */
    std::shared_ptr<const LinOp> get_lh_solver() const { return lh_solver_; }

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Copy-assigns an IC preconditioner. Preserves the executor,
     * shallow-copies the solvers and parameters. Creates a clone of the solvers
     * if they are on the wrong executor.
     */
    Ic& operator=(const Ic& other);

    /**
     * Move-assigns an IC preconditioner. Preserves the executor,
     * moves the solvers and parameters. Creates a clone of the solvers
     * if they are on the wrong executor. The moved-from object is empty (0x0
     * with nullptr solvers and default parameters)
     */
    Ic& operator=(Ic&& other);

    /**
     * Copy-constructs an IC preconditioner. Inherits the executor,
     * shallow-copies the solvers and parameters.
     */
    Ic(const Ic& other);

    /**
     * Move-constructs an IC preconditioner. Inherits the executor,
     * moves the solvers and parameters. The moved-from object is empty (0x0
     * with nullptr solvers and default parameters)
     */
    Ic(Ic&& other);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    explicit Ic(std::shared_ptr<const Executor> exec);

    explicit Ic(const Factory* factory, std::shared_ptr<const LinOp> lin_op);

    /**
     * Prepares the intermediate vector for the solve by creating it and
     * by copying the values from `b`, so `b` acts as the initial guess.
     *
     * @param b  Right hand side of the first solve. Also acts as the
     * initial guess, meaning the intermediate value will be a copy of b
     */
    void set_cache_to(const LinOp* b) const;

private:
    std::shared_ptr<const LinOp> l_solver_{};
    std::shared_ptr<const LinOp> lh_solver_{};
    /**
     * Manages a vector as a cache, so there is no need to allocate one
     * every time an intermediate vector is required. Copying an instance
     * will only yield an empty object since copying the cached vector would
     * not make sense.
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
        std::unique_ptr<LinOp> intermediate{};
    } cache_;
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_IC_HPP_
