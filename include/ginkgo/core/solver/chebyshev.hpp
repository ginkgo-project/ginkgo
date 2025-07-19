// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_CHEBYSHEV_HPP_
#define GKO_PUBLIC_CORE_SOLVER_CHEBYSHEV_HPP_


#include <vector>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/iteration.hpp>


namespace gko {
namespace solver {
namespace detail {


template <typename T>
using coeff_type =
    std::conditional_t<is_complex<T>(), std::complex<double>, double>;


}

/**
 * Chebyshev iteration is an iterative method for solving nonsymmetric problems
 * based on some knowledge of the spectrum of the (preconditioned) system
 * matrix. It avoids the computation of inner products, which may be a
 * performance bottleneck for distributed system. Chebyshev iteration is
 * developed based on Chebyshev polynomials of the first kind.
 * This implementation follows the algorithm in "Templates for the
 * Solution of Linear Systems: Building Blocks for Iterative Methods, 2nd
 * Edition".
 *
 * ```
 * solution = initial_guess
 * while not converged:
 *     residual = b - A solution
 *     error = preconditioner(A) * residual
 *     solution = solution + alpha_i * error + beta_i * (solution_i -
 * solution_{i-1})
 * ```
 *
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Chebyshev final
    : public EnableLinOp<Chebyshev<ValueType>>,
      public EnablePreconditionedIterativeSolver<ValueType,
                                                 Chebyshev<ValueType>>,
      public EnableApplyWithInitialGuess<Chebyshev<ValueType>>,
      public Transposable {
    friend class EnableLinOp<Chebyshev>;
    friend class EnablePolymorphicObject<Chebyshev, LinOp>;
    friend class EnableApplyWithInitialGuess<Chebyshev>;

public:
    using value_type = ValueType;
    using transposed_type = Chebyshev<ValueType>;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Return true as iterative solvers use the data in x as an initial guess.
     *
     * @return true as iterative solvers use the data in x as an initial guess.
     */
    bool apply_uses_initial_guess() const override
    {
        return this->get_default_initial_guess() ==
               initial_guess_mode::provided;
    }

    /**
     * Copy-assigns a Chebyshev solver. Preserves the executor, shallow-copies
     * inner solver, stopping criterion and system matrix. If the executors
     * mismatch, clones inner solver, stopping criterion and system matrix onto
     * this executor.
     */
    Chebyshev& operator=(const Chebyshev&);

    /**
     * Move-assigns a Chebyshev solver. Preserves the executor, moves inner
     * solver, stopping criterion and system matrix. If the executors mismatch,
     * clones inner solver, stopping criterion and system matrix onto this
     * executor. The moved-from object is empty (0x0 and nullptr inner solver,
     * stopping criterion and system matrix)
     */
    Chebyshev& operator=(Chebyshev&&);

    /**
     * Copy-constructs an Chebyshev solver. Inherits the executor,
     * shallow-copies inner solver, stopping criterion and system matrix.
     */
    Chebyshev(const Chebyshev&);

    /**
     * Move-constructs an Chebyshev solver. Preserves the executor, moves inner
     * solver, stopping criterion and system matrix. The moved-from object is
     * empty (0x0 and nullptr inner solver, stopping criterion and system
     * matrix)
     */
    Chebyshev(Chebyshev&&);

    class Factory;

    struct parameters_type
        : enable_preconditioned_iterative_solver_factory_parameters<
              parameters_type, Factory> {
        /**
         * The pair of foci of ellipse, which covers the eigenvalues of
         * preconditioned system. It is usually a pair {lower bound of eigval,
         * upper bound of eigval} of the preconditioned system if the
         * preconditioned system only contains non-complex eigenvalues. The foci
         * value must satisfy real(foci(1)) >= real(foci(0)).
         */
        std::pair<detail::coeff_type<value_type>,
                  detail::coeff_type<value_type>>
            GKO_FACTORY_PARAMETER_VECTOR(foci,
                                         detail::coeff_type<value_type>{0},
                                         detail::coeff_type<value_type>{1});

        /**
         * Default initial guess mode. The available options are under
         * initial_guess_mode.
         */
        initial_guess_mode GKO_FACTORY_PARAMETER_SCALAR(
            default_initial_guess, initial_guess_mode::provided);
    };

    GKO_ENABLE_LIN_OP_FACTORY(Chebyshev, parameters, Factory);
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

    template <typename VectorType>
    void apply_dense_impl(const VectorType* b, VectorType* x,
                          initial_guess_mode guess) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    void apply_with_initial_guess_impl(const LinOp* b, LinOp* x,
                                       initial_guess_mode guess) const override;

    void apply_with_initial_guess_impl(const LinOp* alpha, const LinOp* b,
                                       const LinOp* beta, LinOp* x,
                                       initial_guess_mode guess) const override;

    explicit Chebyshev(std::shared_ptr<const Executor> exec);

    explicit Chebyshev(const Factory* factory,
                       std::shared_ptr<const LinOp> system_matrix);

private:
    std::shared_ptr<const LinOp> solver_{};
    detail::coeff_type<value_type> center_;
    detail::coeff_type<value_type> foci_direction_;
};


template <typename ValueType>
struct workspace_traits<Chebyshev<ValueType>> {
    using Solver = Chebyshev<ValueType>;
    // number of vectors used by this workspace
    static int num_vectors(const Solver&);
    // number of arrays used by this workspace
    static int num_arrays(const Solver&);
    // array containing the num_vectors names for the workspace vectors
    static std::vector<std::string> op_names(const Solver&);
    // array containing the num_arrays names for the workspace vectors
    static std::vector<std::string> array_names(const Solver&);
    // array containing all varying scalar vectors (independent of problem size)
    static std::vector<int> scalars(const Solver&);
    // array containing all varying vectors (dependent on problem size)
    static std::vector<int> vectors(const Solver&);

    // residual vector
    constexpr static int residual = 0;
    // inner solution vector
    constexpr static int inner_solution = 1;
    // update solution
    constexpr static int update_solution = 2;
    // constant 1.0 scalar
    constexpr static int one = 3;
    // constant -1.0 scalar
    constexpr static int minus_one = 4;

    // stopping status array
    constexpr static int stop = 0;
    // stopping indicator array
    constexpr static int indicators = 1;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_CHEBYSHEV_HPP_
