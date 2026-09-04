// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_IR_HPP_
#define GKO_PUBLIC_CORE_SOLVER_IR_HPP_


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


/**
 * Iterative refinement (IR) is an iterative method that uses another coarse
 * method to approximate the error of the current solution via the current
 * residual. Moreover, it can be also considered as preconditioned Richardson
 * iteration with relaxation factor = 1.
 *
 * Let \f$ x_k \f$ be the approximation of the solution of \f$ A x = b \f$
 * after \f$ k \f$ iterations and let \f$ x \f$ denote the exact solution.
 * The residual and the error of \f$ x_k \f$ are
 * \f[
 *   r_k = b - A x_k, \qquad e_k = x - x_k,
 * \f]
 * and they are linked by the residual equation
 * \f$ A e_k = A x - A x_k = b - A x_k = r_k \f$. Knowing \f$ e_k \f$
 * exactly would give the exact solution in a single update
 * \f$ x = x_k + e_k \f$. Instead of solving \f$ A e_k = r_k \f$ exactly,
 * IR approximates \f$ e_k \f$ by \f$ \tilde e_k \f$ using a cheap inner
 * `solver`, applies that correction with a relaxation factor
 * \f$ \alpha \f$,
 * \f[
 *   x_{k+1} = x_k + \alpha \tilde e_k,
 * \f]
 * and repeats the process with the updated iterate. Written with the names
 * used below, \f$ x_k \f$ is `solution`, \f$ r_k \f$ is `residual` and
 * \f$ \tilde e_k \f$ is `error`:
 *
 * ```
 * solution = initial_guess
 * while not converged:
 *     residual = b - A solution
 *     error = solver(A, residual)
 *     solution = solution + relaxation_factor * error
 * ```
 *
 * With `relaxation_factor` equal to 1 (default), the solver is Iterative
 * Refinement, with `relaxation_factor` equal to a value other than `1`, the
 * solver is a Richardson iteration, with possibility for additional
 * preconditioning.
 *
 * Assume \f$ \alpha = 1 \f$ and that `solver` has accuracy \f$ c \f$,
 * i.e. \f$ \| e_k - \tilde e_k \| \le c \| e_k \| \f$. Then iterative
 * refinement converges with a convergence rate of \f$ c \f$: from
 * \f$ e_k - \tilde e_k = (x - x_k) - \tilde e_k = x - x_{k+1} \f$ it
 * follows that \f$ \| x - x_{k+1} \| \le c \| x - x_k \| \f$.
 *
 * Unless otherwise specified via the `solver` factory parameter, this
 * implementation uses the identity operator (i.e. the solver that approximates
 * the solution of a system \f$ A x = b \f$ by setting \f$ x := b \f$) as the
 * default inner solver. It leaves the residual unchanged,
 * \f$ \tilde e_k = r_k \f$, so the iteration reduces to the Richardson
 * iteration \f$ x_{k+1} = x_k + \alpha r_k \f$, which converges for every
 * initial guess if and only if \f$ | 1 - \alpha \lambda | < 1 \f$ holds for
 * every eigenvalue \f$ \lambda \f$ of \f$ A \f$.
 *
 * @par References
 * - Saad, Y. *Iterative Methods for Sparse Linear Systems.* 2nd ed.
 *   SIAM, 2003. <https://doi.org/10.1137/1.9780898718003>
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Ir : public LinOp,
           public EnableSolverBase<Ir<ValueType>>,
           public EnableIterativeBase<Ir<ValueType>>,
           public EnableApplyWithInitialGuess<Ir<ValueType>>,
           public Transposable {
    friend class EnableApplyWithInitialGuess<Ir>;
    GKO_ASSERT_SUPPORTED_VALUE_TYPE;

public:
    using value_type = ValueType;
    using transposed_type = Ir<ValueType>;

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
     * Returns the solver operator used as the inner solver.
     *
     * @return the solver operator used as the inner solver
     */
    std::shared_ptr<const LinOp> get_solver() const { return solver_; }

    /**
     * Sets the solver operator used as the inner solver.
     *
     * @param new_solver  the new inner solver
     */
    void set_solver(std::shared_ptr<const LinOp> new_solver);

    /**
     * Copy-assigns an IR solver. Preserves the executor, shallow-copies inner
     * solver, stopping criterion and system matrix. If the executors mismatch,
     * clones inner solver, stopping criterion and system matrix onto this
     * executor.
     */
    Ir& operator=(const Ir&);

    /**
     * Move-assigns an IR solver. Preserves the executor, moves inner solver,
     * stopping criterion and system matrix. If the executors mismatch, clones
     * inner solver, stopping criterion and system matrix onto this executor.
     * The moved-from object is empty (0x0 and nullptr inner solver, stopping
     * criterion and system matrix)
     */
    Ir& operator=(Ir&&);

    /**
     * Copy-constructs an IR solver. Inherits the executor, shallow-copies inner
     * solver, stopping criterion and system matrix.
     */
    Ir(const Ir&);

    /**
     * Move-constructs an IR solver. Preserves the executor, moves inner solver,
     * stopping criterion and system matrix. The moved-from object is empty (0x0
     * and nullptr inner solver, stopping criterion and system matrix)
     */
    Ir(Ir&&);

    class Factory;

    struct parameters_type
        : enable_iterative_solver_factory_parameters<parameters_type, Factory> {
        /**
         * Inner solver factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_DEFERRED_FACTORY_PARAMETER(
            solver);

        /**
         * Already generated solver. If one is provided, the factory `solver`
         * will be ignored.
         */
        std::shared_ptr<const LinOp> GKO_FACTORY_PARAMETER_SCALAR(
            generated_solver, nullptr);

        /**
         * Relaxation factor for Richardson iteration
         */
        ValueType GKO_FACTORY_PARAMETER_SCALAR(relaxation_factor,
                                               value_type{1});

        /**
         * Default initial guess mode. The available options are under
         * initial_guess_mode.
         */
        initial_guess_mode GKO_FACTORY_PARAMETER_SCALAR(
            default_initial_guess, initial_guess_mode::provided);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Ir, parameters, Factory);
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

    void validate_data() const override;

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

    void set_relaxation_factor(
        std::shared_ptr<const matrix::Dense<ValueType>> new_factor);

    explicit Ir(std::shared_ptr<const Executor> exec) : LinOp(std::move(exec))
    {}

    explicit Ir(const Factory* factory,
                std::shared_ptr<const LinOp> system_matrix)
        : LinOp(factory->get_executor(),
                gko::transpose(system_matrix->get_size())),
          EnableSolverBase<Ir>{std::move(system_matrix)},
          EnableIterativeBase<Ir>{
              stop::combine(factory->get_parameters().criteria)},
          parameters_{factory->get_parameters()}
    {
        if (parameters_.generated_solver) {
            this->set_solver(parameters_.generated_solver);
        } else if (parameters_.solver) {
            this->set_solver(
                parameters_.solver->generate(this->get_system_matrix()));
        } else {
            this->set_solver(matrix::Identity<ValueType>::create(
                this->get_executor(), this->get_size()[0]));
        }
        this->set_default_initial_guess(parameters_.default_initial_guess);
        relaxation_factor_ = gko::initialize<matrix::Dense<ValueType>>(
            {parameters_.relaxation_factor}, this->get_executor());
    }

private:
    std::shared_ptr<const LinOp> solver_{};
    std::shared_ptr<const matrix::Dense<ValueType>> relaxation_factor_{};
};


template <typename ValueType = default_precision>
using Richardson = Ir<ValueType>;


template <typename ValueType>
struct workspace_traits<Ir<ValueType>> {
    using Solver = Ir<ValueType>;
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
    // constant 1.0 scalar
    constexpr static int one = 2;
    // constant -1.0 scalar
    constexpr static int minus_one = 3;

    // stopping status array
    constexpr static int stop = 0;
};


/**
 * build_smoother gives a shortcut to build a smoother by IR(Richardson) with
 * limited stop criterion(iterations and relacation_factor).
 *
 * @param factory  the shared pointer of factory
 * @param iteration  the maximum number of iteration, which default is 1
 * @param relaxation_factor  the relaxation factor for Richardson
 *
 * @return the pointer of Ir(Richardson)
 */
template <typename ValueType>
auto build_smoother(std::shared_ptr<const LinOpFactory> factory,
                    size_type iteration = 1, ValueType relaxation_factor = 0.9)
{
    auto exec = factory->get_executor();
    return Ir<ValueType>::build()
        .with_solver(factory)
        .with_relaxation_factor(relaxation_factor)
        .with_criteria(gko::stop::Iteration::build().with_max_iters(iteration))
        .on(exec);
}

/**
 * build_smoother gives a shortcut to build a smoother by IR(Richardson) with
 * limited stop criterion(iterations and relacation_factor).
 *
 * @param solver  the shared pointer of solver
 * @param iteration  the maximum number of iteration, which default is 1
 * @param relaxation_factor  the relaxation factor for Richardson
 *
 * @return the pointer of Ir(Richardson)
 *
 * @note this is the overload function for LinOp.
 */
template <typename ValueType>
auto build_smoother(std::shared_ptr<const LinOp> solver,
                    size_type iteration = 1, ValueType relaxation_factor = 0.9)
{
    auto exec = solver->get_executor();
    return Ir<ValueType>::build()
        .with_generated_solver(solver)
        .with_relaxation_factor(relaxation_factor)
        .with_criteria(gko::stop::Iteration::build().with_max_iters(iteration))
        .on(exec);
}


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_IR_HPP_
