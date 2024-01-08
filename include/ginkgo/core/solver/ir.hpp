// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_IR_HPP_
#define GKO_PUBLIC_CORE_SOLVER_IR_HPP_


#include <vector>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
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
 * For any approximation of the solution `solution` to the system `Ax = b`, the
 * residual is defined as: `residual = b - A solution`. The error in
 * `solution`,  `e = x - solution` (with `x` being the exact solution) can be
 * obtained as the solution to the residual equation `Ae = residual`, since `A e
 * = Ax - A solution = b - A solution = residual`. Then, the real solution is
 * computed as `x = relaxation_factor * solution + e`. Instead of accurately
 * solving the residual equation `Ae = residual`, the solution of the system `e`
 * can be approximated to obtain the approximation `error` using a coarse method
 * `solver`, which is used to update `solution`, and the entire process is
 * repeated with the updated `solution`.  This yields the iterative refinement
 * method:
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
 * Assuming that `solver` has accuracy `c`, i.e., `| e - error | <= c | e |`,
 * iterative refinement will converge with a convergence rate of `c`. Indeed,
 * from `e - error = x - solution - error = x - solution*` (where `solution*`
 * denotes the value stored in `solution` after the update) and `e = inv(A)
 * residual = inv(A)b - inv(A) A solution = x - solution` it follows that | x -
 * solution* | <= c | x - solution |.
 *
 * Unless otherwise specified via the `solver` factory parameter, this
 * implementation uses the identity operator (i.e. the solver that approximates
 * the solution of a system Ax = b by setting x := b) as the default inner
 * solver. Such a setting results in a relaxation method known as the Richardson
 * iteration with parameter 1, which is guaranteed to converge for matrices
 * whose spectrum is strictly contained within the unit disc around 1 (i.e., all
 * its eigenvalues `lambda` have to satisfy the equation `|relaxation_factor *
 * lambda - 1| < 1).
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Ir : public EnableLinOp<Ir<ValueType>>,
           public EnableSolverBase<Ir<ValueType>>,
           public EnableIterativeBase<Ir<ValueType>>,
           public EnableApplyWithInitialGuess<Ir<ValueType>>,
           public Transposable {
    friend class EnableLinOp<Ir>;
    friend class EnablePolymorphicObject<Ir, LinOp>;
    friend class EnableApplyWithInitialGuess<Ir>;

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

    explicit Ir(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Ir>(std::move(exec))
    {}

    explicit Ir(const Factory* factory,
                std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Ir>(factory->get_executor(),
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
                this->get_executor(), this->get_size()));
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
