// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_ASYNC_JACOBI_HPP_
#define GKO_PUBLIC_CORE_SOLVER_ASYNC_JACOBI_HPP_


#include <string>
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
 * Iterative refinement (ASYNC_JACOBI) is an iterative method that uses
 * another coarse method to approximate the error of the current solution via
 * the current residual. Moreover, it can be also considered as preconditioned
 * Jacobi iteration with relaxation factor = 1.
 *
 * For any approximation of the solution `solution` to the system `Ax = b`, the
 * residual is defined as: `residual = b - A solution`. The error in
 * `solution`,  `e = x - solution` (with `x` being the exact solution) can be
 * obtained as the solution to the residual equation `Ae = residual`, since `A e
 * = Ax - A solution = b - A solution = residual`. Then, the real solution is
 * computed as `x = relaxation_factor * solution + e`. Instead of accurately
 * solving the residual equation `Ae = residual`, the solution of the system `e`
 * can be approximated to obtain the approximation `error` using a coarse method
 * `solver`, which is used to update `solution`, and the entasync_jacobie
 * process is repeated with the updated `solution`.  This yields the iterative
 * refinement method:
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
 * solver is a Jacobi iteration, with possibility for additional
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
 * solver. Such a setting results in a relaxation method known as the Jacobi
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
template <typename ValueType = default_precision,
          typename IndexType = gko::int32>
class AsyncJacobi
    : public EnableLinOp<AsyncJacobi<ValueType, IndexType>>,
      public EnableSolverBase<AsyncJacobi<ValueType, IndexType>>,
      public EnableIterativeBase<AsyncJacobi<ValueType, IndexType>>,
      public Transposable {
    friend class EnableLinOp<AsyncJacobi>;
    friend class EnablePolymorphicObject<AsyncJacobi, LinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using transposed_type = AsyncJacobi<ValueType, IndexType>;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Return true as iterative solvers use the data in x as an initial guess.
     *
     * @return true as iterative solvers use the data in x as an initial guess.
     */
    bool apply_uses_initial_guess() const override { return true; }

    /**
     * Copy-assigns an ASYNC_JACOBI solver. Preserves the executor,
     * shallow-copies inner solver, stopping criterion and system matrix. If the
     * executors mismatch, clones inner solver, stopping criterion and system
     * matrix onto this executor.
     */
    AsyncJacobi& operator=(const AsyncJacobi&);

    /**
     * Move-assigns an ASYNC_JACOBI solver. Preserves the executor, moves
     * inner solver, stopping criterion and system matrix. If the executors
     * mismatch, clones inner solver, stopping criterion and system matrix onto
     * this executor. The moved-from object is empty (0x0 and nullptr inner
     * solver, stopping criterion and system matrix)
     */
    AsyncJacobi& operator=(AsyncJacobi&&);

    /**
     * Copy-constructs an ASYNC_JACOBI solver. Inherits the executor,
     * shallow-copies inner solver, stopping criterion and system matrix.
     */
    AsyncJacobi(const AsyncJacobi&);

    /**
     * Move-constructs an ASYNC_JACOBI solver. Preserves the executor, moves
     * inner solver, stopping criterion and system matrix. The moved-from object
     * is empty (0x0 and nullptr inner solver, stopping criterion and system
     * matrix)
     */
    AsyncJacobi(AsyncJacobi&&);

    class Factory;

    struct parameters_type
        : enable_iterative_solver_factory_parameters<parameters_type, Factory> {
        /**
         * Relaxation factor for Jacobi iteration
         */
        ValueType GKO_FACTORY_PARAMETER_SCALAR(relaxation_factor,
                                               value_type{1});

        gko::size_type GKO_FACTORY_PARAMETER_SCALAR(max_iters, 100);

        /**
         * Second factor for Jacobi iteration
         */
        ValueType GKO_FACTORY_PARAMETER_SCALAR(second_factor, value_type{0});

        /**
         * check is to get the async iteration information (it is only designed
         * for 5-pt example with specific settings)
         * - "flow": get the source.
         * - "halfflow": get the source but only record the information until
         * half of max_iters.
         * - "time": record the start and end time information
         * - others: normal operation
         */
        std::string GKO_FACTORY_PARAMETER_SCALAR(check, std::string("normal"));
    };
    GKO_ENABLE_LIN_OP_FACTORY(AsyncJacobi, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_dense_impl(const matrix::Dense<ValueType>* b,
                          matrix::Dense<ValueType>* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    void set_relaxation_factor(
        std::shared_ptr<const matrix::Dense<ValueType>> new_factor);

    explicit AsyncJacobi(std::shared_ptr<const Executor> exec)
        : EnableLinOp<AsyncJacobi>(std::move(exec))
    {}

    explicit AsyncJacobi(const Factory* factory,
                         std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<AsyncJacobi>(factory->get_executor(),
                                   gko::transpose(system_matrix->get_size())),
          EnableSolverBase<AsyncJacobi>{std::move(system_matrix)},
          EnableIterativeBase<AsyncJacobi>{
              stop::combine(factory->get_parameters().criteria)},
          parameters_{factory->get_parameters()}
    {
        relaxation_factor_ = gko::initialize<matrix::Dense<ValueType>>(
            {parameters_.relaxation_factor}, this->get_executor());
        second_factor_ = gko::initialize<matrix::Dense<ValueType>>(
            {parameters_.second_factor}, this->get_executor());
    }

private:
    std::shared_ptr<const LinOp> solver_{};
    std::shared_ptr<const matrix::Dense<ValueType>> relaxation_factor_{};
    std::shared_ptr<const matrix::Dense<ValueType>> second_factor_{};
};


template <typename ValueType, typename IndexType>
struct workspace_traits<AsyncJacobi<ValueType, IndexType>> {
    using Solver = AsyncJacobi<ValueType, IndexType>;
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


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_ASYNC_JACOBI_HPP_
