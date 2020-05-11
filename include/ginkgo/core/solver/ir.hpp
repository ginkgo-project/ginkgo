/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_SOLVER_IR_HPP_
#define GKO_CORE_SOLVER_IR_HPP_


#include <vector>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace solver {


/**
 * Iterative refinement (IR) is an iterative method that uses another coarse
 * method to approximate the error of the current solution via the current
 * residual.
 *
 * For any approximation of the solution `solution` to the system `Ax = b`, the
 * residual is defined as: `residual = b - A solution`. The error in
 * `solution`,  `e = x - solution` (with `x` being the exact solution) can be
 * obtained as the solution to the residual equation `Ae = residual`, since `A e
 * = Ax - A solution = b - A solution = residual`. Then, the real solution is
 * computed as `x = solution + e`. Instead of accurately solving the residual
 * equation `Ae = residual`, the solution of the system `e` can be approximated
 * to obtain the approximation `error` using a coarse method `solver`, which is
 * used to update `solution`, and the entire process is repeated with the
 * updated `solution`.  This yields the iterative refinement method:
 *
 * ```
 * solution = initial_guess
 * while not converged:
 *     residual = b - A solution
 *     error = solver(A, residual)
 *     solution = solution + error
 * ```
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
 * its eigenvalues `lambda` have to satisfy the equation `|lambda - 1| < 1).
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Ir : public EnableLinOp<Ir<ValueType>> {
    friend class EnableLinOp<Ir>;
    friend class EnablePolymorphicObject<Ir, LinOp>;

public:
    using value_type = ValueType;

    /**
     * Returns the system operator (matrix) of the linear system.
     *
     * @return the system operator (matrix)
     */
    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    /**
     * Return true as iterative solvers use the data in x as an initial guess.
     */
    virtual bool apply_uses_initial_guess() const { return true; }

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
    void set_solver(std::shared_ptr<const LinOp> new_solver)
    {
        GKO_ASSERT_EQUAL_DIMENSIONS(new_solver, this);
        solver_ = new_solver;
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Criterion factories.
         */
        std::vector<std::shared_ptr<const stop::CriterionFactory>>
            GKO_FACTORY_PARAMETER(criteria, nullptr);

        /**
         * Inner solver factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER(solver,
                                                                  nullptr);

        /**
         * Already generated solver. If one is provided, the factory `solver`
         * will be ignored.
         */
        std::shared_ptr<const LinOp> GKO_FACTORY_PARAMETER(generated_solver,
                                                           nullptr);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Ir, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    explicit Ir(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Ir>(std::move(exec))
    {}

    explicit Ir(const Factory *factory,
                std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Ir>(factory->get_executor(),
                          transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()},
          system_matrix_{std::move(system_matrix)}
    {
        if (parameters_.generated_solver) {
            solver_ = parameters_.generated_solver;
            GKO_ASSERT_EQUAL_DIMENSIONS(solver_, this);
        } else if (parameters_.solver) {
            solver_ = parameters_.solver->generate(system_matrix_);
        } else {
            solver_ = matrix::Identity<ValueType>::create(this->get_executor(),
                                                          this->get_size()[0]);
        }
        stop_criterion_factory_ =
            stop::combine(std::move(parameters_.criteria));
    }

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    std::shared_ptr<const LinOp> solver_{};
    std::shared_ptr<const stop::CriterionFactory> stop_criterion_factory_{};
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_CORE_SOLVER_IR_HPP_
