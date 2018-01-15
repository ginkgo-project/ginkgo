/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_SOLVER_CG_HPP_
#define GKO_CORE_SOLVER_CG_HPP_


#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/lin_op.hpp"
#include "core/base/types.hpp"


namespace gko {
namespace solver {


template <typename>
class CgFactory;

/**
 * Cg or the conjugate gradient method is an iterative type krylov subspace
 * method which is suitable for symmetric positive definite methods.
 *
 * Though this method performs very well for symmetric positive definite
 * matrices, it is in general not suitable for general matrices.
 *
 * The implementation in Ginkgo makes use of the merged kernel to make the best
 * use of data locality. The inner operations in one iteration of CG are merged
 * into 2 separate steps.
 *
 * @tparam Valuetype precision of matrix elements
 */


template <typename ValueType = default_precision>
class Cg : public LinOp {
    friend class CgFactory<ValueType>;

public:
    using value_type = ValueType;

    /**
     * Creates a copy of the CG solver from another cg solver.
     *
     * @param other  the Cg solver instance to copy
     */
    void copy_from(const LinOp *other) override;

    /**
     * Moves the Cg solver from another cg solver.
     *
     * @param other  the Cg solver instance from which it will be moved.
     */
    void copy_from(std::unique_ptr<LinOp> other) override;

    /**
     * Applies the Cg solver to a vector.
     *
     * @param b  The right hand side of the linear system.
     *
     * @param x  The solution vector of the linear system.
     */
    void apply(const LinOp *b, LinOp *x) const override;

    /**
     * Applies the Cg solver to a vector and performs a scaled addtion.
     *
     * Performs the operation x = alpha * cg(b,x) + beta * x
     *
     * @param alpha Scaling of the result of cg(b,x).
     * @param b  The right hand side of the linear system.
     * @param beta  Scaling of the input x.
     * @param x  The solution vector of the linear system.
     */
    void apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
               LinOp *x) const override;

    /**
     * Creates a clone of the Cg solver
     *
     * @return  A clone of the Cg solver.
     */
    std::unique_ptr<LinOp> clone_type() const override;

    /**
     * Clears the Cg solver.
     */
    void clear() override;

    /**
     * Gets the system matrix of the linear system.
     *
     * @return  The system matrix.
     */
    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    /**
     * Gets the maximum number of iterations of the Cg solver.
     *
     * @return  The maximum number of iterations.
     */
    int get_max_iters() const { return max_iters_; }


    /**
     * Gets the relative residual goal of the solver.
     *
     * @return  The relative residual goal.
     */
    remove_complex<value_type> get_rel_residual_goal() const
    {
        return rel_residual_goal_;
    }


private:
    Cg(std::shared_ptr<const Executor> exec, int max_iters,
       remove_complex<value_type> rel_residual_goal,
       std::shared_ptr<const LinOp> system_matrix)
        : LinOp(exec, system_matrix->get_num_cols(),
                system_matrix->get_num_rows(),
                system_matrix->get_num_rows() * system_matrix->get_num_cols()),
          system_matrix_(std::move(system_matrix)),
          max_iters_(max_iters),
          rel_residual_goal_(rel_residual_goal)
    {}

    static std::unique_ptr<Cg> create(
        std::shared_ptr<const Executor> exec, int max_iters,
        remove_complex<value_type> rel_residual_goal,
        std::shared_ptr<const LinOp> system_matrix)
    {
        return std::unique_ptr<Cg>(new Cg(std::move(exec), max_iters,
                                          rel_residual_goal,
                                          std::move(system_matrix)));
    }

    std::shared_ptr<const LinOp> system_matrix_;
    int max_iters_;
    remove_complex<value_type> rel_residual_goal_;
};

/** The CgFactory class is derived from the LinOpFactory class and is used to
 * generate the Cg solver.
 */
template <typename ValueType = default_precision>
class CgFactory : public LinOpFactory {
public:
    using value_type = ValueType;
    /**
     * Creates the Cg solver.
     *
     * @param exec The executor on which the Cg solver is to be created.
     * @param max_iters  The maximum number of iterations to be pursued.
     * @param rel_residual_goal  The relative residual required for
     * convergence.
     */
    static std::unique_ptr<CgFactory> create(
        std::shared_ptr<const Executor> exec, int max_iters,
        remove_complex<value_type> rel_residual_goal)
    {
        return std::unique_ptr<CgFactory>(
            new CgFactory(std::move(exec), max_iters, rel_residual_goal));
    }

    /**
     * Generates a Cg solver from a base solver.
     *
     * @param base The base Cg solver.
     */
    std::unique_ptr<LinOp> generate(
        std::shared_ptr<const LinOp> base) const override;

    /**
     * Gets the maximum number of iterations of the Cg solver.
     *
     * @return  The maximum number of iterations.
     */
    int get_max_iters() const { return max_iters_; }

    /**
     * Gets the relative residual goal of the solver.
     *
     * @return  The relative residual goal.
     */
    remove_complex<value_type> get_rel_residual_goal() const
    {
        return rel_residual_goal_;
    }

protected:
    explicit CgFactory(std::shared_ptr<const Executor> exec, int max_iters,
                       remove_complex<value_type> rel_residual_goal)
        : LinOpFactory(std::move(exec)),
          max_iters_(max_iters),
          rel_residual_goal_(rel_residual_goal)
    {}

    int max_iters_;
    remove_complex<value_type> rel_residual_goal_;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_CORE_SOLVER_CG_HPP
