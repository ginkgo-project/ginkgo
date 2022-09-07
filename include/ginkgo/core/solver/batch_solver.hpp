/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_SOLVER_BATCH_SOLVER_HPP_
#define GKO_PUBLIC_CORE_SOLVER_BATCH_SOLVER_HPP_


#include <ginkgo/core/matrix/batch_dense.hpp>


namespace gko {
namespace solver {


class BatchSolver {
public:
    /**
     * Returns the system operator (matrix) of the linear system.
     *
     * @return the system operator (matrix)
     */
    std::shared_ptr<const BatchLinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    /**
     * Returns the generated preconditioner.
     *
     * @return the generated preconditioner.
     */
    std::shared_ptr<const BatchLinOp> get_preconditioner() const
    {
        return preconditioner_;
    }

    /**
     * @return The left scaling vector.
     */
    std::shared_ptr<const BatchLinOp> get_left_scaling_op() const
    {
        return left_scaling_;
    }

    /**
     * @return The right scaling vector.
     */
    std::shared_ptr<const BatchLinOp> get_right_scaling_op() const
    {
        return right_scaling_;
    }

    /**
     * Get the residual tolerance used by the solver.
     *
     * @return The residual tolerance.
     */
    double get_residual_tolerance() const { return residual_tol_; }

    /**
     * Update the residual tolerance to be used by the solver.
     *
     * @param res_tol  The residual tolerance to be used for subsequent
     *                 invocations of the solver.
     */
    void set_residual_tolerance(double res_tol) { residual_tol_ = res_tol; }

    /**
     * Get the maximum number of iterations set on the solver.
     *
     * @return  Maximum number of iterations.
     */
    int get_max_iterations() const { return max_iterations_; }

    /**
     * Set the maximum number of iterations for the solver to use,
     * independent of the factory that created it.
     *
     * @param max_iterations  The maximum number of iterations for the solver.
     */
    void set_max_iterations(int max_iterations)
    {
        max_iterations_ = max_iterations;
    }

protected:
    BatchSolver() {}

    BatchSolver(std::shared_ptr<const BatchLinOp> system_matrix,
                std::shared_ptr<const BatchLinOp> gen_preconditioner,
                std::shared_ptr<const BatchLinOp> left_scaling,
                std::shared_ptr<const BatchLinOp> right_scaling,
                const double res_tol, const int max_iterations)
        : system_matrix_{std::move(system_matrix)},
          preconditioner_{std::move(gen_preconditioner)},
          left_scaling_{std::move(left_scaling)},
          right_scaling_{std::move(right_scaling)},
          residual_tol_{res_tol},
          max_iterations_{max_iterations}
    {}

    std::shared_ptr<const BatchLinOp> system_matrix_{};
    std::shared_ptr<const BatchLinOp> preconditioner_{};
    std::shared_ptr<const BatchLinOp> left_scaling_{};
    std::shared_ptr<const BatchLinOp> right_scaling_{};
    double residual_tol_{};
    int max_iterations_{};
};


namespace detail {


struct common_batch_params {
    std::shared_ptr<const BatchLinOpFactory> prec_factory;
    std::shared_ptr<const BatchLinOp> generated_prec;
    std::shared_ptr<const BatchLinOp> left_scaling_op;
    std::shared_ptr<const BatchLinOp> right_scaling_op;
    double residual_tolerance;
    int max_iterations;
};


template <typename ParamsType>
common_batch_params extract_common_batch_params(ParamsType& params)
{
    return {params.preconditioner,       params.generated_preconditioner,
            params.left_scaling_op,      params.right_scaling_op,
            params.default_residual_tol, params.default_max_iterations};
}


}  // namespace detail


struct BatchInfo;


/**
 * @tparam PolymorphicBase  The base class; must be a subclass of BatchLinOp.
 */
template <typename ConcreteSolver, typename PolymorphicBase = BatchLinOp>
class EnableBatchSolver
    : public BatchSolver,
      public EnableBatchLinOp<ConcreteSolver, PolymorphicBase> {
protected:
    explicit EnableBatchSolver(std::shared_ptr<const Executor> exec)
        : EnableBatchLinOp<ConcreteSolver, PolymorphicBase>(std::move(exec))
    {}

    explicit EnableBatchSolver(std::shared_ptr<const Executor> exec,
                               std::shared_ptr<const BatchLinOp> system_matrix,
                               detail::common_batch_params common_params);

    void apply_impl(const BatchLinOp* b, BatchLinOp* x) const override;

    void apply_impl(const BatchLinOp* alpha, const BatchLinOp* b,
                    const BatchLinOp* beta, BatchLinOp* x) const override;

private:
    /**
     * Calls the concrete solver on the given system (not necessarily on
     * system_matrix_).
     *
     * @param mtx  Left-hand side matrix for the linear solve.
     * @param b  Right-hand side vector.
     * @param x  Solution vector and initial guess.
     * @param info  Batch logging information.
     */
    virtual void solver_apply(const BatchLinOp* b, BatchLinOp* x,
                              BatchInfo* const info) const = 0;
};


}  // namespace solver
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_SOLVER_BATCH_SOLVER_HPP_
