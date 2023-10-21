/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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


#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/log/batch_logger.hpp>
#include <ginkgo/core/matrix/batch_identity.hpp>


namespace gko {
namespace batch {
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
                const double res_tol, const int max_iterations)
        : system_matrix_{std::move(system_matrix)},
          preconditioner_{std::move(gen_preconditioner)},
          residual_tol_{res_tol},
          max_iterations_{max_iterations}
    {}

    std::shared_ptr<const BatchLinOp> system_matrix_{};
    std::shared_ptr<const BatchLinOp> preconditioner_{};
    double residual_tol_{};
    int max_iterations_{};
};


namespace detail {


struct common_batch_params {
    std::shared_ptr<const BatchLinOpFactory> prec_factory;
    std::shared_ptr<const BatchLinOp> generated_prec;
    double residual_tolerance;
    int max_iterations;
};


template <typename ParamsType>
common_batch_params extract_common_batch_params(ParamsType& params)
{
    return {params.preconditioner, params.generated_preconditioner,
            params.default_residual_tol, params.default_max_iterations};
}


}  // namespace detail


struct BatchInfo {
    std::unique_ptr<log::BatchLogDataBase> logdata;
};


/**
 * @tparam PolymorphicBase  The base class; must be a subclass of BatchLinOp.
 */
template <typename ConcreteSolver,
          typename ValueType = typename ConcreteSolver::value_type,
          typename PolymorphicBase = BatchLinOp>
class EnableBatchSolver
    : public BatchSolver,
      public EnableBatchLinOp<ConcreteSolver, PolymorphicBase> {
protected:
    explicit EnableBatchSolver(std::shared_ptr<const Executor> exec)
        : EnableBatchLinOp<ConcreteSolver, PolymorphicBase>(std::move(exec))
    {}

    explicit EnableBatchSolver(std::shared_ptr<const Executor> exec,
                               std::shared_ptr<const BatchLinOp> system_matrix,
                               detail::common_batch_params common_params)
        : BatchSolver(system_matrix, nullptr, common_params.residual_tolerance,
                      common_params.max_iterations),
          EnableBatchLinOp<ConcreteSolver, PolymorphicBase>(
              exec, gko::transpose(system_matrix->get_size()))
    {
        GKO_ASSERT_BATCH_HAS_SQUARE_DIMENSIONS(system_matrix_);

        using value_type = typename ConcreteSolver::value_type;
        using Identity = matrix::BatchIdentity<value_type>;
        using real_type = remove_complex<value_type>;

        if (common_params.generated_prec) {
            GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(common_params.generated_prec,
                                              this);
            preconditioner_ = std::move(common_params.generated_prec);
        } else if (common_params.prec_factory) {
            preconditioner_ =
                common_params.prec_factory->generate(system_matrix_);
        } else {
            auto id = Identity::create(exec, system_matrix->get_size());
            preconditioner_ = std::move(id);
        }
    }

    ConcreteSolver* apply(ptr_param<const MultiVector<ValueType>> b,
                          ptr_param<MultiVector<ValueType>> x) const
    {
        this->validate_application_parameters(b.get(), x.get());
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        return this;
    }

    ConcreteSolver* apply_impl(ptr_param<const MultiVector<ValueType>>* alpha,
                               ptr_param<const MultiVector<ValueType>>* b,
                               ptr_param<const MultiVector<ValueType>>* beta,
                               ptr_param<MultiVector<ValueType>>* x) const
    {
        this->validate_application_parameters(alpha.get(), b.get(), beta.get(),
                                              x.get());
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        return this;
    }

    void apply_impl(const MultiVector<ValueType>* b,
                    MultiVector<ValueType>* x) const
    {
        using value_type = ValueType;
        using Vector = MultiVector<value_type>;
        using res_log_type = double;

        auto exec = this->get_executor();

        const size_type num_rhs = b->get_common_size()[0];
        const size_type num_batch_items = b->get_num_batch_items();
        batch_dim<2> batch_size(num_batch_items, dim<2>{1, num_rhs});

        BatchInfo info;
        info.logdata =
            std::move(std::make_unique<log::BatchLogData<res_log_type>>());
        auto concrete_logdata =
            static_cast<log::BatchLogData<res_log_type>*>(info.logdata.get());
        concrete_logdata->res_norms =
            MultiVector<res_log_type>::create(this->get_executor(), batch_size);
        concrete_logdata->iter_counts.set_executor(this->get_executor());
        concrete_logdata->iter_counts.resize_and_reset(num_rhs *
                                                       num_batch_items);

        this->solver_apply(b, x, &info);

        this->template log<gko::log::Logger::batch_solver_completed>(
            concrete_logdata->iter_counts, concrete_logdata->res_norms.get());
    }

    void apply_impl(const MultiVector<ValueType>* alpha,
                    const MultiVector<ValueType>* b,
                    const MultiVector<ValueType>* beta,
                    MultiVector<ValueType>* x) const
    {
        auto x_clone = x->clone();
        this->apply(b, x_clone.get());
        x->scale(beta);
        x->add_scaled(alpha, x_clone.get());
    }

    /**
     * Calls the concrete solver on the given system (not necessarily on
     * system_matrix_).
     *
     * @param mtx  Left-hand side matrix for the linear solve.
     * @param b  Right-hand side vector.
     * @param x  Solution vector and initial guess.
     * @param info  Batch logging information.
     */
    virtual void solver_apply(const MultiVector<ValueType>* b,
                              MultiVector<ValueType>* x,
                              BatchInfo* const info) const = 0;
};


}  // namespace solver
}  // namespace batch
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_BATCH_SOLVER_HPP_