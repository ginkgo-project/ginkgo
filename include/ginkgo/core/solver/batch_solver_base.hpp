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


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/utils_helper.hpp>
#include <ginkgo/core/log/batch_logger.hpp>
#include <ginkgo/core/matrix/batch_identity.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>


namespace gko {
namespace batch {
namespace solver {


/**
 * The BatchSolver is a base class for all batched solvers and provides the
 * common getters and setter for these batched solver classes.
 *
 * @ingroup solvers
 */
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
    void set_residual_tolerance(double res_tol)
    {
        if (res_tol < 0) {
            GKO_INVALID_STATE("Tolerance cannot be negative!");
        }
        residual_tol_ = res_tol;
    }

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
        if (max_iterations < 0) {
            GKO_INVALID_STATE("Max iterations cannot be negative!");
        }
        max_iterations_ = max_iterations;
    }

    /**
     * Get the tolerance type.
     *
     * @return  The tolerance type.
     */
    ::gko::batch::stop::tolerance_type get_tolerance_type() const
    {
        return tol_type_;
    }

    /**
     * Set the type of tolerance check to use inside the solver
     *
     * @param tol_type  The tolerance type.
     */
    void set_tolerance_type(::gko::batch::stop::tolerance_type tol_type)
    {
        if (tol_type != ::gko::batch::stop::tolerance_type::absolute ||
            tol_type != ::gko::batch::stop::tolerance_type::relative) {
            GKO_INVALID_STATE("Invalid tolerance type specified!");
        }
        tol_type_ = tol_type;
    }

protected:
    BatchSolver() {}

    BatchSolver(std::shared_ptr<const BatchLinOp> system_matrix,
                std::shared_ptr<const BatchLinOp> gen_preconditioner,
                const double res_tol, const int max_iterations,
                const ::gko::batch::stop::tolerance_type tol_type)
        : system_matrix_{std::move(system_matrix)},
          preconditioner_{std::move(gen_preconditioner)},
          residual_tol_{res_tol},
          max_iterations_{max_iterations},
          tol_type_{tol_type},
          workspace_{}
    {}

    std::shared_ptr<const BatchLinOp> system_matrix_{};
    std::shared_ptr<const BatchLinOp> preconditioner_{};
    double residual_tol_{};
    int max_iterations_{};
    ::gko::batch::stop::tolerance_type tol_type_{};
    mutable array<unsigned char> workspace_{};
};


/**
 * The parameter type shared between all preconditioned iterative solvers,
 * excluding the parameters available in iterative_solver_factory_parameters.
 * @see GKO_CREATE_FACTORY_PARAMETERS
 */
struct preconditioned_iterative_solver_factory_parameters {
    /**
     * The preconditioner to be used by the iterative solver. By default, no
     * preconditioner is used.
     */
    std::shared_ptr<const BatchLinOpFactory> preconditioner{nullptr};

    /**
     * Already generated preconditioner. If one is provided, the factory
     * `preconditioner` will be ignored.
     */
    std::shared_ptr<const BatchLinOp> generated_preconditioner{nullptr};
};


template <typename Parameters, typename Factory>
struct enable_preconditioned_iterative_solver_factory_parameters
    : enable_parameters_type<Parameters, Factory>,
      preconditioned_iterative_solver_factory_parameters {
    /**
     * Default maximum number iterations allowed.
     *
     * Generated solvers are initialized with this value for their maximum
     * iterations.
     */
    int GKO_FACTORY_PARAMETER_SCALAR(default_max_iterations, 100);

    /**
     * Default residual tolerance.
     *
     * Generated solvers are initialized with this value for their residual
     * tolerance.
     */
    double GKO_FACTORY_PARAMETER_SCALAR(default_tolerance, 1e-11);

    /**
     * To specify which type of tolerance check is to be considered, absolute or
     * relative (to the rhs l2 norm)
     */
    ::gko::batch::stop::tolerance_type GKO_FACTORY_PARAMETER_SCALAR(
        tolerance_type, ::gko::batch::stop::tolerance_type::absolute);

    /**
     * Provides a preconditioner factory to be used by the iterative solver in a
     * fluent interface.
     * @see preconditioned_iterative_solver_factory_parameters::preconditioner
     */
    Parameters& with_preconditioner(
        deferred_factory_parameter<BatchLinOpFactory> preconditioner)
    {
        this->preconditioner_generator = std::move(preconditioner);
        this->deferred_factories["preconditioner"] = [](const auto& exec,
                                                        auto& params) {
            if (!params.preconditioner_generator.is_empty()) {
                params.preconditioner =
                    params.preconditioner_generator.on(exec);
            }
        };
        return *self();
    }

    /**
     * Provides a concrete preconditioner to be used by the iterative solver in
     * a fluent interface.
     * @see preconditioned_iterative_solver_factory_parameters::preconditioner
     */
    Parameters& with_generated_preconditioner(
        std::shared_ptr<const BatchLinOp> generated_preconditioner)
    {
        this->generated_preconditioner = std::move(generated_preconditioner);
        return *self();
    }

private:
    GKO_ENABLE_SELF(Parameters);

    deferred_factory_parameter<BatchLinOpFactory> preconditioner_generator;
};


/**
 * This mixin provides apply and common iterative solver functionality to all
 * the batched solvers.
 *
 * @tparam ConcreteSolver  The concrete solver class.
 * @tparam ValueType  The value type of the multivectors.
 * @tparam PolymorphicBase  The base class; must be a subclass of BatchLinOp.
 */
template <typename ConcreteSolver,
          typename ValueType = typename ConcreteSolver::value_type,
          typename PolymorphicBase = BatchLinOp>
class EnableBatchSolver
    : public BatchSolver,
      public EnableBatchLinOp<ConcreteSolver, PolymorphicBase> {
public:
    using real_type = remove_complex<ValueType>;
    const ConcreteSolver* apply(ptr_param<const MultiVector<ValueType>> b,
                                ptr_param<MultiVector<ValueType>> x) const
    {
        this->validate_application_parameters(b.get(), x.get());
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        return self();
    }

    const ConcreteSolver* apply(ptr_param<const MultiVector<ValueType>> alpha,
                                ptr_param<const MultiVector<ValueType>> b,
                                ptr_param<const MultiVector<ValueType>> beta,
                                ptr_param<MultiVector<ValueType>> x) const
    {
        this->validate_application_parameters(alpha.get(), b.get(), beta.get(),
                                              x.get());
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        return self();
    }

    ConcreteSolver* apply(ptr_param<const MultiVector<ValueType>> b,
                          ptr_param<MultiVector<ValueType>> x)
    {
        static_cast<const ConcreteSolver*>(this)->apply(b, x);
        return self();
    }

    ConcreteSolver* apply(ptr_param<const MultiVector<ValueType>> alpha,
                          ptr_param<const MultiVector<ValueType>> b,
                          ptr_param<const MultiVector<ValueType>> beta,
                          ptr_param<MultiVector<ValueType>> x)
    {
        static_cast<const ConcreteSolver*>(this)->apply(alpha, b, beta, x);
        return self();
    }

protected:
    GKO_ENABLE_SELF(ConcreteSolver);

    explicit EnableBatchSolver(std::shared_ptr<const Executor> exec)
        : EnableBatchLinOp<ConcreteSolver, PolymorphicBase>(std::move(exec))
    {}

    template <typename FactoryParameters>
    explicit EnableBatchSolver(std::shared_ptr<const Executor> exec,
                               std::shared_ptr<const BatchLinOp> system_matrix,
                               const FactoryParameters& params)
        : BatchSolver(system_matrix, nullptr, params.default_tolerance,
                      params.default_max_iterations, params.tolerance_type),
          EnableBatchLinOp<ConcreteSolver, PolymorphicBase>(
              exec, gko::transpose(system_matrix->get_size()))
    {
        GKO_ASSERT_BATCH_HAS_SQUARE_DIMENSIONS(system_matrix_);

        using value_type = typename ConcreteSolver::value_type;
        using Identity = matrix::Identity<value_type>;
        using real_type = remove_complex<value_type>;

        if (params.generated_preconditioner) {
            GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(params.generated_preconditioner,
                                              this);
            preconditioner_ = std::move(params.generated_preconditioner);
        } else if (params.preconditioner) {
            preconditioner_ = params.preconditioner->generate(system_matrix_);
        } else {
            auto id = Identity::create(exec, system_matrix->get_size());
            preconditioner_ = std::move(id);
        }
        // FIXME
        const size_type workspace_size = system_matrix->get_num_batch_items() *
                                         (sizeof(real_type) + sizeof(int));
        workspace_.set_executor(exec);
        workspace_.resize_and_reset(workspace_size);
    }

    void apply_impl(const MultiVector<ValueType>* b,
                    MultiVector<ValueType>* x) const
    {
        auto exec = this->get_executor();
        if (b->get_common_size()[1] > 1) {
            GKO_NOT_IMPLEMENTED;
        }
        auto log_data_ = std::make_unique<log::detail::log_data<real_type>>(
            exec, b->get_num_batch_items(), workspace_.as_view());

        this->solver_apply(b, x, log_data_.get());

        // TODO: This needs to allocate data with every call.
        this->template log<gko::log::Logger::batch_solver_completed>(
            log_data_->iter_counts, log_data_->res_norms);
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

    virtual void solver_apply(const MultiVector<ValueType>* b,
                              MultiVector<ValueType>* x,
                              log::detail::log_data<real_type>* info) const = 0;
};


}  // namespace solver
}  // namespace batch
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_BATCH_SOLVER_HPP_
