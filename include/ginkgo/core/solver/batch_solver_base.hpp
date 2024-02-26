// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_BATCH_SOLVER_BASE_HPP_
#define GKO_PUBLIC_CORE_SOLVER_BATCH_SOLVER_BASE_HPP_


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
        return this->system_matrix_;
    }

    /**
     * Returns the generated preconditioner.
     *
     * @return the generated preconditioner.
     */
    std::shared_ptr<const BatchLinOp> get_preconditioner() const
    {
        return this->preconditioner_;
    }

    /**
     * Get the residual tolerance used by the solver.
     *
     * @return The residual tolerance.
     */
    double get_tolerance() const { return this->residual_tol_; }

    /**
     * Update the residual tolerance to be used by the solver.
     *
     * @param res_tol  The residual tolerance to be used for subsequent
     *                 invocations of the solver.
     */
    void reset_tolerance(double res_tol)
    {
        if (res_tol < 0) {
            GKO_INVALID_STATE("Tolerance cannot be negative!");
        }
        this->residual_tol_ = res_tol;
    }

    /**
     * Get the maximum number of iterations set on the solver.
     *
     * @return  Maximum number of iterations.
     */
    int get_max_iterations() const { return this->max_iterations_; }

    /**
     * Set the maximum number of iterations for the solver to use,
     * independent of the factory that created it.
     *
     * @param max_iterations  The maximum number of iterations for the solver.
     */
    void reset_max_iterations(int max_iterations)
    {
        if (max_iterations < 0) {
            GKO_INVALID_STATE("Max iterations cannot be negative!");
        }
        this->max_iterations_ = max_iterations;
    }

    /**
     * Get the tolerance type.
     *
     * @return  The tolerance type.
     */
    ::gko::batch::stop::tolerance_type get_tolerance_type() const
    {
        return this->tol_type_;
    }

    /**
     * Set the type of tolerance check to use inside the solver
     *
     * @param tol_type  The tolerance type.
     */
    void reset_tolerance_type(::gko::batch::stop::tolerance_type tol_type)
    {
        if (tol_type == ::gko::batch::stop::tolerance_type::absolute ||
            tol_type == ::gko::batch::stop::tolerance_type::relative) {
            this->tol_type_ = tol_type;
        } else {
            GKO_INVALID_STATE("Invalid tolerance type specified!");
        }
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

    void set_system_matrix_base(std::shared_ptr<const BatchLinOp> system_matrix)
    {
        this->system_matrix_ = std::move(system_matrix);
    }

    void set_preconditioner_base(std::shared_ptr<const BatchLinOp> precond)
    {
        this->preconditioner_ = std::move(precond);
    }

    std::shared_ptr<const BatchLinOp> system_matrix_{};
    std::shared_ptr<const BatchLinOp> preconditioner_{};
    double residual_tol_{};
    int max_iterations_{};
    ::gko::batch::stop::tolerance_type tol_type_{};
    mutable array<unsigned char> workspace_{};
};


template <typename Parameters, typename Factory>
struct enable_preconditioned_iterative_solver_factory_parameters
    : enable_parameters_type<Parameters, Factory> {
    /**
     * Default maximum number iterations allowed.
     *
     * Generated solvers are initialized with this value for their maximum
     * iterations.
     */
    int GKO_FACTORY_PARAMETER_SCALAR(max_iterations, 100);

    /**
     * Default residual tolerance.
     *
     * Generated solvers are initialized with this value for their residual
     * tolerance.
     */
    double GKO_FACTORY_PARAMETER_SCALAR(tolerance, 1e-11);

    /**
     * To specify which type of tolerance check is to be considered, absolute or
     * relative (to the rhs l2 norm)
     */
    ::gko::batch::stop::tolerance_type GKO_FACTORY_PARAMETER_SCALAR(
        tolerance_type, ::gko::batch::stop::tolerance_type::absolute);

    /**
     * The preconditioner to be used by the iterative solver. By default, no
     * preconditioner is used.
     */
    std::shared_ptr<const BatchLinOpFactory> GKO_DEFERRED_FACTORY_PARAMETER(
        preconditioner);

    /**
     * Already generated preconditioner. If one is provided, the factory
     * `preconditioner` will be ignored.
     */
    std::shared_ptr<const BatchLinOp> GKO_FACTORY_PARAMETER_SCALAR(
        generated_preconditioner, nullptr);
};


/**
 * This mixin provides apply and common iterative solver functionality to all
 * the batched solvers.
 *
 * @tparam ConcreteSolver  The concrete solver class.
 * @tparam ValueType  The value type of the multivectors.
 * @tparam PolymorphicBase  The base class; must be a subclass of BatchLinOp.
 */
template <typename ConcreteSolver, typename ValueType,
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
        this->validate_application_parameters(b.get(), x.get());
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        return self();
    }

    ConcreteSolver* apply(ptr_param<const MultiVector<ValueType>> alpha,
                          ptr_param<const MultiVector<ValueType>> b,
                          ptr_param<const MultiVector<ValueType>> beta,
                          ptr_param<MultiVector<ValueType>> x)
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

protected:
    GKO_ENABLE_SELF(ConcreteSolver);

    explicit EnableBatchSolver(std::shared_ptr<const Executor> exec)
        : EnableBatchLinOp<ConcreteSolver, PolymorphicBase>(std::move(exec))
    {}

    template <typename FactoryParameters>
    explicit EnableBatchSolver(std::shared_ptr<const Executor> exec,
                               std::shared_ptr<const BatchLinOp> system_matrix,
                               const FactoryParameters& params)
        : BatchSolver(system_matrix, nullptr, params.tolerance,
                      params.max_iterations, params.tolerance_type),
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
        const size_type workspace_size = system_matrix->get_num_batch_items() *
                                         (sizeof(real_type) + sizeof(int));
        workspace_.set_executor(exec);
        workspace_.resize_and_reset(workspace_size);
    }

    void set_system_matrix(std::shared_ptr<const BatchLinOp> new_system_matrix)
    {
        auto exec = self()->get_executor();
        if (new_system_matrix) {
            GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(self(), new_system_matrix);
            GKO_ASSERT_BATCH_HAS_SQUARE_DIMENSIONS(new_system_matrix);
            if (new_system_matrix->get_executor() != exec) {
                new_system_matrix = gko::clone(exec, new_system_matrix);
            }
        }
        this->set_system_matrix_base(new_system_matrix);
    }

    void set_preconditioner(std::shared_ptr<const BatchLinOp> new_precond)
    {
        auto exec = self()->get_executor();
        if (new_precond) {
            GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(self(), new_precond);
            GKO_ASSERT_BATCH_HAS_SQUARE_DIMENSIONS(new_precond);
            if (new_precond->get_executor() != exec) {
                new_precond = gko::clone(exec, new_precond);
            }
        }
        this->set_preconditioner_base(new_precond);
    }

    EnableBatchSolver& operator=(const EnableBatchSolver& other)
    {
        if (&other != this) {
            this->set_size(other.get_size());
            this->set_system_matrix(other.get_system_matrix());
            this->set_preconditioner(other.get_preconditioner());
            this->reset_tolerance(other.get_tolerance());
            this->reset_max_iterations(other.get_max_iterations());
            this->reset_tolerance_type(other.get_tolerance_type());
        }

        return *this;
    }

    EnableBatchSolver& operator=(EnableBatchSolver&& other)
    {
        if (&other != this) {
            this->set_size(other.get_size());
            this->set_system_matrix(other.get_system_matrix());
            this->set_preconditioner(other.get_preconditioner());
            this->reset_tolerance(other.get_tolerance());
            this->reset_max_iterations(other.get_max_iterations());
            this->reset_tolerance_type(other.get_tolerance_type());
            other.set_system_matrix(nullptr);
            other.set_preconditioner(nullptr);
        }
        return *this;
    }

    EnableBatchSolver(const EnableBatchSolver& other)
        : EnableBatchLinOp<ConcreteSolver, PolymorphicBase>(
              other.self()->get_executor(), other.self()->get_size())
    {
        *this = other;
    }

    EnableBatchSolver(EnableBatchSolver&& other)
        : EnableBatchLinOp<ConcreteSolver, PolymorphicBase>(
              other.self()->get_executor(), other.self()->get_size())
    {
        *this = std::move(other);
    }

    void apply_impl(const MultiVector<ValueType>* b,
                    MultiVector<ValueType>* x) const
    {
        auto exec = this->get_executor();
        if (b->get_common_size()[1] > 1) {
            GKO_NOT_IMPLEMENTED;
        }
        auto workspace_view = workspace_.as_view();
        auto log_data_ = std::make_unique<log::detail::log_data<real_type>>(
            exec, b->get_num_batch_items(), workspace_view);

        this->solver_apply(b, x, log_data_.get());

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


#endif  // GKO_PUBLIC_CORE_SOLVER_BATCH_SOLVER_BASE_HPP_
