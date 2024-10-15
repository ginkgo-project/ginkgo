// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_EXAMPLES_BATCHED - MATRIX - FREE - TEMPLATED_BATCHED_BATCH_CG_HPP_
#define GKO_EXAMPLES_BATCHED -MATRIX - FREE - TEMPLATED_BATCHED_BATCH_CG_HPP_


#include <vector>

#include <batch_cg_kernels.hpp>
#include <batch_multi_vector.hpp>

#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/batch_logger.hpp>
#include <ginkgo/core/solver/batch_solver_base.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>


namespace gko::batch_template::solver {
namespace cg {


GKO_REGISTER_OPERATION(apply, batch_tempalte::batch_cg::apply);


}

template <typename T>
class BatchSolver {
public:
    std::shared_ptr<const T> get_system_matrix() const
    {
        return this->system_matrix_;
    }

    double get_tolerance() const { return this->residual_tol_; }

    int get_max_iterations() const { return this->max_iterations_; }

    batch::stop::tolerance_type get_tolerance_type() const
    {
        return this->tol_type_;
    }

protected:
    BatchSolver() {}

    BatchSolver(std::shared_ptr<const T> system_matrix, const double res_tol,
                const int max_iterations,
                const ::gko::batch::stop::tolerance_type tol_type)
        : system_matrix_{std::move(system_matrix)},
          residual_tol_{res_tol},
          max_iterations_{max_iterations},
          tol_type_{tol_type},
          workspace_{}
    {}

    void set_system_matrix_base(std::shared_ptr<const T> system_matrix)
    {
        this->system_matrix_ = std::move(system_matrix);
    }

    std::shared_ptr<const T> system_matrix_{};
    double residual_tol_{};
    int max_iterations_{};
    ::gko::batch::stop::tolerance_type tol_type_{};
    mutable array<unsigned char> workspace_{};
};

template <typename T>
class Cg final : public BatchSolver<T>, public batch::EnableBatchLinOp<Cg<T>> {
    friend class batch::EnableBatchLinOp<Cg>;
    friend class EnablePolymorphicObject<Cg, batch::BatchLinOp>;

public:
    using value_type = typename T::value_type;
    using real_type = remove_complex<value_type>;

    class Factory;

    struct parameters_type
        : batch::solver::
              enable_preconditioned_iterative_solver_factory_parameters<
                  parameters_type, Factory> {};

    const parameters_type& get_parameters() const { return parameters_; }

    class Factory : public log::EnableLogging<Factory> {
    public:
        explicit Factory(std::shared_ptr<const Executor> exec)
            : exec_(std::move(exec))
        {}

        explicit Factory(std::shared_ptr<const Executor> exec,
                         const parameters_type& parameters)
            : exec_(std::move(exec)), parameters_(parameters)
        {}

        std::unique_ptr<Cg> generate(std::shared_ptr<T> input) const
        {
            if (input->get_executor() != exec_) {
                input = gko::clone(exec_, input);
            }
            return std::unique_ptr<Cg>(new Cg(this, input));
        }

        const parameters_type& get_parameters() const noexcept
        {
            return parameters_;
        };

        auto get_executor() const { return exec_; }

    private:
        std::shared_ptr<const Executor> exec_;
        parameters_type parameters_;
    };

    static auto build() -> parameters_type { return {}; }

    void apply(ptr_param<const MultiVector<value_type>> b,
               ptr_param<MultiVector<value_type>> x) const
    {
        // this->validate_application_parameters(b.get(), x.get());
        auto exec = this->get_executor();
        array<unsigned char> workspace_{exec};
        auto log_data_ =
            std::make_unique<batch::log::detail::log_data<real_type>>(
                exec, b->get_num_batch_items(), workspace_);
        this->solver_apply(make_temporary_clone(exec, b).get(),
                           make_temporary_clone(exec, x).get(),
                           log_data_.get());
    }

private:
    explicit Cg(std::shared_ptr<const Executor> exec)
        : batch::EnableBatchLinOp<Cg>(std::move(exec))
    {}

    explicit Cg(const Factory* factory, std::shared_ptr<const T> system_matrix)
        : BatchSolver<T>(std::move(system_matrix),
                         factory->get_parameters().tolerance,
                         factory->get_parameters().max_iterations,
                         factory->get_parameters().tolerance_type),
          batch::EnableBatchLinOp<Cg>(factory->get_executor(),
                                      transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()}
    {}

    void solver_apply(const MultiVector<value_type>* b,
                      MultiVector<value_type>* x,
                      batch::log::detail::log_data<real_type>* log_data) const
    {
        const kernels::batch_cg::settings<remove_complex<value_type>> settings{
            this->max_iterations_, static_cast<real_type>(this->residual_tol_),
            parameters_.tolerance_type};
        auto exec = this->get_executor();
        exec->run(cg::make_apply(settings, this->system_matrix_.get(),
                                 b->create_view(), x->create_view(),
                                 *log_data));
    }

    parameters_type parameters_;
};


}  // namespace gko::batch_template::solver


#endif  // GKO_EXAMPLES_BATCHED-MATRIX-FREE-TEMPLATED_BATCHED_BATCH_CG_HPP_
