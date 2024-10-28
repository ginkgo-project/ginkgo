// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/log/solver_progress.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/solver_base.hpp>

#include "core/base/dispatch_helper.hpp"


namespace gko {
namespace log {
namespace {


bool is_dense(const LinOp* value)
{
    using conv_to_double = ConvertibleTo<matrix::Dense<double>>;
    using conv_to_complex = ConvertibleTo<matrix::Dense<std::complex<double>>>;
    return dynamic_cast<const conv_to_double*>(value) ||
           dynamic_cast<const conv_to_complex*>(value);
}


class SolverProgressPrint : public SolverProgress {
    friend class SolverProgress;

public:
    /* Internal solver events */
    void on_linop_apply_started(const LinOp* solver, const LinOp* in,
                                const LinOp* out) const override
    {
        printed_header_ = false;
    }

    void on_iteration_complete(
        const LinOp* solver, const LinOp* right_hand_side,
        const LinOp* solution, const size_type& num_iterations,
        const LinOp* residual, const LinOp* residual_norm,
        const LinOp* implicit_sq_residual_norm,
        const array<stopping_status>* status, bool stopped) const override
    {
        using solver_base = solver::detail::SolverBaseLinOp;
        auto dynamic_type = name_demangling::get_dynamic_type(*solver);
        auto& stream = *output_;
        auto base = gko::as<solver_base>(solver);
        if (!printed_header_) {
            stream << dynamic_type << "::apply(" << right_hand_side << ','
                   << solution << ") of dimensions " << solver->get_size()
                   << " and " << right_hand_side->get_size()[1] << " rhs\n";
            const auto scalars = base->get_workspace_scalars();
            const auto names = base->get_workspace_op_names();
            stream << std::setw(column_width_) << "Iteration";
            for (auto scalar : scalars) {
                if (separator_) {
                    stream << separator_;
                }
                stream << std::setw(column_width_) << names[scalar];
            }
            if (residual_norm) {
                if (separator_) {
                    stream << separator_;
                }
                stream << std::setw(column_width_) << "residual_norm";
            }
            if (implicit_sq_residual_norm) {
                if (separator_) {
                    stream << separator_;
                }
                stream << std::setw(column_width_)
                       << "implicit_sq_residual_norm";
            }
            stream << '\n';
            printed_header_ = true;
        }
        stream << std::setprecision(precision_);
        const auto scalars = base->get_workspace_scalars();
        stream << std::setw(column_width_) << num_iterations;
        for (auto scalar : scalars) {
            print_scalar(base->get_workspace_op(scalar), stream);
        }
        if (residual_norm) {
            print_scalar(residual_norm, stream);
        }
        if (implicit_sq_residual_norm) {
            print_scalar(implicit_sq_residual_norm, stream);
        }
        stream << '\n';
    }

    GKO_DEPRECATED(
        "Please use the version with the additional stopping "
        "information.")
    void on_iteration_complete(const LinOp* solver,
                               const size_type& num_iterations,
                               const LinOp* residual, const LinOp* solution,
                               const LinOp* residual_norm) const override
    {
        on_iteration_complete(solver, nullptr, solution, num_iterations,
                              residual, residual_norm, nullptr, nullptr, false);
    }

    GKO_DEPRECATED(
        "Please use the version with the additional stopping "
        "information.")
    void on_iteration_complete(
        const LinOp* solver, const size_type& num_iterations,
        const LinOp* residual, const LinOp* solution,
        const LinOp* residual_norm,
        const LinOp* implicit_sq_residual_norm) const override
    {
        on_iteration_complete(solver, nullptr, solution, num_iterations,
                              residual, residual_norm,
                              implicit_sq_residual_norm, nullptr, false);
    }

private:
    void print_scalar(const LinOp* value, std::ostream& stream) const
    {
        if (separator_) {
            stream << separator_;
        }
        stream << std::setw(column_width_);
        if (!value->get_size()) {
            stream << "<empty>";
        } else if (value->get_size()[0] != 1) {
            stream << "<vector>";
        } else if (is_dense(value)) {
            auto host_exec = value->get_executor()->get_master();
            run<ConvertibleTo<matrix::Dense<double>>,
                ConvertibleTo<matrix::Dense<std::complex<double>>>>(
                value, [&](auto vector) {
                    using vector_type =
                        typename detail::pointee<decltype(vector)>::result_type;
                    auto host_vec = vector_type::create(host_exec);
                    vector->convert_to(host_vec);
                    stream << host_vec->at(0, 0);
                });

        } else {
            stream << "<unknown>";
        }
    }

    SolverProgressPrint(std::ostream& output, int precision, int column_width,
                        char separator)
        : output_{&output},
          precision_{precision},
          column_width_{column_width},
          separator_{separator},
          printed_header_(false)
    {}

    std::ostream* output_;
    int precision_;
    int column_width_;
    char separator_;
    mutable bool printed_header_;
};


class SolverProgressStore : public SolverProgress {
    friend class SolverProgress;

public:
    /* Internal solver events */
    void on_linop_apply_started(const LinOp* solver, const LinOp* in,
                                const LinOp* out) const override
    {
        using solver_base = solver::detail::SolverBaseLinOp;
        auto dynamic_type = name_demangling::get_dynamic_type(*solver);
        auto base = gko::as<solver_base>(solver);
        store_vector(base->get_system_matrix().get(), "system_matrix");
        store_vector(in, "rhs");
        store_vector(out, "initial_guess");
    }

    void on_iteration_complete(
        const LinOp* solver, const LinOp* right_hand_side,
        const LinOp* solution, const size_type& num_iterations,
        const LinOp* residual, const LinOp* residual_norm,
        const LinOp* implicit_sq_residual_norm,
        const array<stopping_status>* status, bool stopped) const override
    {
        using solver_base = solver::detail::SolverBaseLinOp;
        auto base = gko::as<solver_base>(solver);
        const auto num_vectors = base->get_num_workspace_ops();
        const auto names = base->get_workspace_op_names();
        for (int i = 0; i < num_vectors; i++) {
            store_vector(base->get_workspace_op(i), num_iterations,
                         base->get_workspace_op_names()[i]);
        }
        store_vector(solution, num_iterations, "solution");
        store_vector(residual, num_iterations, "residual");
        store_vector(residual_norm, num_iterations, "residual_norm");
        store_vector(implicit_sq_residual_norm, num_iterations,
                     "implicit_sq_residual_norm");
    }

    GKO_DEPRECATED(
        "Please use the version with the additional stopping "
        "information.")
    void on_iteration_complete(const LinOp* solver,
                               const size_type& num_iterations,
                               const LinOp* residual, const LinOp* solution,
                               const LinOp* residual_norm) const override
    {
        on_iteration_complete(solver, nullptr, solution, num_iterations,
                              residual, residual_norm, nullptr, nullptr, false);
    }

    GKO_DEPRECATED(
        "Please use the version with the additional stopping "
        "information.")
    void on_iteration_complete(
        const LinOp* solver, const size_type& num_iterations,
        const LinOp* residual, const LinOp* solution,
        const LinOp* residual_norm,
        const LinOp* implicit_sq_residual_norm) const override
    {
        on_iteration_complete(solver, nullptr, solution, num_iterations,
                              residual, residual_norm,
                              implicit_sq_residual_norm, nullptr, false);
    }

private:
    void store_vector(const LinOp* value, const std::string& name) const
    {
        const auto filename =
            output_file_prefix_ + "_" + name + (binary_ ? ".bin" : ".mtx");
        if (!value) {
            return;
        }
        // putting Dense first here causes gko::write to use dense output
        run<gko::matrix::Dense<double>, gko::matrix::Dense<float>,
            gko::matrix::Dense<std::complex<double>>,
            gko::matrix::Dense<std::complex<float>>,
#if GINKGO_ENABLE_HALF
            gko::matrix::Dense<gko::half>,
            gko::matrix::Dense<std::complex<gko::half>>,
            gko::WritableToMatrixData<gko::half, int32>,
            gko::WritableToMatrixData<std::complex<gko::half>, int32>,
            gko::WritableToMatrixData<gko::half, int64>,
            gko::WritableToMatrixData<std::complex<gko::half>, int64>,
#endif
            // fallback for other matrix types
            gko::WritableToMatrixData<double, int32>,
            gko::WritableToMatrixData<float, int32>,
            gko::WritableToMatrixData<std::complex<double>, int32>,
            gko::WritableToMatrixData<std::complex<float>, int32>,
            gko::WritableToMatrixData<double, int64>,
            gko::WritableToMatrixData<float, int64>,
            gko::WritableToMatrixData<std::complex<double>, int64>,
            gko::WritableToMatrixData<std::complex<float>, int64>>(
            value, [&](auto vector) {
                std::ofstream output{
                    filename, binary_ ? (std::ios::out | std::ios::binary)
                                      : std::ios::out};
                if (binary_) {
                    gko::write_binary(output, vector);
                } else {
                    gko::write(output, vector);
                }
            });
    }

    void store_vector(const LinOp* value, size_type iteration,
                      const std::string& name) const
    {
        store_vector(value, std::to_string(iteration) + "_" + name);
    }

    SolverProgressStore(std::string output_file_prefix, bool binary)
        : output_file_prefix_{std::move(output_file_prefix)}, binary_{binary}
    {}

    std::string output_file_prefix_;
    bool binary_;
};


}  // namespace


std::shared_ptr<SolverProgress> SolverProgress::create_scalar_table_writer(
    std::ostream& output, int precision, int column_width)
{
    return std::shared_ptr<SolverProgress>{
        new SolverProgressPrint{output, precision, column_width, '\0'}};
}


std::shared_ptr<SolverProgress> SolverProgress::create_scalar_csv_writer(
    std::ostream& output, int precision, char separator)
{
    return std::shared_ptr<SolverProgress>{
        new SolverProgressPrint{output, precision, 0, separator}};
}


std::shared_ptr<SolverProgress> SolverProgress::create_vector_storage(
    std::string output_file_prefix, bool binary)
{
    return std::shared_ptr<SolverProgress>{
        new SolverProgressStore{output_file_prefix, binary}};
}


}  // namespace log
}  // namespace gko
