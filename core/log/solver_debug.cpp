// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/log/solver_debug.hpp>


#include <iomanip>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


namespace gko {
namespace log {


static void print_scalar(const LinOp* value, std::ostream& stream)
{
    using conv_to_double = ConvertibleTo<matrix::Dense<double>>;
    using conv_to_complex = ConvertibleTo<matrix::Dense<std::complex<double>>>;
    const auto host_exec = value->get_executor()->get_master();
    if (value->get_size()[0] == 0) {
        stream << "<empty>";
    } else if (value->get_size()[0] != 1) {
        stream << "<matrix>";
    } else if (dynamic_cast<const conv_to_double*>(value)) {
        auto host_value = matrix::Dense<double>::create(host_exec);
        host_value->copy_from(value);
        stream << host_value->at(0, 0);
    } else if (dynamic_cast<const conv_to_complex*>(value)) {
        auto host_value =
            matrix::Dense<std::complex<double>>::create(host_exec);
        host_value->copy_from(value);
        stream << host_value->at(0, 0);
    } else {
        stream << "<unknown>";
    }
}


void SolverDebug::on_linop_apply_started(const LinOp* solver, const LinOp* in,
                                         const LinOp* out) const
{
    using solver_base = solver::detail::SolverBaseLinOp;
    auto dynamic_type = name_demangling::get_dynamic_type(*solver);
    auto& stream = *output_;
    stream << dynamic_type << "::apply(" << in << ',' << out
           << ") of dimensions " << solver->get_size() << " and "
           << in->get_size()[1] << " rhs\n";
    if (const auto base = dynamic_cast<const solver_base*>(solver)) {
        const auto scalars = base->get_workspace_scalars();
        const auto names = base->get_workspace_op_names();
        stream << std::setw(column_width_) << "Iteration";
        for (auto scalar : scalars) {
            stream << std::setw(column_width_) << names[scalar];
        }
        stream << '\n';
    } else {
        stream << "This solver type is not supported by the SolverDebug logger";
    }
}


void SolverDebug::on_iteration_complete(
    const LinOp* solver, const LinOp* right_hand_side, const LinOp* solution,
    const size_type& num_iterations, const LinOp* residual,
    const LinOp* residual_norm, const LinOp* implicit_sq_residual_norm,
    const array<stopping_status>* status, bool stopped) const
{
    using solver_base = solver::detail::SolverBaseLinOp;
    auto& stream = *output_;
    stream << std::setprecision(precision_);
    if (const auto base = dynamic_cast<const solver_base*>(solver)) {
        const auto scalars = base->get_workspace_scalars();
        stream << std::setw(column_width_) << num_iterations;
        for (auto scalar : scalars) {
            stream << std::setw(column_width_);
            print_scalar(base->get_workspace_op(scalar), stream);
        }
        stream << '\n';
    }
}


void SolverDebug::on_iteration_complete(const LinOp* solver,
                                        const size_type& num_iterations,
                                        const LinOp* residual,
                                        const LinOp* solution,
                                        const LinOp* residual_norm) const
{
    on_iteration_complete(solver, nullptr, solution, num_iterations, residual,
                          residual_norm, nullptr, nullptr, false);
}


void SolverDebug::on_iteration_complete(
    const LinOp* solver, const size_type& num_iterations, const LinOp* residual,
    const LinOp* solution, const LinOp* residual_norm,
    const LinOp* implicit_sq_residual_norm) const
{
    on_iteration_complete(solver, nullptr, solution, num_iterations, residual,
                          residual_norm, implicit_sq_residual_norm, nullptr,
                          false);
}


SolverDebug::SolverDebug(std::ostream& stream, int precision, int column_width)
    : output_{&stream}, precision_{precision}, column_width_{column_width}
{}


std::shared_ptr<SolverDebug> SolverDebug::create(std::ostream& output,
                                                 int precision,
                                                 int column_width)
{
    return std::shared_ptr<SolverDebug>{
        new SolverDebug{output, precision, column_width}};
}


}  // namespace log
}  // namespace gko
