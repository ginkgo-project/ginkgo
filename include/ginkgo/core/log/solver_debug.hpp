// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_LOG_SOLVER_DEBUG_HPP_
#define GKO_PUBLIC_CORE_LOG_SOLVER_DEBUG_HPP_


#include <iosfwd>


#include <ginkgo/config.hpp>
#include <ginkgo/core/log/logger.hpp>


namespace gko {
namespace log {


/**
 * This Logger prints the value of all scalar values stored internally by the
 * solver after each iteration. If the solver is applied to multiple right-hand
 * sides, only the first right-hand side gets printed.
 */
class SolverDebug : public Logger {
public:
    /* Internal solver events */
    void on_linop_apply_started(const LinOp* A, const LinOp* b,
                                const LinOp* x) const override;

    void on_iteration_complete(
        const LinOp* solver, const LinOp* right_hand_side,
        const LinOp* solution, const size_type& num_iterations,
        const LinOp* residual, const LinOp* residual_norm,
        const LinOp* implicit_sq_residual_norm,
        const array<stopping_status>* status, bool stopped) const override;

    GKO_DEPRECATED(
        "Please use the version with the additional stopping "
        "information.")
    void on_iteration_complete(const LinOp* solver,
                               const size_type& num_iterations,
                               const LinOp* residual, const LinOp* solution,
                               const LinOp* residual_norm) const override;

    GKO_DEPRECATED(
        "Please use the version with the additional stopping "
        "information.")
    void on_iteration_complete(
        const LinOp* solver, const size_type& num_iterations,
        const LinOp* residual, const LinOp* solution,
        const LinOp* residual_norm,
        const LinOp* implicit_sq_residual_norm) const override;

    /**
     * Creates a logger printing the value for all scalar values in the solver
     * after each iteration.
     *
     * @param output  the stream to write the output to.
     * @param precision  the number of digits of precision to print
     * @param column_width  the number of characters an output column is wide
     */
    static std::shared_ptr<SolverDebug> create(std::ostream& output,
                                               int precision = 6,
                                               int column_width = 12);

private:
    SolverDebug(std::ostream& output, int precision, int column_width);

    std::ostream* output_;
    int precision_;
    int column_width_;
};


}  // namespace log
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_LOG_SOLVER_DEBUG_HPP_
