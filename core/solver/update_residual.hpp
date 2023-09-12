// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_UPDATE_RESIDUAL_HPP_
#define GKO_CORE_SOLVER_UPDATE_RESIDUAL_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace solver {


template <typename SolverType, typename VectorType, typename LogFunc>
bool update_residual(SolverType* solver, int iter, const VectorType* dense_b,
                     VectorType* dense_x, VectorType* residual,
                     const VectorType*& residual_ptr,
                     std::unique_ptr<gko::stop::Criterion>& stop_criterion,
                     array<stopping_status>& stop_status, LogFunc log)
{
    using ws = workspace_traits<std::remove_cv_t<SolverType>>;
    constexpr uint8 relative_stopping_id{1};

    // It's required to be initialized outside.
    auto one_op = solver->get_workspace_op(ws::one);
    auto neg_one_op = solver->get_workspace_op(ws::minus_one);

    bool one_changed{};
    if (iter == 0) {
        // In iter 0, the iteration and residual are updated.
        bool all_stopped =
            stop_criterion->update()
                .num_iterations(iter)
                .residual(residual_ptr)
                .solution(dense_x)
                .check(relative_stopping_id, true, &stop_status, &one_changed);
        log(solver, dense_b, dense_x, iter, residual_ptr, stop_status,
            all_stopped);
        return all_stopped;
    } else {
        // In the other iterations, the residual can be updated separately.
        bool all_stopped =
            stop_criterion->update()
                .num_iterations(iter)
                .solution(dense_x)
                // we have the residual check later
                .ignore_residual_check(true)
                .check(relative_stopping_id, false, &stop_status, &one_changed);
        if (all_stopped) {
            log(solver, dense_b, dense_x, iter, nullptr, stop_status,
                all_stopped);
            return all_stopped;
        }
        residual_ptr = residual;
        // residual = b - A * x
        residual->copy_from(dense_b);
        solver->get_system_matrix()->apply(neg_one_op, dense_x, one_op,
                                           residual);
        all_stopped =
            stop_criterion->update()
                .num_iterations(iter)
                .residual(residual_ptr)
                .solution(dense_x)
                .check(relative_stopping_id, true, &stop_status, &one_changed);
        log(solver, dense_b, dense_x, iter, residual_ptr, stop_status,
            all_stopped);
        return all_stopped;
    }
}


}  // namespace solver
}  // namespace gko

#endif  // GKO_CORE_SOLVER_UPDATE_RESIDUAL_HPP_
