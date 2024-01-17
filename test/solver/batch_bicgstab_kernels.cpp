// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/batch_bicgstab_kernels.hpp"


#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/log/batch_logger.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>
#include <ginkgo/core/solver/batch_bicgstab.hpp>


#include "core/base/batch_utilities.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_helpers.hpp"
#include "test/utils/executor.hpp"


class BatchBicgstab : public CommonTestFixture {
protected:
    using real_type = gko::remove_complex<value_type>;
    using solver_type = gko::batch::solver::Bicgstab<value_type>;
    using Mtx = gko::batch::matrix::Dense<value_type>;
    using EllMtx = gko::batch::matrix::Ell<value_type>;
    using MVec = gko::batch::MultiVector<value_type>;
    using RealMVec = gko::batch::MultiVector<real_type>;
    using Settings = gko::kernels::batch_bicgstab::settings<real_type>;
    using LogData = gko::batch::log::detail::log_data<real_type>;
    using Logger = gko::batch::log::BatchConvergence<real_type>;

    BatchBicgstab() {}

    template <typename MatrixType>
    gko::test::LinearSystem<MatrixType> setup_linsys_and_solver(
        std::shared_ptr<const MatrixType> mat, const int num_rhs,
        const real_type tol, const int max_iters)
    {
        auto executor = exec;
        solve_lambda = [executor](const Settings settings,
                                  const gko::batch::BatchLinOp* prec,
                                  const Mtx* mtx, const MVec* b, MVec* x,
                                  LogData& log_data) {
            gko::kernels::EXEC_NAMESPACE::batch_bicgstab::apply<
                typename Mtx::value_type>(executor, settings, mtx, prec, b, x,
                                          log_data);
        };
        solver_settings = Settings{max_iters, tol,
                                   gko::batch::stop::tolerance_type::relative};

        solver_factory =
            solver_type::build()
                .with_max_iterations(max_iters)
                .with_tolerance(tol)
                .with_tolerance_type(gko::batch::stop::tolerance_type::relative)
                .on(exec);
        return gko::test::generate_batch_linear_system(mat, num_rhs);
    }

    std::function<void(const Settings, const gko::batch::BatchLinOp*,
                       const Mtx*, const MVec*, MVec*, LogData&)>
        solve_lambda;
    Settings solver_settings{};
    std::shared_ptr<solver_type::Factory> solver_factory;
};


TEST_F(BatchBicgstab, SolvesStencilSystem)
{
    const int num_batch_items = 2;
    const int num_rows = 33;
    const int num_rhs = 1;
    const real_type tol = 1e-5;
    const int max_iters = 100;
    auto mat =
        gko::share(gko::test::generate_3pt_stencil_batch_matrix<const Mtx>(
            exec, num_batch_items, num_rows));
    auto linear_system = setup_linsys_and_solver(mat, num_rhs, tol, max_iters);

    auto res = gko::test::solve_linear_system(exec, solve_lambda,
                                              solver_settings, linear_system);

    for (size_t i = 0; i < num_batch_items; i++) {
        ASSERT_LE(res.host_res_norm->get_const_values()[i] /
                      linear_system.host_rhs_norm->get_const_values()[i],
                  solver_settings.residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(res.x, linear_system.exact_sol, tol);
}


TEST_F(BatchBicgstab, StencilSystemLoggerLogsResidual)
{
    const int num_batch_items = 2;
    const int num_rows = 33;
    const int num_rhs = 1;
    const real_type tol = 1e-5;
    const int max_iters = 100;
    auto mat =
        gko::share(gko::test::generate_3pt_stencil_batch_matrix<const Mtx>(
            exec, num_batch_items, num_rows));
    auto linear_system = setup_linsys_and_solver(mat, num_rhs, tol, max_iters);

    auto res = gko::test::solve_linear_system(exec, solve_lambda,
                                              solver_settings, linear_system);

    auto res_log_array = res.log_data->res_norms.get_const_data();
    for (size_t i = 0; i < num_batch_items; i++) {
        ASSERT_LE(res_log_array[i] / linear_system.host_rhs_norm->at(i, 0, 0),
                  solver_settings.residual_tol);
        ASSERT_NEAR(res_log_array[i], res.host_res_norm->get_const_values()[i],
                    10 * tol);
    }
}


TEST_F(BatchBicgstab, StencilSystemLoggerLogsIterations)
{
    const int num_batch_items = 2;
    const int num_rows = 33;
    const int num_rhs = 1;
    const int ref_iters = 5;
    auto mat =
        gko::share(gko::test::generate_3pt_stencil_batch_matrix<const Mtx>(
            exec, num_batch_items, num_rows));
    auto linear_system = setup_linsys_and_solver(mat, num_rhs, 0, ref_iters);

    auto res = gko::test::solve_linear_system(exec, solve_lambda,
                                              solver_settings, linear_system);

    auto iter_array = res.log_data->iter_counts.get_const_data();
    for (size_t i = 0; i < num_batch_items; i++) {
        ASSERT_EQ(iter_array[i], ref_iters);
    }
}


TEST_F(BatchBicgstab, CanSolve3ptStencilSystem)
{
    const int num_batch_items = 8;
    const int num_rows = 100;
    const int num_rhs = 1;
    const real_type tol = 1e-5;
    const int max_iters = 500;
    auto mat =
        gko::share(gko::test::generate_3pt_stencil_batch_matrix<const Mtx>(
            exec, num_batch_items, num_rows));
    auto linear_system = setup_linsys_and_solver(mat, num_rhs, tol, max_iters);
    auto solver = gko::share(solver_factory->generate(linear_system.matrix));

    auto res = gko::test::solve_linear_system(exec, linear_system, solver);

    GKO_ASSERT_BATCH_MTX_NEAR(res.x, linear_system.exact_sol, tol * 10);
    for (size_t i = 0; i < num_batch_items; i++) {
        auto comp_res_norm = res.host_res_norm->get_const_values()[i] /
                             linear_system.host_rhs_norm->get_const_values()[i];
        ASSERT_LE(comp_res_norm, tol);
    }
}


TEST_F(BatchBicgstab, CanSolveLargeBatchSizeHpdSystem)
{
    const int num_batch_items = 100;
    const int num_rows = 102;
    const int num_rhs = 1;
    const real_type tol = 1e-5;
    const int max_iters = num_rows * 2;
    std::shared_ptr<Logger> logger = Logger::create();
    auto mat =
        gko::share(gko::test::generate_diag_dominant_batch_matrix<const Mtx>(
            exec, num_batch_items, num_rows, true));
    auto linear_system = setup_linsys_and_solver(mat, num_rhs, tol, max_iters);
    auto solver = gko::share(solver_factory->generate(linear_system.matrix));
    solver->add_logger(logger);

    auto res = gko::test::solve_linear_system(exec, linear_system, solver);

    solver->remove_logger(logger);
    auto iter_counts = gko::make_temporary_clone(exec->get_master(),
                                                 &logger->get_num_iterations());
    auto res_norm = gko::make_temporary_clone(exec->get_master(),
                                              &logger->get_residual_norm());
    GKO_ASSERT_BATCH_MTX_NEAR(res.x, linear_system.exact_sol, tol * 500);
    for (size_t i = 0; i < num_batch_items; i++) {
        auto comp_res_norm = res.host_res_norm->get_const_values()[i] /
                             linear_system.host_rhs_norm->get_const_values()[i];
        ASSERT_LE(iter_counts->get_const_data()[i], max_iters);
        EXPECT_LE(res_norm->get_const_data()[i] /
                      linear_system.host_rhs_norm->get_const_values()[i],
                  tol);
        EXPECT_GT(res_norm->get_const_data()[i], real_type{0.0});
        ASSERT_LE(comp_res_norm, tol * 10);
    }
}


TEST_F(BatchBicgstab, CanSolveLargeMatrixSizeHpdSystem)
{
    const int num_batch_items = 12;
    const int num_rows = 1025;
    const int num_rhs = 1;
    const real_type tol = 1e-5;
    const int max_iters = num_rows * 2;
    std::shared_ptr<Logger> logger = Logger::create();
    auto mat =
        gko::share(gko::test::generate_diag_dominant_batch_matrix<const Mtx>(
            exec, num_batch_items, num_rows, true));
    auto linear_system = setup_linsys_and_solver(mat, num_rhs, tol, max_iters);
    auto solver = gko::share(solver_factory->generate(linear_system.matrix));
    solver->add_logger(logger);

    auto res = gko::test::solve_linear_system(exec, linear_system, solver);

    solver->remove_logger(logger);
    auto iter_counts = gko::make_temporary_clone(exec->get_master(),
                                                 &logger->get_num_iterations());
    auto res_norm = gko::make_temporary_clone(exec->get_master(),
                                              &logger->get_residual_norm());
    GKO_ASSERT_BATCH_MTX_NEAR(res.x, linear_system.exact_sol, tol * 500);
    for (size_t i = 0; i < num_batch_items; i++) {
        auto comp_res_norm = res.host_res_norm->get_const_values()[i] /
                             linear_system.host_rhs_norm->get_const_values()[i];
        ASSERT_LE(iter_counts->get_const_data()[i], max_iters);
        EXPECT_LE(res_norm->get_const_data()[i] /
                      linear_system.host_rhs_norm->get_const_values()[i],
                  tol);
        EXPECT_GT(res_norm->get_const_data()[i], real_type{0.0});
        ASSERT_LE(comp_res_norm, tol * 10);
    }
}
