// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/batch_bicgstab.hpp>


#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/log/batch_logger.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>


#include "core/base/batch_utilities.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/solver/batch_bicgstab_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_helpers.hpp"


template <typename T>
class BatchBicgstab : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using solver_type = gko::batch::solver::Bicgstab<value_type>;
    using Mtx = gko::batch::matrix::Dense<value_type>;
    using EllMtx = gko::batch::matrix::Ell<value_type>;
    using CsrMtx = gko::batch::matrix::Csr<value_type>;
    using MVec = gko::batch::MultiVector<value_type>;
    using RealMVec = gko::batch::MultiVector<real_type>;
    using Settings = gko::kernels::batch_bicgstab::settings<real_type>;
    using LogData = gko::batch::log::detail::log_data<real_type>;
    using LinSys = gko::test::LinearSystem<Mtx>;

    BatchBicgstab()
        : exec(gko::ReferenceExecutor::create()),
          mat(gko::share(
              gko::test::generate_3pt_stencil_batch_matrix<const Mtx>(
                  exec, num_batch_items, num_rows))),
          linear_system(gko::test::generate_batch_linear_system(mat, num_rhs))
    {
        auto executor = this->exec;
        solve_lambda = [executor](const Settings opts,
                                  const gko::batch::BatchLinOp* prec,
                                  const Mtx* mtx, const MVec* b, MVec* x,
                                  LogData& log_data) {
            gko::kernels::reference::batch_bicgstab::apply<
                typename Mtx::value_type>(executor, opts, mtx, prec, b, x,
                                          log_data);
        };
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    const real_type eps = 1e-3;
    const gko::size_type num_batch_items = 2;
    const int num_rows = 15;
    const int num_rhs = 1;
    const Settings solver_settings{100, eps,
                                   gko::batch::stop::tolerance_type::relative};
    std::shared_ptr<const Mtx> mat;
    LinSys linear_system;
    std::function<void(const Settings, const gko::batch::BatchLinOp*,
                       const Mtx*, const MVec*, MVec*, LogData&)>
        solve_lambda;
};

TYPED_TEST_SUITE(BatchBicgstab, gko::test::RealValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(BatchBicgstab, SolvesStencilSystem)
{
    auto res = gko::test::solve_linear_system(this->exec, this->solve_lambda,
                                              this->solver_settings,
                                              this->linear_system);

    for (size_t i = 0; i < this->num_batch_items; i++) {
        ASSERT_LE(res.host_res_norm->get_const_values()[i] /
                      this->linear_system.host_rhs_norm->get_const_values()[i],
                  this->solver_settings.residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(res.x, this->linear_system.exact_sol,
                              this->eps * 10);
}


TYPED_TEST(BatchBicgstab, StencilSystemLoggerLogsResidual)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    auto res = gko::test::solve_linear_system(this->exec, this->solve_lambda,
                                              this->solver_settings,
                                              this->linear_system);

    const int ref_iters = 2;
    auto iter_array = res.log_data->iter_counts.get_const_data();
    auto res_log_array = res.log_data->res_norms.get_const_data();
    for (size_t i = 0; i < this->num_batch_items; i++) {
        ASSERT_LE(
            res_log_array[i] / this->linear_system.host_rhs_norm->at(i, 0, 0),
            this->solver_settings.residual_tol);
        ASSERT_NEAR(res_log_array[i], res.host_res_norm->get_const_values()[i],
                    10 * this->eps);
    }
}


TYPED_TEST(BatchBicgstab, StencilSystemLoggerLogsIterations)
{
    using value_type = typename TestFixture::value_type;
    using Settings = typename TestFixture::Settings;
    using real_type = gko::remove_complex<value_type>;
    const int ref_iters = 5;
    const Settings solver_settings{ref_iters, 0,
                                   gko::batch::stop::tolerance_type::relative};

    auto res = gko::test::solve_linear_system(
        this->exec, this->solve_lambda, solver_settings, this->linear_system);

    auto iter_array = res.log_data->iter_counts.get_const_data();
    for (size_t i = 0; i < this->num_batch_items; i++) {
        ASSERT_EQ(iter_array[i], ref_iters);
    }
}


TYPED_TEST(BatchBicgstab, CanSolveDenseSystem)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;
    using Solver = typename TestFixture::solver_type;
    using Mtx = typename TestFixture::Mtx;
    const real_type tol = 1e-5;
    const int max_iters = 1000;
    auto solver_factory =
        Solver::build()
            .with_max_iterations(max_iters)
            .with_tolerance(tol)
            .with_tolerance_type(gko::batch::stop::tolerance_type::relative)
            .on(this->exec);
    const int num_rows = 13;
    const size_t num_batch_items = 5;
    const int num_rhs = 1;
    auto stencil_mat =
        gko::share(gko::test::generate_3pt_stencil_batch_matrix<const Mtx>(
            this->exec, num_batch_items, num_rows));
    auto linear_system =
        gko::test::generate_batch_linear_system(stencil_mat, num_rhs);
    auto solver = gko::share(solver_factory->generate(linear_system.matrix));

    auto res =
        gko::test::solve_linear_system(this->exec, linear_system, solver);

    GKO_ASSERT_BATCH_MTX_NEAR(res.x, linear_system.exact_sol, tol * 10);
    for (size_t i = 0; i < num_batch_items; i++) {
        ASSERT_LE(res.host_res_norm->get_const_values()[i] /
                      linear_system.host_rhs_norm->get_const_values()[i],
                  tol);
    }
}


TYPED_TEST(BatchBicgstab, ApplyLogsResAndIters)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;
    using Solver = typename TestFixture::solver_type;
    using Mtx = typename TestFixture::Mtx;
    using Logger = gko::batch::log::BatchConvergence<value_type>;
    const real_type tol = 1e-5;
    const int max_iters = 1000;
    auto solver_factory =
        Solver::build()
            .with_max_iterations(max_iters)
            .with_tolerance(tol)
            .with_tolerance_type(gko::batch::stop::tolerance_type::relative)
            .on(this->exec);
    const int num_rows = 13;
    const size_t num_batch_items = 5;
    const int num_rhs = 1;
    std::shared_ptr<Logger> logger = Logger::create();
    auto stencil_mat =
        gko::share(gko::test::generate_3pt_stencil_batch_matrix<const Mtx>(
            this->exec, num_batch_items, num_rows));
    auto linear_system =
        gko::test::generate_batch_linear_system(stencil_mat, num_rhs);
    auto solver = gko::share(solver_factory->generate(linear_system.matrix));

    solver->add_logger(logger);
    auto res =
        gko::test::solve_linear_system(this->exec, linear_system, solver);
    solver->remove_logger(logger);

    auto iter_counts = logger->get_num_iterations();
    auto res_norm = logger->get_residual_norm();
    GKO_ASSERT_BATCH_MTX_NEAR(res.x, linear_system.exact_sol, tol * 50);
    for (size_t i = 0; i < num_batch_items; i++) {
        auto rel_res_norm = res.host_res_norm->get_const_values()[i] /
                            linear_system.host_rhs_norm->get_const_values()[i];
        ASSERT_LE(iter_counts.get_const_data()[i], max_iters);
        EXPECT_LE(res_norm.get_const_data()[i], tol * 50);
        ASSERT_LE(rel_res_norm, tol * 50);
    }
}


TYPED_TEST(BatchBicgstab, CanSolveEllSystem)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;
    using Solver = typename TestFixture::solver_type;
    using Mtx = typename TestFixture::EllMtx;
    const real_type tol = 1e-5;
    const int max_iters = 1000;
    auto solver_factory =
        Solver::build()
            .with_max_iterations(max_iters)
            .with_tolerance(tol)
            .with_tolerance_type(gko::batch::stop::tolerance_type::relative)
            .on(this->exec);
    const int num_rows = 13;
    const size_t num_batch_items = 2;
    const int num_rhs = 1;
    auto stencil_mat =
        gko::share(gko::test::generate_3pt_stencil_batch_matrix<const Mtx>(
            this->exec, num_batch_items, num_rows, 3));
    auto linear_system =
        gko::test::generate_batch_linear_system(stencil_mat, num_rhs);
    auto solver = gko::share(solver_factory->generate(linear_system.matrix));

    auto res =
        gko::test::solve_linear_system(this->exec, linear_system, solver);

    GKO_ASSERT_BATCH_MTX_NEAR(res.x, linear_system.exact_sol, tol * 10);
    for (size_t i = 0; i < num_batch_items; i++) {
        ASSERT_LE(res.host_res_norm->get_const_values()[i] /
                      linear_system.host_rhs_norm->get_const_values()[i],
                  tol * 10);
    }
}


TYPED_TEST(BatchBicgstab, CanSolveCsrSystem)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;
    using Solver = typename TestFixture::solver_type;
    using Mtx = typename TestFixture::CsrMtx;
    const real_type tol = 1e-5;
    const int max_iters = 1000;
    auto solver_factory =
        Solver::build()
            .with_max_iterations(max_iters)
            .with_tolerance(tol)
            .with_tolerance_type(gko::batch::stop::tolerance_type::relative)
            .on(this->exec);
    const int num_rows = 13;
    const size_t num_batch_items = 2;
    const int num_rhs = 1;
    auto stencil_mat =
        gko::share(gko::test::generate_3pt_stencil_batch_matrix<const Mtx>(
            this->exec, num_batch_items, num_rows, (num_rows * 3 - 2)));
    auto linear_system =
        gko::test::generate_batch_linear_system(stencil_mat, num_rhs);
    auto solver = gko::share(solver_factory->generate(linear_system.matrix));

    auto res =
        gko::test::solve_linear_system(this->exec, linear_system, solver);

    GKO_ASSERT_BATCH_MTX_NEAR(res.x, linear_system.exact_sol, tol * 10);
    for (size_t i = 0; i < num_batch_items; i++) {
        ASSERT_LE(res.host_res_norm->get_const_values()[i] /
                      linear_system.host_rhs_norm->get_const_values()[i],
                  tol * 10);
    }
}


TYPED_TEST(BatchBicgstab, CanSolveDenseHpdSystem)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;
    using Solver = typename TestFixture::solver_type;
    using Mtx = typename TestFixture::Mtx;
    const real_type tol = 1e-5;
    const int max_iters = 1000;
    auto solver_factory =
        Solver::build()
            .with_max_iterations(max_iters)
            .with_tolerance(tol)
            .with_tolerance_type(gko::batch::stop::tolerance_type::absolute)
            .on(this->exec);
    const int num_rows = 65;
    const gko::size_type num_batch_items = 5;
    const int num_rhs = 1;
    auto diag_dom_mat =
        gko::share(gko::test::generate_diag_dominant_batch_matrix<const Mtx>(
            this->exec, num_batch_items, num_rows, true));
    auto linear_system =
        gko::test::generate_batch_linear_system(diag_dom_mat, num_rhs);
    auto solver = gko::share(solver_factory->generate(linear_system.matrix));

    auto res =
        gko::test::solve_linear_system(this->exec, linear_system, solver);

    GKO_ASSERT_BATCH_MTX_NEAR(res.x, linear_system.exact_sol, tol * 50);
    for (size_t i = 0; i < num_batch_items; i++) {
        ASSERT_LE(res.host_res_norm->get_const_values()[i], tol * 50);
    }
}
