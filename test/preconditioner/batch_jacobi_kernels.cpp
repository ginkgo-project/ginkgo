// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/batch_jacobi_kernels.hpp"

#include <limits>
#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_identity.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/batch_bicgstab.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/solver/batch_bicgstab_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "core/test/utils/batch_helpers.hpp"
#include "test/utils/common_fixture.hpp"


namespace detail {


template <typename ValueType>
void is_equivalent_to_ref(
    std::unique_ptr<gko::batch::preconditioner::Jacobi<ValueType>> ref_prec,
    std::unique_ptr<gko::batch::preconditioner::Jacobi<ValueType>> d_prec)
{
    auto ref = ref_prec->get_executor();
    auto exec = d_prec->get_executor();
    const auto nbatch = ref_prec->get_num_batch_items();
    const auto num_rows = ref_prec->get_common_size()[0];
    const auto num_blocks = ref_prec->get_num_blocks();
    const auto cumul_block_size =
        ref_prec->get_const_blocks_cumulative_offsets()[num_blocks];

    const auto tol = 10 * r<ValueType>::value;

    GKO_EXPECT_ARRAY_EQ(gko::array<int>::const_view(
                            exec, num_blocks + 1,
                            d_prec->get_const_blocks_cumulative_offsets()),
                        gko::array<int>::const_view(
                            ref, num_blocks + 1,
                            ref_prec->get_const_blocks_cumulative_offsets()));
    GKO_EXPECT_ARRAY_EQ(
        gko::array<int>::const_view(exec, num_blocks + 1,
                                    d_prec->get_const_block_pointers()),
        gko::array<int>::const_view(ref, num_blocks + 1,
                                    ref_prec->get_const_block_pointers()));
    GKO_EXPECT_ARRAY_EQ(
        gko::array<int>::const_view(exec, num_rows,
                                    d_prec->get_const_map_block_to_row()),
        gko::array<int>::const_view(ref, num_rows,
                                    ref_prec->get_const_map_block_to_row()));
    GKO_EXPECT_ARRAY_NEAR(
        gko::array<ValueType>::const_view(exec, nbatch * cumul_block_size,
                                          d_prec->get_const_blocks()),
        gko::array<ValueType>::const_view(ref, nbatch * cumul_block_size,
                                          ref_prec->get_const_blocks()),
        tol);
}


}  // namespace detail


class BatchJacobi : public CommonTestFixture {
protected:
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::batch::matrix::Csr<value_type, int>;
    using BMVec = gko::batch::MultiVector<value_type>;
    using BJ = gko::batch::preconditioner::Jacobi<value_type>;
    using solver_type = gko::batch::solver::Bicgstab<value_type>;
    using precond_type = gko::batch::preconditioner::Jacobi<value_type>;
    using CsrMtx = gko::batch::matrix::Csr<value_type>;
    using MVec = gko::batch::MultiVector<value_type>;
    using RealMVec = gko::batch::MultiVector<real_type>;
    using Settings = gko::kernels::batch_bicgstab::settings<real_type>;
    using LogData = gko::batch::log::detail::log_data<real_type>;
    using Logger = gko::batch::log::BatchConvergence<real_type>;

    BatchJacobi()
        : ref_mtx(gko::share(gko::test::generate_3pt_stencil_batch_matrix<Mtx>(
              ref, nbatch, nrows, 3 * nrows - 2))),
          d_mtx(gko::share(Mtx::create(exec)))
    {
        d_mtx->copy_from(ref_mtx.get());
        ref_block_jacobi_prec = BJ::build()
                                    .with_max_block_size(max_blk_sz)
                                    .on(ref)
                                    ->generate(ref_mtx);

        d_block_jacobi_prec = BJ::build()
                                  .with_max_block_size(max_blk_sz)
                                  .on(exec)
                                  ->generate(d_mtx);
    }

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
            if (prec == nullptr) {
                auto identity =
                    gko::batch::matrix::Identity<value_type>::create(
                        executor, mtx->get_size());
                gko::kernels::GKO_DEVICE_NAMESPACE::batch_bicgstab::apply(
                    executor, settings, mtx, identity.get(), b, x, log_data);
            } else {
                gko::run<gko::batch::matrix::Identity<value_type>,
                         gko::batch::preconditioner::Jacobi<value_type>>(
                    prec, [&](auto preconditioner) {
                        gko::kernels::GKO_DEVICE_NAMESPACE::batch_bicgstab::
                            apply(executor, settings, mtx, preconditioner, b, x,
                                  log_data);
                    });
            }
        };
        solver_settings = Settings{max_iters, tol,
                                   gko::batch::stop::tolerance_type::relative};
        solver_factory =
            solver_type::build()
                .with_max_iterations(max_iters)
                .with_tolerance(tol)
                .with_tolerance_type(gko::batch::stop::tolerance_type::relative)
                .on(exec);
        precond_solver_factory =
            solver_type::build()
                .with_max_iterations(max_iters)
                .with_tolerance(tol)
                .with_tolerance_type(gko::batch::stop::tolerance_type::relative)
                .with_preconditioner(
                    precond_type::build().with_max_block_size(4u))
                .on(exec);
        scalar_jac_solver_factory =
            solver_type::build()
                .with_max_iterations(max_iters)
                .with_tolerance(tol)
                .with_tolerance_type(gko::batch::stop::tolerance_type::relative)
                .with_preconditioner(
                    precond_type::build().with_max_block_size(1u))
                .on(exec);
        scalar_jac_solver_factory =
            solver_type::build()
                .with_max_iterations(max_iters)
                .with_tolerance(tol)
                .with_tolerance_type(gko::batch::stop::tolerance_type::relative)
                .with_preconditioner(
                    precond_type::build().with_max_block_size(1u))
                .on(exec);
        return gko::test::generate_batch_linear_system(mat, num_rhs);
    }

    std::function<void(const Settings, const gko::batch::BatchLinOp*,
                       const Mtx*, const MVec*, MVec*, LogData&)>
        solve_lambda;
    Settings solver_settings{};
    std::shared_ptr<typename solver_type::Factory> solver_factory;
    std::shared_ptr<typename solver_type::Factory> precond_solver_factory;
    std::shared_ptr<typename solver_type::Factory> scalar_jac_solver_factory;

    const size_t nbatch = 3;
    const int nrows = 300;
    std::shared_ptr<Mtx> ref_mtx;
    std::shared_ptr<Mtx> d_mtx;
    const gko::uint32 max_blk_sz = 6u;
    std::unique_ptr<BJ> ref_block_jacobi_prec;
    std::unique_ptr<BJ> d_block_jacobi_prec;
};


TEST_F(BatchJacobi, BatchBlockJacobiGenerationIsEquivalentToRef)
{
    auto& ref_prec = ref_block_jacobi_prec;
    auto& d_prec = d_block_jacobi_prec;

    detail::is_equivalent_to_ref(std::move(ref_prec), std::move(d_prec));
}


TEST_F(BatchJacobi, CanSolveLargeMatrixSizeHpdSystemWithScalarJacobi)
{
    const int num_batch_items = 12;
    const int num_rows = 1025;
    const int num_rhs = 1;
    const real_type tol = 1e-5;
    const int max_iters = num_rows;
    std::shared_ptr<Logger> logger = Logger::create();
    auto mat =
        gko::share(gko::test::generate_3pt_stencil_batch_matrix<const CsrMtx>(
            exec, num_batch_items, num_rows, (3 * num_rows - 2)));
    auto linear_system = setup_linsys_and_solver(mat, num_rhs, tol, max_iters);
    auto precond_solver =
        gko::share(scalar_jac_solver_factory->generate(linear_system.matrix));
    precond_solver->add_logger(logger);

    auto res =
        gko::test::solve_linear_system(exec, linear_system, precond_solver);

    precond_solver->remove_logger(logger);
    auto iter_counts = gko::make_temporary_clone(exec->get_master(),
                                                 &logger->get_num_iterations());
    auto res_norm = gko::make_temporary_clone(exec->get_master(),
                                              &logger->get_residual_norm());
    GKO_ASSERT_BATCH_MTX_NEAR(res.x, linear_system.exact_sol, tol * 500);
    for (size_t i = 0; i < num_batch_items; i++) {
        auto comp_res_norm = res.host_res_norm->get_const_values()[i] /
                             linear_system.host_rhs_norm->get_const_values()[i];
        EXPECT_LE(res_norm->get_const_data()[i] /
                      linear_system.host_rhs_norm->get_const_values()[i],
                  tol);
        EXPECT_LT(iter_counts->get_const_data()[i], max_iters);
        EXPECT_GT(res_norm->get_const_data()[i], real_type{0.0});
        ASSERT_LE(comp_res_norm, tol * 10);
    }
}


TEST_F(BatchJacobi, CanSolveLargeMatrixSizeHpdSystemWithBlockJacobi)
{
    const int num_batch_items = 12;
    const int num_rows = 513;
    const int num_rhs = 1;
    const real_type tol = 1e-5;
    const int max_iters = num_rows;
    std::shared_ptr<Logger> logger = Logger::create();
    auto mat =
        gko::share(gko::test::generate_diag_dominant_batch_matrix<const CsrMtx>(
            exec, num_batch_items, num_rows, false, (4 * num_rows - 3)));
    auto linear_system = setup_linsys_and_solver(mat, num_rhs, tol, max_iters);
    auto precond_solver =
        gko::share(precond_solver_factory->generate(linear_system.matrix));
    precond_solver->add_logger(logger);

    auto res =
        gko::test::solve_linear_system(exec, linear_system, precond_solver);

    precond_solver->remove_logger(logger);
    auto iter_counts = gko::make_temporary_clone(exec->get_master(),
                                                 &logger->get_num_iterations());
    auto res_norm = gko::make_temporary_clone(exec->get_master(),
                                              &logger->get_residual_norm());
    GKO_ASSERT_BATCH_MTX_NEAR(res.x, linear_system.exact_sol, tol * 500);
    for (size_t i = 0; i < num_batch_items; i++) {
        auto comp_res_norm = res.host_res_norm->get_const_values()[i] /
                             linear_system.host_rhs_norm->get_const_values()[i];
        EXPECT_LT(iter_counts->get_const_data()[i], max_iters);
        EXPECT_LE(res_norm->get_const_data()[i] /
                      linear_system.host_rhs_norm->get_const_values()[i],
                  tol);
        EXPECT_GT(res_norm->get_const_data()[i], real_type{0.0});
        ASSERT_LE(comp_res_norm, tol * 10);
    }
}
