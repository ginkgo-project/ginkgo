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
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>
#include <ginkgo/core/solver/batch_bicgstab.hpp>


#include "core/solver/batch_bicgstab_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "core/test/utils/batch_helpers.hpp"
#include "test/utils/executor.hpp"


namespace detail {


template <typename ValueType>
void is_equivalent_to_ref(
    std::unique_ptr<gko::batch::preconditioner::Jacobi<ValueType>> ref_prec,
    std::unique_ptr<gko::batch::preconditioner::Jacobi<ValueType>> d_prec)
{
    auto ref = ref_prec->get_executor();
    auto exec = d_prec->get_executor();
    const auto nbatch = ref_prec->get_num_batch_items();
    const auto num_blocks = ref_prec->get_num_blocks();
    const auto cumul_block_size =
        ref_prec->get_const_blocks_cumulative_offsets()[num_blocks];
    const auto block_pointers_ref = ref_prec->get_const_block_pointers();

    const auto tol = 10 * r<ValueType>::value;

    GKO_ASSERT_ARRAY_EQ(
        gko::array<int>::const_view(exec, num_blocks + 1,
                                    d_prec->get_const_block_pointers()),
        gko::array<int>::const_view(exec, num_blocks + 1, block_pointers_ref));
    GKO_ASSERT_ARRAY_NEAR(
        gko::array<ValueType>::const_view(exec, nbatch * cumul_block_size,
                                          d_prec->get_const_blocks()),
        gko::array<ValueType>::const_view(exec, nbatch * cumul_block_size,
                                          ref_prec->get_const_blocks()),
        tol);
}


}  // namespace detail


class BatchJacobi : public CommonTestFixture {
protected:
    using value_type = double;
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
        : ref_mtx(
              gko::share(gko::test::generate_diag_dominant_batch_matrix<Mtx>(
                  ref, nbatch, nrows, false, 4 * nrows - 3))),
          d_mtx(gko::share(Mtx::create(exec))),
          ref_b(gko::test::generate_random_batch_matrix<BMVec>(
              nbatch, nrows, 1, std::uniform_int_distribution<>(nrows, nrows),
              std::normal_distribution<real_type>(),
              std::default_random_engine(34), ref)),
          d_b(BMVec::create(exec,
                            gko::batch_dim<2>(nbatch, gko::dim<2>(nrows, 1)))),
          ref_x(BMVec::create(
              ref, gko::batch_dim<2>(nbatch, gko::dim<2>(nrows, 1)))),
          d_x(BMVec::create(exec,
                            gko::batch_dim<2>(nbatch, gko::dim<2>(nrows, 1))))
    {
        d_mtx->copy_from(ref_mtx.get());
        d_b->copy_from(ref_b.get());
        ref_scalar_jacobi_prec =
            BJ::build().with_max_block_size(1u).on(ref)->generate(ref_mtx);
        d_scalar_jacobi_prec =
            BJ::build().with_max_block_size(1u).on(exec)->generate(d_mtx);
        ref_block_jacobi_prec = BJ::build()
                                    .with_max_block_size(max_blk_sz)
                                    .on(ref)
                                    ->generate(ref_mtx);

        // TODO (before merging device kernels): Check if it is the same for
        // other device kernels
        // // so that the block pointers are exactly the same for ref and device
        // const int* block_pointers_generated_by_ref =
        //     ref_block_jacobi_prec->get_const_block_pointers();
        // const auto num_blocks_generated_by_ref =
        //     ref_block_jacobi_prec->get_num_blocks();

        // gko::array<int> block_pointers_for_device(
        //     this->exec, block_pointers_generated_by_ref,
        //     block_pointers_generated_by_ref + num_blocks_generated_by_ref +
        //     1);

        d_block_jacobi_prec =
            BJ::build()
                .with_max_block_size(max_blk_sz)
                // .with_block_pointers(block_pointers_for_device)
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
            gko::kernels::EXEC_NAMESPACE::batch_bicgstab::apply<
                typename Mtx::value_type>(executor, settings, mtx, prec, b, x,
                                          log_data);
        };
        solver_settings = Settings{max_iters, tol,
                                   gko::batch::stop::tolerance_type::relative};
        precond_solver_factory =
            solver_type::build()
                .with_max_iterations(max_iters)
                .with_tolerance(tol)
                .with_tolerance_type(gko::batch::stop::tolerance_type::relative)
                .with_preconditioner(
                    precond_type::build().with_max_block_size(2u))
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
    std::shared_ptr<typename solver_type::Factory> precond_solver_factory;
    std::shared_ptr<typename solver_type::Factory> scalar_jac_solver_factory;

    const size_t nbatch = 3;
    const int nrows = 300;
    std::shared_ptr<Mtx> ref_mtx;
    std::shared_ptr<Mtx> d_mtx;
    std::unique_ptr<BMVec> ref_b;
    std::unique_ptr<BMVec> d_b;
    std::unique_ptr<BMVec> ref_x;
    std::unique_ptr<BMVec> d_x;
    const gko::uint32 max_blk_sz = 6u;
    std::unique_ptr<BJ> ref_scalar_jacobi_prec;
    std::unique_ptr<BJ> d_scalar_jacobi_prec;
    std::unique_ptr<BJ> ref_block_jacobi_prec;
    std::unique_ptr<BJ> d_block_jacobi_prec;
};


TEST_F(BatchJacobi, BatchBlockJacobiGenerationIsEquivalentToRef)
{
    auto& ref_prec = ref_block_jacobi_prec;
    auto& d_prec = d_block_jacobi_prec;

    detail::is_equivalent_to_ref(std::move(ref_prec), std::move(d_prec));
}


TEST_F(BatchJacobi, CanSolveLargeMatrixSizeHpdSystemWithBlockJacobi)
{
    const int num_batch_items = 12;
    const int num_rows = 1025;
    const int num_rhs = 1;
    const real_type tol = 1e-5;
    const int max_iters = num_rows * 2;
    std::shared_ptr<Logger> logger = Logger::create();
    auto mat =
        gko::share(gko::test::generate_diag_dominant_batch_matrix<const CsrMtx>(
            exec, num_batch_items, num_rows, false, (4 * num_rows - 3)));
    auto linear_system = setup_linsys_and_solver(mat, num_rhs, tol, max_iters);
    auto solver =
        gko::share(precond_solver_factory->generate(linear_system.matrix));
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


TEST_F(BatchJacobi, CanSolveLargeMatrixSizeHpdSystemWithScalarJacobi)
{
    const int num_batch_items = 12;
    const int num_rows = 1025;
    const int num_rhs = 1;
    const real_type tol = 1e-5;
    const int max_iters = num_rows * 2;
    std::shared_ptr<Logger> logger = Logger::create();
    auto mat =
        gko::share(gko::test::generate_diag_dominant_batch_matrix<const CsrMtx>(
            exec, num_batch_items, num_rows, false, (4 * num_rows - 3)));
    auto linear_system = setup_linsys_and_solver(mat, num_rhs, tol, max_iters);
    auto solver =
        gko::share(scalar_jac_solver_factory->generate(linear_system.matrix));
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
