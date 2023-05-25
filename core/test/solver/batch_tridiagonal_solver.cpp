/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/solver/batch_tridiagonal_solver.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


template <typename T>
class BatchTridiagonalSolver : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<T>;
    using Mtx = gko::matrix::BatchTridiagonal<value_type>;
    using Dense = gko::matrix::BatchDense<value_type>;
    using Solver = gko::solver::BatchTridiagonalSolver<value_type>;

    BatchTridiagonalSolver()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::test::generate_uniform_batch_tridiagonal_random_matrix<
              value_type>(nbatch, nrows,
                          std::normal_distribution<real_type>(0.0, 1.0),
                          rand_engine, exec)),
          batchtridiagsolver_factory(Solver::build().on(exec)),
          solver(batchtridiagsolver_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::ranlux48 rand_engine;
    const gko::size_type nbatch = 2;
    const int nrows = 4;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> batchtridiagsolver_factory;
    const int num_WM_steps = 2;
    const int wm_pge_subwarp_size = 16;
    const enum gko::solver::batch_tridiag_solve_approach approach =
        gko::solver::batch_tridiag_solve_approach::wm_pge_app2;
    std::unique_ptr<gko::BatchLinOp> solver;
};

TYPED_TEST_SUITE(BatchTridiagonalSolver, gko::test::ValueTypes);


TYPED_TEST(BatchTridiagonalSolver, FactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->batchtridiagsolver_factory->get_executor(), this->exec);
}


TYPED_TEST(BatchTridiagonalSolver, FactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_EQ(this->solver->get_size().at(i),
                  gko::dim<2>(this->nrows, this->nrows));
    }
    auto batchtridiag_solver = static_cast<Solver*>(this->solver.get());
    ASSERT_NE(batchtridiag_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(batchtridiag_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(BatchTridiagonalSolver, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy =
        this->batchtridiagsolver_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver.get());

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_EQ(copy->get_size().at(i),
                  gko::dim<2>(this->nrows, this->nrows));
    }
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    const auto copy_batch_mtx = static_cast<const Mtx*>(copy_mtx.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx.get(), copy_batch_mtx, 0.0);
}


TYPED_TEST(BatchTridiagonalSolver, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy =
        this->batchtridiagsolver_factory->generate(Mtx::create(this->exec));

    copy->copy_from(std::move(this->solver));

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_EQ(copy->get_size().at(i),
                  gko::dim<2>(this->nrows, this->nrows));
    }
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    const auto copy_batch_mtx = static_cast<const Mtx*>(copy_mtx.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx.get(), copy_batch_mtx, 0.0);
}


TYPED_TEST(BatchTridiagonalSolver, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto clone = this->solver->clone();

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_EQ(clone->get_size().at(i),
                  gko::dim<2>(this->nrows, this->nrows));
    }
    auto clone_mtx = static_cast<Solver*>(clone.get())->get_system_matrix();
    const auto clone_batch_mtx = static_cast<const Mtx*>(clone_mtx.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx.get(), clone_batch_mtx, 0.0);
}


TYPED_TEST(BatchTridiagonalSolver, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;

    this->solver->clear();

    ASSERT_EQ(this->solver->get_num_batch_entries(), 0);
    ASSERT_EQ(this->solver->get_size().get_num_batch_entries(), 0);
    auto solver_mtx =
        static_cast<Solver*>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(BatchTridiagonalSolver, CanSetCriteriaInFactory)
{
    using Solver = typename TestFixture::Solver;
    using RT = typename TestFixture::real_type;

    auto batch_tridiag_solver_factory =
        Solver::build()
            .with_num_WM_steps(2)
            .with_wm_pge_subwarp_size(16)
            .with_batch_tridiagonal_solution_approach(
                gko::solver::batch_tridiag_solve_approach::wm_pge_app2)
            .on(this->exec);
    auto solver = batch_tridiag_solver_factory->generate(this->mtx);

    ASSERT_EQ(solver->get_parameters().num_WM_steps, 2);
    ASSERT_EQ(solver->get_parameters().wm_pge_subwarp_size, 16);
    ASSERT_EQ(solver->get_parameters().batch_tridiagonal_solution_approach,
              gko::solver::batch_tridiag_solve_approach::wm_pge_app2);
}


// TODO: Implement scaling for batched tridiagonal matrix format
// TYPED_TEST(BatchTridiagonalSolver, CanSetScalingOps)
// {
//     using value_type = typename TestFixture::value_type;
//     using Solver = typename TestFixture::Solver;
//     using Dense = typename TestFixture::Dense;
//     using Diag = gko::matrix::BatchDiagonal<value_type>;
//     auto left_scale = gko::share(Diag::create(
//         this->exec,
//         gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrows,
//         this->nrows))));
//     auto right_scale = gko::share(Diag::create(
//         this->exec,
//         gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrows,
//         this->nrows))));
//     auto batch_tridiag_solver_factory = Solver::build()
//                                      .with_left_scaling_op(left_scale)
//                                      .with_right_scaling_op(right_scale)
//                                      .on(this->exec);
//     auto solver = batch_tridiag_solver_factory->generate(this->mtx);

//     ASSERT_EQ(solver->get_left_scaling_op(), left_scale);
//     ASSERT_EQ(solver->get_right_scaling_op(), right_scale);
// }


}  // namespace
