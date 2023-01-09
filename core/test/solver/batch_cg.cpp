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

#include <ginkgo/core/solver/batch_cg.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


template <typename T>
class BatchCg : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<T>;
    using Mtx = gko::matrix::BatchCsr<value_type>;
    using Dense = gko::matrix::BatchDense<value_type>;
    using Solver = gko::solver::BatchCg<value_type>;

    BatchCg()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::test::create_poisson1d_batch<Mtx>(
              std::static_pointer_cast<const gko::ReferenceExecutor>(
                  this->exec),
              nrows, nbatch)),
          batchcg_factory(Solver::build()
                              .with_default_max_iterations(def_max_iters)
                              .with_default_residual_tol(def_abs_res_tol)
                              .with_tolerance_type(def_tol_type)
                              .on(exec)),
          solver(batchcg_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    const gko::size_type nbatch = 3;
    const int nrows = 5;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> batchcg_factory;
    const int def_max_iters = 100;
    const real_type def_abs_res_tol = 1e-11;
    const gko::stop::batch::ToleranceType def_tol_type =
        gko::stop::batch::ToleranceType::absolute;
    std::unique_ptr<gko::BatchLinOp> solver;
};

TYPED_TEST_SUITE(BatchCg, gko::test::ValueTypes);


TYPED_TEST(BatchCg, FactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->batchcg_factory->get_executor(), this->exec);
}


TYPED_TEST(BatchCg, FactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_EQ(this->solver->get_size().at(i),
                  gko::dim<2>(this->nrows, this->nrows));
    }
    auto batchcg_solver = static_cast<Solver*>(this->solver.get());
    ASSERT_NE(batchcg_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(batchcg_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(BatchCg, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->batchcg_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver.get());

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_EQ(copy->get_size().at(i),
                  gko::dim<2>(this->nrows, this->nrows));
    }
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    const auto copy_batch_mtx = static_cast<const Mtx*>(copy_mtx.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx.get(), copy_batch_mtx, 0.0);
}


TYPED_TEST(BatchCg, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->batchcg_factory->generate(Mtx::create(this->exec));

    copy->copy_from(std::move(this->solver));

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_EQ(copy->get_size().at(i),
                  gko::dim<2>(this->nrows, this->nrows));
    }
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    const auto copy_batch_mtx = static_cast<const Mtx*>(copy_mtx.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx.get(), copy_batch_mtx, 0.0);
}


TYPED_TEST(BatchCg, CanBeCloned)
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


TYPED_TEST(BatchCg, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;

    this->solver->clear();

    ASSERT_EQ(this->solver->get_num_batch_entries(), 0);
    ASSERT_EQ(this->solver->get_size().get_num_batch_entries(), 0);
    auto solver_mtx =
        static_cast<Solver*>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(BatchCg, ApplyUsesInitialGuessReturnsTrue)
{
    ASSERT_TRUE(this->solver->apply_uses_initial_guess());
}


TYPED_TEST(BatchCg, CanSetCriteriaInFactory)
{
    using Solver = typename TestFixture::Solver;
    using RT = typename TestFixture::real_type;

    auto batchcg_factory =
        Solver::build()
            .with_default_max_iterations(22)
            .with_default_residual_tol(static_cast<RT>(0.25))
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .on(this->exec);
    auto solver = batchcg_factory->generate(this->mtx);

    ASSERT_EQ(solver->get_parameters().default_max_iterations, 22);
    const RT tol = std::numeric_limits<RT>::epsilon();
    ASSERT_NEAR(solver->get_parameters().default_residual_tol, 0.25, tol);
    ASSERT_EQ(solver->get_parameters().tolerance_type,
              gko::stop::batch::ToleranceType::relative);
}


TYPED_TEST(BatchCg, CanSetResidualTol)
{
    using Solver = typename TestFixture::Solver;
    using RT = typename TestFixture::real_type;
    auto factory =
        Solver::build()
            .with_default_max_iterations(22)
            .with_default_residual_tol(static_cast<RT>(0.25))
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .on(this->exec);
    auto solver = factory->generate(this->mtx);

    solver->set_residual_tolerance(0.5);

    ASSERT_EQ(solver->get_parameters().default_residual_tol, 0.25);
    ASSERT_EQ(solver->get_residual_tolerance(), 0.5);
}


TYPED_TEST(BatchCg, CanSetPreconditionerFactory)
{
    using value_type = typename TestFixture::value_type;
    using Solver = typename TestFixture::Solver;
    auto prec_factory = gko::share(
        gko::preconditioner::BatchJacobi<value_type>::build().on(this->exec));

    auto batchcg_factory = Solver::build()
                               .with_default_max_iterations(3)
                               .with_preconditioner(prec_factory)
                               .on(this->exec);
    auto solver = batchcg_factory->generate(this->mtx);
    auto precond = solver->get_parameters().preconditioner;


    ASSERT_EQ(precond, prec_factory);
}


TYPED_TEST(BatchCg, ThrowsOnRectangularMatrixInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> rectangular_mtx =
        Mtx::create(this->exec, 2, gko::dim<2>{3, 5}, 3);

    ASSERT_THROW(this->batchcg_factory->generate(rectangular_mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(BatchCg, CanSetScalingOps)
{
    using value_type = typename TestFixture::value_type;
    using Solver = typename TestFixture::Solver;
    using Dense = typename TestFixture::Dense;
    using Diag = gko::matrix::BatchDiagonal<value_type>;
    auto left_scale = gko::share(Diag::create(
        this->exec,
        gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrows, this->nrows))));
    auto right_scale = gko::share(Diag::create(
        this->exec,
        gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrows, this->nrows))));
    auto batchcg_factory = Solver::build()
                               .with_left_scaling_op(left_scale)
                               .with_right_scaling_op(right_scale)
                               .on(this->exec);
    auto solver = batchcg_factory->generate(this->mtx);

    ASSERT_EQ(solver->get_left_scaling_op(), left_scale);
    ASSERT_EQ(solver->get_right_scaling_op(), right_scale);
}

// TYPED_TEST(BatchCg, SolverTransposeRetainsFactoryParameters)
// {
//     using Solver = typename TestFixture::Solver;

//     auto batchcg_factory =
//         Solver::build()
//             .with_default_max_iterations(3)
//             .with_default_residual_tol(0.25f)
//             .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
//             .with_preconditioner(gko::preconditioner::batch::type::none)
//             .on(this->exec);
//     auto solver = batchcg_factory->generate(this->mtx);
//     auto solver_trans = gko::as<Solver>(solver->transpose());
//     auto params = solver_trans->get_parameters();

//     ASSERT_EQ(params.preconditioner, gko::preconditioner::batch::type::none);
//     ASSERT_EQ(params.default_max_iterations, 3);
//     ASSERT_EQ(params.default_residual_tol, 0.25);
//     ASSERT_EQ(params.tolerance_type,
//     gko::stop::batch::ToleranceType::relative);
// }


}  // namespace
