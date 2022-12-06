/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/solver/batch_gmres.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


template <typename T>
class BatchGmres : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<T>;
    using Mtx = gko::matrix::BatchCsr<value_type>;
    using Dense = gko::matrix::BatchDense<value_type>;
    using Solver = gko::solver::BatchGmres<value_type>;

    BatchGmres()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::test::create_poisson1d_batch<value_type>(
              std::static_pointer_cast<const gko::ReferenceExecutor>(
                  this->exec),
              nrows, nbatch)),
          batchgmres_factory(
              Solver::build()
                  .with_max_iterations(def_max_iters)
                  .with_abs_residual_tol(def_abs_res_tol)
                  .with_tolerance_type(def_tol_type)
                  .with_preconditioner(gko::preconditioner::batch::type::none)
                  .with_restart(2)
                  .on(exec)),
          solver(batchgmres_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    const gko::size_type nbatch = 3;
    const int nrows = 5;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> batchgmres_factory;
    const int def_max_iters = 100;
    const real_type def_abs_res_tol = 1e-11;
    const gko::stop::batch::ToleranceType def_tol_type =
        gko::stop::batch::ToleranceType::absolute;
    std::unique_ptr<gko::BatchLinOp> solver;
};

TYPED_TEST_SUITE(BatchGmres, gko::test::ValueTypes);


TYPED_TEST(BatchGmres, FactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->batchgmres_factory->get_executor(), this->exec);
}


TYPED_TEST(BatchGmres, FactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_EQ(this->solver->get_size().at(i),
                  gko::dim<2>(this->nrows, this->nrows));
    }
    auto batchgmres_solver = static_cast<Solver *>(this->solver.get());
    ASSERT_NE(batchgmres_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(batchgmres_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(BatchGmres, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->batchgmres_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver.get());

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_EQ(copy->get_size().at(i),
                  gko::dim<2>(this->nrows, this->nrows));
    }
    auto copy_mtx = static_cast<Solver *>(copy.get())->get_system_matrix();
    const auto copy_batch_mtx = static_cast<const Mtx *>(copy_mtx.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx.get(), copy_batch_mtx, 0.0);
}


TYPED_TEST(BatchGmres, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->batchgmres_factory->generate(Mtx::create(this->exec));

    copy->copy_from(std::move(this->solver));

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_EQ(copy->get_size().at(i),
                  gko::dim<2>(this->nrows, this->nrows));
    }
    auto copy_mtx = static_cast<Solver *>(copy.get())->get_system_matrix();
    const auto copy_batch_mtx = static_cast<const Mtx *>(copy_mtx.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx.get(), copy_batch_mtx, 0.0);
}


TYPED_TEST(BatchGmres, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto clone = this->solver->clone();

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_EQ(clone->get_size().at(i),
                  gko::dim<2>(this->nrows, this->nrows));
    }
    auto clone_mtx = static_cast<Solver *>(clone.get())->get_system_matrix();
    const auto clone_batch_mtx = static_cast<const Mtx *>(clone_mtx.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx.get(), clone_batch_mtx, 0.0);
}


TYPED_TEST(BatchGmres, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;

    this->solver->clear();

    ASSERT_EQ(this->solver->get_num_batch_entries(), 0);
    ASSERT_EQ(this->solver->get_size().get_num_batch_entries(), 0);
    auto solver_mtx =
        static_cast<Solver *>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(BatchGmres, ApplyUsesInitialGuessReturnsTrue)
{
    ASSERT_TRUE(this->solver->apply_uses_initial_guess());
}


TYPED_TEST(BatchGmres, CanSetCriteria)
{
    using Solver = typename TestFixture::Solver;
    using RT = typename TestFixture::real_type;

    auto batchgmres_factory =
        Solver::build()
            .with_max_iterations(22)
            .with_rel_residual_tol(static_cast<RT>(0.25))
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_restart(3)
            .on(this->exec);
    auto solver = batchgmres_factory->generate(this->mtx);

    ASSERT_EQ(solver->get_parameters().max_iterations, 22);
    const RT tol = std::numeric_limits<RT>::epsilon();
    ASSERT_NEAR(solver->get_parameters().rel_residual_tol, 0.25, tol);
    ASSERT_EQ(solver->get_parameters().tolerance_type,
              gko::stop::batch::ToleranceType::relative);
    ASSERT_EQ(solver->get_parameters().restart, 3);
}


TYPED_TEST(BatchGmres, CanChangeCriteriaThroughSolverObject)
{
    using Solver = typename TestFixture::Solver;
    using RT = typename TestFixture::real_type;

    auto batchgmres_factory = Solver::build().with_restart(3).on(this->exec);
    auto solver = batchgmres_factory->generate(this->mtx);

    ASSERT_EQ(solver->get_parameters().restart, 3);
    ASSERT_EQ(solver->get_restart_number(), 3);

    solver->set_restart_number(2);

    ASSERT_EQ(solver->get_restart_number(), 2);
    ASSERT_EQ(solver->get_parameters().restart, 2);
}


TYPED_TEST(BatchGmres, CanSetPreconditionerInFactory)
{
    using Solver = typename TestFixture::Solver;
    const auto batchgmres_precond = gko::preconditioner::batch::type::none;

    auto batchgmres_factory =
        Solver::build()
            .with_max_iterations(3)
            .with_preconditioner(gko::preconditioner::batch::type::none)
            .on(this->exec);
    auto solver = batchgmres_factory->generate(this->mtx);
    auto precond = solver->get_parameters().preconditioner;


    ASSERT_EQ(precond, batchgmres_precond);
}


TYPED_TEST(BatchGmres, ThrowsOnRectangularMatrixInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> rectangular_mtx =
        Mtx::create(this->exec, 2, gko::dim<2>{3, 5}, 3);

    ASSERT_THROW(this->batchgmres_factory->generate(rectangular_mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(BatchGmres, CanSetScalingVectors)
{
    using value_type = typename TestFixture::value_type;
    using Solver = typename TestFixture::Solver;
    using Dense = typename TestFixture::Dense;

    auto batchgmres_factory = Solver::build().on(this->exec);
    auto solver = batchgmres_factory->generate(this->mtx);
    auto left_scale = Dense::create(
        this->exec, gko::batch_dim<>(2, gko::dim<2>(this->nrows, 1)));
    auto right_scale = Dense::create_with_config_of(left_scale.get());
    solver->batch_scale(left_scale.get(), right_scale.get());

    auto s_solver =
        dynamic_cast<gko::EnableBatchScaledSolver<value_type> *>(solver.get());
    ASSERT_TRUE(s_solver);
    ASSERT_EQ(s_solver->get_left_scaling_vector(), left_scale.get());
    ASSERT_EQ(s_solver->get_right_scaling_vector(), right_scale.get());
}

// TYPED_TEST(BatchGmres, SolverTransposeRetainsFactoryParameters)
// {
//    using Solver = typename TestFixture::Solver;

//    auto batchgmres_factory =
//        Solver::build()
//            .with_max_iterations(3)
//            .with_rel_residual_tol(0.25f)
//            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
//            .with_preconditioner(gko::preconditioner::batch::type::none)
//            .on(this->exec);
//    auto solver = batchgmres_factory->generate(this->mtx);
//    auto solver_trans = gko::as<Solver>(solver->transpose());
//    auto params = solver_trans->get_parameters();

//    ASSERT_EQ(params.preconditioner, gko::preconditioner::batch::type::none);
//    ASSERT_EQ(params.max_iterations, 3);
//    ASSERT_EQ(params.rel_residual_tol, 0.25);
//    ASSERT_EQ(params.tolerance_type,
//    gko::stop::batch::ToleranceType::relative);
// }


}  // namespace
