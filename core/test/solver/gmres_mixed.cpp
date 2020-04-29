/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <core/solver/gmres_mixed.cpp>
#include <ginkgo/core/solver/gmres_mixed.hpp>

#include <typeinfo>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>

namespace {

class GmresMixed : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using Solver = gko::solver::GmresMixed<>;
    //    using Solver = gko::solver::GmresMixed<double,double>;
    using Big_solver = gko::solver::GmresMixed<double>;

    GmresMixed()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{1.0, 2.0, 3.0}, {3.0, 2.0, -1.0}, {0.0, -1.0, 2}}, exec)),
          gmres_mixed_factory(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(3u).on(exec),
                      gko::stop::ResidualNormReduction<>::build()
                          .with_reduction_factor(1e-6)
                          .on(exec))
                  .on(exec)),
          solver(gmres_mixed_factory->generate(mtx)),
          gmres_mixed_big_factory(
              Big_solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(128u).on(
                          exec),
                      gko::stop::ResidualNormReduction<>::build()
                          .with_reduction_factor(1e-6)
                          .on(exec))
                  .on(exec)),
          big_solver(gmres_mixed_big_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<Solver::Factory> gmres_mixed_factory;
    std::unique_ptr<gko::LinOp> solver;
    std::unique_ptr<Big_solver::Factory> gmres_mixed_big_factory;
    std::unique_ptr<gko::LinOp> big_solver;

    static void assert_same_matrices(const Mtx *m1, const Mtx *m2)
    {
        ASSERT_EQ(m1->get_size()[0], m2->get_size()[0]);
        ASSERT_EQ(m1->get_size()[1], m2->get_size()[1]);
        for (gko::size_type i = 0; i < m1->get_size()[0]; ++i) {
            for (gko::size_type j = 0; j < m2->get_size()[1]; ++j) {
                EXPECT_EQ(m1->at(i, j), m2->at(i, j));
            }
        }
    }
};

TEST_F(GmresMixed, GmresMixedFactoryKnowsItsExecutor)
{
    ASSERT_EQ(gmres_mixed_factory->get_executor(), exec);
}

TEST_F(GmresMixed, GmresMixedFactoryCreatesCorrectSolver)
{
    ASSERT_EQ(solver->get_size(), gko::dim<2>(3, 3));
    auto gmres_mixed_solver = static_cast<Solver *>(solver.get());
    ASSERT_NE(gmres_mixed_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(gmres_mixed_solver->get_system_matrix(), mtx);
}

TEST_F(GmresMixed, CanBeCopied)
{
    auto copy = gmres_mixed_factory->generate(Mtx::create(exec));

    copy->copy_from(solver.get());

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver *>(copy.get())->get_system_matrix();
    assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()), mtx.get());
}

TEST_F(GmresMixed, CanBeMoved)
{
    auto copy = gmres_mixed_factory->generate(Mtx::create(exec));

    copy->copy_from(std::move(solver));

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver *>(copy.get())->get_system_matrix();
    assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()), mtx.get());
}

TEST_F(GmresMixed, CanBeCloned)
{
    auto clone = solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    auto clone_mtx = static_cast<Solver *>(clone.get())->get_system_matrix();
    assert_same_matrices(static_cast<const Mtx *>(clone_mtx.get()), mtx.get());
}

TEST_F(GmresMixed, CanBeCleared)
{
    solver->clear();

    ASSERT_EQ(solver->get_size(), gko::dim<2>(0, 0));
    auto solver_mtx = static_cast<Solver *>(solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}

TEST_F(GmresMixed, CanSetPreconditionerGenerator)
{
    auto gmres_mixed_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(exec),
                gko::stop::ResidualNormReduction<>::build()
                    .with_reduction_factor(1e-6)
                    .on(exec))
            .with_preconditioner(Solver::build().on(exec))
            .on(exec);
    auto solver = gmres_mixed_factory->generate(mtx);
    auto precond = dynamic_cast<const gko::solver::GmresMixed<> *>(
        static_cast<gko::solver::GmresMixed<> *>(solver.get())
            ->get_preconditioner()
            .get());

    ASSERT_NE(precond, nullptr);
    ASSERT_EQ(precond->get_size(), gko::dim<2>(3, 3));
    ASSERT_EQ(precond->get_system_matrix(), mtx);
}

TEST_F(GmresMixed, CanSetKrylovDim)
{
    auto gmres_mixed_factory =
        Solver::build()
            .with_krylov_dim_mixed(4u)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(4u).on(exec),
                gko::stop::ResidualNormReduction<>::build()
                    .with_reduction_factor(1e-6)
                    .on(exec))
            .on(exec);
    auto solver = gmres_mixed_factory->generate(mtx);
    auto krylov_dim_mixed = solver->get_krylov_dim_mixed();

    ASSERT_EQ(krylov_dim_mixed, 4);
}

TEST_F(GmresMixed, CanSetPreconditionerInFactory)
{
    std::shared_ptr<Solver> gmres_mixed_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(exec))
            .on(exec)
            ->generate(mtx);

    auto gmres_mixed_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(exec))
            .with_generated_preconditioner(gmres_mixed_precond)
            .on(exec);
    auto solver = gmres_mixed_factory->generate(mtx);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), gmres_mixed_precond.get());
}

TEST_F(GmresMixed, ThrowsOnWrongPreconditionerInFactory)
{
    std::shared_ptr<Mtx> wrong_sized_mtx = Mtx::create(exec, gko::dim<2>{1, 3});
    std::shared_ptr<Solver> gmres_mixed_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(exec))
            .on(exec)
            ->generate(wrong_sized_mtx);

    auto gmres_mixed_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(exec))
            .with_generated_preconditioner(gmres_mixed_precond)
            .on(exec);

    ASSERT_THROW(gmres_mixed_factory->generate(mtx), gko::DimensionMismatch);
}

TEST_F(GmresMixed, CanSetPreconditioner)
{
    std::shared_ptr<Solver> gmres_mixed_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(exec))
            .on(exec)
            ->generate(mtx);

    auto gmres_mixed_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(exec))
            .on(exec);
    auto solver = gmres_mixed_factory->generate(mtx);
    solver->set_preconditioner(gmres_mixed_precond);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), gmres_mixed_precond.get());
}

}  // namespace
