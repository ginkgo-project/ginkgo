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

#include <ginkgo/core/solver/fcg.hpp>


#include <gtest/gtest.h>


#include <core/test/utils.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>


namespace {


class Fcg : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using Solver = gko::solver::Fcg<>;

    Fcg()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          fcg_factory(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(3u).on(exec),
                      gko::stop::ResidualNormReduction<>::build()
                          .with_reduction_factor(1e-6)
                          .on(exec))
                  .on(exec)),
          solver(fcg_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<Solver::Factory> fcg_factory;
    std::unique_ptr<gko::LinOp> solver;
};


TEST_F(Fcg, FcgFactoryKnowsItsExecutor)
{
    ASSERT_EQ(fcg_factory->get_executor(), exec);
}


TEST_F(Fcg, FcgFactoryCreatesCorrectSolver)
{
    ASSERT_EQ(solver->get_size(), gko::dim<2>(3, 3));
    auto fcg_solver = dynamic_cast<Solver *>(solver.get());
    ASSERT_NE(fcg_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(fcg_solver->get_system_matrix(), mtx);
}


TEST_F(Fcg, CanBeCopied)
{
    auto copy = fcg_factory->generate(Mtx::create(exec));

    copy->copy_from(solver.get());

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = dynamic_cast<Solver *>(copy.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(dynamic_cast<const Mtx *>(copy_mtx.get()), mtx.get(),
                        1e-14);
}


TEST_F(Fcg, CanBeMoved)
{
    auto copy = fcg_factory->generate(Mtx::create(exec));

    copy->copy_from(std::move(solver));

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = dynamic_cast<Solver *>(copy.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(dynamic_cast<const Mtx *>(copy_mtx.get()), mtx.get(),
                        1e-14);
}


TEST_F(Fcg, CanBeCloned)
{
    auto clone = solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    auto clone_mtx = dynamic_cast<Solver *>(clone.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(dynamic_cast<const Mtx *>(clone_mtx.get()), mtx.get(),
                        1e-14);
}


TEST_F(Fcg, CanBeCleared)
{
    solver->clear();

    ASSERT_EQ(solver->get_size(), gko::dim<2>(0, 0));
    auto solver_mtx = static_cast<Solver *>(solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TEST_F(Fcg, CanSetPreconditionerGenerator)
{
    auto fcg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(exec),
                gko::stop::ResidualNormReduction<>::build()
                    .with_reduction_factor(1e-6)
                    .on(exec))
            .with_preconditioner(Solver::build().on(exec))
            .on(exec);
    auto solver = fcg_factory->generate(mtx);
    auto precond = dynamic_cast<const gko::solver::Fcg<> *>(
        static_cast<gko::solver::Fcg<> *>(solver.get())
            ->get_preconditioner()
            .get());

    ASSERT_NE(precond, nullptr);
    ASSERT_EQ(precond->get_size(), gko::dim<2>(3, 3));
    ASSERT_EQ(precond->get_system_matrix(), mtx);
}


TEST_F(Fcg, CanSetPreconditionerInFactory)
{
    std::shared_ptr<Solver> fcg_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(exec))
            .on(exec)
            ->generate(mtx);

    auto fcg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(exec))
            .with_generated_preconditioner(fcg_precond)
            .on(exec);
    auto solver = fcg_factory->generate(mtx);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), fcg_precond.get());
}


TEST_F(Fcg, ThrowsOnWrongPreconditionerInFactory)
{
    std::shared_ptr<Mtx> wrong_sized_mtx = Mtx::create(exec, gko::dim<2>{1, 3});
    std::shared_ptr<Solver> fcg_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(exec))
            .on(exec)
            ->generate(wrong_sized_mtx);

    auto fcg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(exec))
            .with_generated_preconditioner(fcg_precond)
            .on(exec);

    ASSERT_THROW(fcg_factory->generate(mtx), gko::DimensionMismatch);
}


TEST_F(Fcg, CanSetPreconditioner)
{
    std::shared_ptr<Solver> fcg_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(exec))
            .on(exec)
            ->generate(mtx);

    auto fcg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(exec))
            .on(exec);
    auto solver = fcg_factory->generate(mtx);
    solver->set_preconditioner(fcg_precond);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), fcg_precond.get());
}


}  // namespace
