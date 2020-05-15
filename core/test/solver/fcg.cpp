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


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Fcg : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Fcg<value_type>;

    Fcg()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          fcg_factory(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(3u).on(exec),
                      gko::stop::ResidualNormReduction<value_type>::build()
                          .with_reduction_factor(gko::remove_complex<T>{1e-6})
                          .on(exec))
                  .on(exec)),
          solver(fcg_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> fcg_factory;
    std::unique_ptr<gko::LinOp> solver;
};

TYPED_TEST_CASE(Fcg, gko::test::ValueTypes);


TYPED_TEST(Fcg, FcgFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->fcg_factory->get_executor(), this->exec);
}


TYPED_TEST(Fcg, FcgFactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(3, 3));
    auto fcg_solver = dynamic_cast<Solver *>(this->solver.get());
    ASSERT_NE(fcg_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(fcg_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(Fcg, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->fcg_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver.get());

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = dynamic_cast<Solver *>(copy.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(dynamic_cast<const Mtx *>(copy_mtx.get()),
                        this->mtx.get(), 0.0);
}


TYPED_TEST(Fcg, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->fcg_factory->generate(Mtx::create(this->exec));

    copy->copy_from(std::move(this->solver));

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = dynamic_cast<Solver *>(copy.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(dynamic_cast<const Mtx *>(copy_mtx.get()),
                        this->mtx.get(), 0.0);
}


TYPED_TEST(Fcg, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto clone = this->solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    auto clone_mtx = dynamic_cast<Solver *>(clone.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(dynamic_cast<const Mtx *>(clone_mtx.get()),
                        this->mtx.get(), 0.0);
}


TYPED_TEST(Fcg, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;
    this->solver->clear();

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
    auto solver_mtx =
        static_cast<Solver *>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(Fcg, ApplyUsesInitialGuessReturnsTrue)
{
    ASSERT_TRUE(this->solver->apply_uses_initial_guess());
}


TYPED_TEST(Fcg, CanSetPreconditionerGenerator)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto fcg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec),
                gko::stop::ResidualNormReduction<value_type>::build()
                    .with_reduction_factor(
                        gko::remove_complex<value_type>(1e-6))
                    .on(this->exec))
            .with_preconditioner(Solver::build().on(this->exec))
            .on(this->exec);
    auto solver = fcg_factory->generate(this->mtx);
    auto precond = dynamic_cast<const gko::solver::Fcg<value_type> *>(
        static_cast<gko::solver::Fcg<value_type> *>(solver.get())
            ->get_preconditioner()
            .get());

    ASSERT_NE(precond, nullptr);
    ASSERT_EQ(precond->get_size(), gko::dim<2>(3, 3));
    ASSERT_EQ(precond->get_system_matrix(), this->mtx);
}


TYPED_TEST(Fcg, CanSetPreconditionerInFactory)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> fcg_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec)
            ->generate(this->mtx);

    auto fcg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .with_generated_preconditioner(fcg_precond)
            .on(this->exec);
    auto solver = fcg_factory->generate(this->mtx);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), fcg_precond.get());
}


TYPED_TEST(Fcg, ThrowsOnWrongPreconditionerInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> wrong_sized_mtx =
        Mtx::create(this->exec, gko::dim<2>{1, 3});
    std::shared_ptr<Solver> fcg_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec)
            ->generate(wrong_sized_mtx);

    auto fcg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .with_generated_preconditioner(fcg_precond)
            .on(this->exec);

    ASSERT_THROW(fcg_factory->generate(this->mtx), gko::DimensionMismatch);
}


TYPED_TEST(Fcg, CanSetPreconditioner)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> fcg_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec)
            ->generate(this->mtx);

    auto fcg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec);
    auto solver = fcg_factory->generate(this->mtx);
    solver->set_preconditioner(fcg_precond);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), fcg_precond.get());
}


}  // namespace
