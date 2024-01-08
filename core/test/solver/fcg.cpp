// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/fcg.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


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
                      gko::stop::Iteration::build().with_max_iters(3u),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(gko::remove_complex<T>{1e-6}))
                  .on(exec)),
          solver(fcg_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> fcg_factory;
    std::unique_ptr<gko::LinOp> solver;
};

TYPED_TEST_SUITE(Fcg, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Fcg, FcgFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->fcg_factory->get_executor(), this->exec);
}


TYPED_TEST(Fcg, FcgFactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(3, 3));
    auto fcg_solver = dynamic_cast<Solver*>(this->solver.get());
    ASSERT_NE(fcg_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(fcg_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(Fcg, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->fcg_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver);

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = dynamic_cast<Solver*>(copy.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(dynamic_cast<const Mtx*>(copy_mtx.get()), this->mtx,
                        0.0);
}


TYPED_TEST(Fcg, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->fcg_factory->generate(Mtx::create(this->exec));

    copy->move_from(this->solver);

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = dynamic_cast<Solver*>(copy.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(dynamic_cast<const Mtx*>(copy_mtx.get()), this->mtx,
                        0.0);
}


TYPED_TEST(Fcg, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto clone = this->solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    auto clone_mtx = dynamic_cast<Solver*>(clone.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(dynamic_cast<const Mtx*>(clone_mtx.get()), this->mtx,
                        0.0);
}


TYPED_TEST(Fcg, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;
    this->solver->clear();

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
    auto solver_mtx =
        static_cast<Solver*>(this->solver.get())->get_system_matrix();
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
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(
                                   gko::remove_complex<value_type>(1e-6)))
            .with_preconditioner(Solver::build().with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u)))
            .on(this->exec);
    auto solver = fcg_factory->generate(this->mtx);
    auto precond = dynamic_cast<const gko::solver::Fcg<value_type>*>(
        static_cast<gko::solver::Fcg<value_type>*>(solver.get())
            ->get_preconditioner()
            .get());

    ASSERT_NE(precond, nullptr);
    ASSERT_EQ(precond->get_size(), gko::dim<2>(3, 3));
    ASSERT_EQ(precond->get_system_matrix(), this->mtx);
}


TYPED_TEST(Fcg, CanSetCriteriaAgain)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<gko::stop::CriterionFactory> init_crit =
        gko::stop::Iteration::build().with_max_iters(3u).on(this->exec);
    auto fcg_factory = Solver::build().with_criteria(init_crit).on(this->exec);

    ASSERT_EQ((fcg_factory->get_parameters().criteria).back(), init_crit);

    auto solver = fcg_factory->generate(this->mtx);
    std::shared_ptr<gko::stop::CriterionFactory> new_crit =
        gko::stop::Iteration::build().with_max_iters(5u).on(this->exec);

    solver->set_stop_criterion_factory(new_crit);
    auto new_crit_fac = solver->get_stop_criterion_factory();
    auto niter =
        static_cast<const gko::stop::Iteration::Factory*>(new_crit_fac.get())
            ->get_parameters()
            .max_iters;

    ASSERT_EQ(niter, 5);
}


TYPED_TEST(Fcg, CanSetPreconditionerInFactory)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> fcg_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(this->mtx);

    auto fcg_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
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
        Mtx::create(this->exec, gko::dim<2>{2, 2});
    std::shared_ptr<Solver> fcg_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(wrong_sized_mtx);

    auto fcg_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_generated_preconditioner(fcg_precond)
            .on(this->exec);

    ASSERT_THROW(fcg_factory->generate(this->mtx), gko::DimensionMismatch);
}


TYPED_TEST(Fcg, ThrowsOnRectangularMatrixInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> rectangular_mtx =
        Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->fcg_factory->generate(rectangular_mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(Fcg, CanSetPreconditioner)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> fcg_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(this->mtx);

    auto fcg_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec);
    auto solver = fcg_factory->generate(this->mtx);
    solver->set_preconditioner(fcg_precond);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), fcg_precond.get());
}


TYPED_TEST(Fcg, PassExplicitFactory)
{
    using Solver = typename TestFixture::Solver;
    auto stop_factory = gko::share(
        gko::stop::Iteration::build().with_max_iters(1u).on(this->exec));
    auto precond_factory = gko::share(Solver::build().on(this->exec));

    auto factory = Solver::build()
                       .with_criteria(stop_factory)
                       .with_preconditioner(precond_factory)
                       .on(this->exec);

    ASSERT_EQ(factory->get_parameters().criteria.front(), stop_factory);
    ASSERT_EQ(factory->get_parameters().preconditioner, precond_factory);
}


}  // namespace
