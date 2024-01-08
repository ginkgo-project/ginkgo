// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/bicg.hpp>


#include <typeinfo>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Bicg : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Bicg<value_type>;

    Bicg()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          bicg_factory(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(3u),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(gko::remove_complex<T>{1e-6}))
                  .on(exec)),
          solver(bicg_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> bicg_factory;
    std::unique_ptr<gko::LinOp> solver;
};

TYPED_TEST_SUITE(Bicg, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Bicg, BicgFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->bicg_factory->get_executor(), this->exec);
}


TYPED_TEST(Bicg, BicgFactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(3, 3));
    auto bicg_solver = static_cast<Solver*>(this->solver.get());
    ASSERT_NE(bicg_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(bicg_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(Bicg, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->bicg_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver);

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_mtx), this->mtx, 0.0);
}


TYPED_TEST(Bicg, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->bicg_factory->generate(Mtx::create(this->exec));

    copy->move_from(this->solver);

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_mtx), this->mtx, 0.0);
}


TYPED_TEST(Bicg, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto clone = this->solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    auto clone_mtx = static_cast<Solver*>(clone.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(clone_mtx), this->mtx, 0.0);
}


TYPED_TEST(Bicg, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;
    this->solver->clear();

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
    auto solver_mtx =
        static_cast<Solver*>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(Bicg, ApplyUsesInitialGuessReturnsTrue)
{
    using Solver = typename TestFixture::Solver;
    ASSERT_TRUE(this->solver->apply_uses_initial_guess());
}


TYPED_TEST(Bicg, CanSetPreconditionerGenerator)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto bicg_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(
                                   gko::remove_complex<value_type>(1e-6)))
            .with_preconditioner(Solver::build().with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u)))
            .on(this->exec);
    auto solver = bicg_factory->generate(this->mtx);
    auto precond = dynamic_cast<const gko::solver::Bicg<value_type>*>(
        static_cast<gko::solver::Bicg<value_type>*>(solver.get())
            ->get_preconditioner()
            .get());

    ASSERT_NE(precond, nullptr);
    ASSERT_EQ(precond->get_size(), gko::dim<2>(3, 3));
    ASSERT_EQ(precond->get_system_matrix(), this->mtx);
}


TYPED_TEST(Bicg, CanSetPreconditionerInFactory)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> bicg_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(this->mtx);

    auto bicg_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_generated_preconditioner(bicg_precond)
            .on(this->exec);
    auto solver = bicg_factory->generate(this->mtx);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), bicg_precond.get());
}


TYPED_TEST(Bicg, CanSetCriteriaAgain)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<gko::stop::CriterionFactory> init_crit =
        gko::stop::Iteration::build().with_max_iters(3u).on(this->exec);
    auto bicg_factory = Solver::build().with_criteria(init_crit).on(this->exec);

    ASSERT_EQ((bicg_factory->get_parameters().criteria).back(), init_crit);

    auto solver = bicg_factory->generate(this->mtx);
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


TYPED_TEST(Bicg, ThrowsOnWrongPreconditionerInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> wrong_sized_mtx =
        Mtx::create(this->exec, gko::dim<2>{2, 2});
    std::shared_ptr<Solver> bicg_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(wrong_sized_mtx);

    auto bicg_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_generated_preconditioner(bicg_precond)
            .on(this->exec);

    ASSERT_THROW(bicg_factory->generate(this->mtx), gko::DimensionMismatch);
}


TYPED_TEST(Bicg, ThrowsOnRectangularMatrixInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> rectangular_mtx =
        Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->bicg_factory->generate(rectangular_mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bicg, CanSetPreconditioner)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> bicg_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(this->mtx);

    auto bicg_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec);
    auto solver = bicg_factory->generate(this->mtx);
    solver->set_preconditioner(bicg_precond);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), bicg_precond.get());
}


TYPED_TEST(Bicg, PassExplicitFactory)
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
