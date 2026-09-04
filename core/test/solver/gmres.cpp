// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <typeinfo>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename T>
class Gmres : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Gmres<value_type>;
    using Big_solver = gko::solver::Gmres<double>;

    const gko::remove_complex<T> reduction_factor =
        r<gko::remove_complex<T>>::value;

    Gmres()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{1.0, 2.0, 3.0}, {3.0, 2.0, -1.0}, {0.0, -1.0, 2}}, exec)),
          gmres_factory(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(3u),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(reduction_factor))
                  .on(exec)),
          solver(gmres_factory->generate(mtx)),
          gmres_big_factory(
              Big_solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(128u),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(reduction_factor))
                  .on(exec)),
          big_solver(gmres_big_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> gmres_factory;
    std::unique_ptr<gko::LinOp> solver;
    std::unique_ptr<Big_solver::Factory> gmres_big_factory;
    std::unique_ptr<gko::LinOp> big_solver;
};

TYPED_TEST_SUITE(Gmres, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Gmres, GmresFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->gmres_factory->get_executor(), this->exec);
}


TYPED_TEST(Gmres, GmresFactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(3, 3));
    auto gmres_solver = static_cast<Solver*>(this->solver.get());
    ASSERT_NE(gmres_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(gmres_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(Gmres, ApplyUsesInitialGuessReturnsTrue)
{
    ASSERT_TRUE(this->solver->apply_uses_initial_guess());
}


TYPED_TEST(Gmres, CanSetPreconditionerGenerator)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto gmres_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(this->reduction_factor))
            .with_preconditioner(Solver::build().with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u)))
            .on(this->exec);
    auto solver = gmres_factory->generate(this->mtx);
    auto precond = dynamic_cast<const gko::solver::Gmres<value_type>*>(
        static_cast<gko::solver::Gmres<value_type>*>(solver.get())
            ->get_preconditioner()
            .get());

    ASSERT_NE(precond, nullptr);
    ASSERT_EQ(precond->get_size(), gko::dim<2>(3, 3));
    ASSERT_EQ(precond->get_system_matrix(), this->mtx);
}


TYPED_TEST(Gmres, CanSetCriteriaAgain)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<gko::stop::CriterionFactory> init_crit =
        gko::stop::Iteration::build().with_max_iters(3u).on(this->exec);
    auto gmres_factory =
        Solver::build().with_criteria(init_crit).on(this->exec);

    ASSERT_EQ((gmres_factory->get_parameters().criteria).back(), init_crit);

    auto solver = gmres_factory->generate(this->mtx);
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


TYPED_TEST(Gmres, CanSetKrylovDim)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto gmres_factory =
        Solver::build()
            .with_krylov_dim(4u)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(4u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(this->reduction_factor))
            .on(this->exec);
    auto solver = gmres_factory->generate(this->mtx);
    auto krylov_dim = solver->get_krylov_dim();

    ASSERT_EQ(krylov_dim, 4);
}


TYPED_TEST(Gmres, CanSetKrylovDimAgain)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<gko::stop::CriterionFactory> init_crit =
        gko::stop::Iteration::build().with_max_iters(3u).on(this->exec);
    auto gmres_factory =
        Solver::build().with_criteria(init_crit).with_krylov_dim(10u).on(
            this->exec);

    ASSERT_EQ(gmres_factory->get_parameters().krylov_dim, 10);

    auto solver = gmres_factory->generate(this->mtx);

    solver->set_krylov_dim(20);

    ASSERT_EQ(solver->get_krylov_dim(), 20);
}


TYPED_TEST(Gmres, RestartRatioIsZeroByDefault)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;
    auto gmres_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(4u))
            .on(this->exec);
    auto solver = gmres_factory->generate(this->mtx);

    ASSERT_EQ(solver->get_restart_ratio(), real_type{0});
}


TYPED_TEST(Gmres, CanSetRestartRatio)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;
    const real_type new_restart_ratio{0.9};

    auto gmres_factory =
        Solver::build()
            .with_restart_ratio(new_restart_ratio)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(4u))
            .on(this->exec);
    auto solver = gmres_factory->generate(this->mtx);

    ASSERT_EQ(solver->get_restart_ratio(), new_restart_ratio);
}


TYPED_TEST(Gmres, CanSetRestartRatioAgain)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;
    const real_type new_restart_ratio{0.95};
    auto gmres_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(4u))
            .on(this->exec);
    auto solver = gmres_factory->generate(this->mtx);

    solver->set_restart_ratio(new_restart_ratio);

    ASSERT_EQ(solver->get_restart_ratio(), new_restart_ratio);
}


TYPED_TEST(Gmres, RejectsRestartRatioOutsideUnitInterval)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;
    auto build_with = [this](real_type tol) {
        return Solver::build()
            .with_restart_ratio(tol)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(4u))
            .on(this->exec);
    };

    ASSERT_THROW(build_with(real_type{1.0})->generate(this->mtx),
                 gko::InvalidStateError);
    ASSERT_THROW(build_with(real_type{2.0})->generate(this->mtx),
                 gko::InvalidStateError);
    ASSERT_THROW(build_with(-real_type{0.5})->generate(this->mtx),
                 gko::InvalidStateError);
}


TYPED_TEST(Gmres, CanSetPreconditionerInFactory)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> gmres_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(this->mtx);

    auto gmres_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_generated_preconditioner(gmres_precond)
            .on(this->exec);
    auto solver = gmres_factory->generate(this->mtx);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), gmres_precond.get());
}


TYPED_TEST(Gmres, ThrowsOnWrongPreconditionerInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> wrong_sized_mtx =
        Mtx::create(this->exec, gko::dim<2>{2, 2});
    std::shared_ptr<Solver> gmres_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(wrong_sized_mtx);

    auto gmres_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_generated_preconditioner(gmres_precond)
            .on(this->exec);

    ASSERT_THROW(gmres_factory->generate(this->mtx), gko::DimensionMismatch);
}


TYPED_TEST(Gmres, ThrowsOnRectangularMatrixInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> rectangular_mtx =
        Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->gmres_factory->generate(rectangular_mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(Gmres, CanSetPreconditioner)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> gmres_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(this->mtx);

    auto gmres_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec);
    auto solver = gmres_factory->generate(this->mtx);
    solver->set_preconditioner(gmres_precond);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), gmres_precond.get());
}


TYPED_TEST(Gmres, PassExplicitFactory)
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
