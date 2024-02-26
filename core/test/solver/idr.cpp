// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/idr.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Idr : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Idr<value_type>;

    Idr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          idr_factory(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(3u),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(gko::remove_complex<T>{1e-6}))
                  .on(exec)),
          solver(idr_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> idr_factory;
    std::unique_ptr<gko::LinOp> solver;
};

TYPED_TEST_SUITE(Idr, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Idr, IdrFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->idr_factory->get_executor(), this->exec);
}


TYPED_TEST(Idr, IdrFactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(3, 3));
    auto idr_solver = gko::as<Solver>(this->solver.get());
    ASSERT_NE(idr_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(idr_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(Idr, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->idr_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver);

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = gko::as<Solver>(copy.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_mtx), this->mtx, 0.0);
}


TYPED_TEST(Idr, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->idr_factory->generate(Mtx::create(this->exec));

    copy->move_from(this->solver);

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = gko::as<Solver>(copy.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_mtx), this->mtx, 0.0);
}


TYPED_TEST(Idr, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;

    auto clone = this->solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    auto clone_mtx = gko::as<Solver>(clone.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(clone_mtx.get()), this->mtx, 0.0);
}


TYPED_TEST(Idr, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;

    this->solver->clear();

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
    auto solver_mtx = gko::as<Solver>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(Idr, ApplyUsesInitialGuessReturnsTrue)
{
    ASSERT_TRUE(this->solver->apply_uses_initial_guess());
}


TYPED_TEST(Idr, CanSetPreconditionerGenerator)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto idr_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_preconditioner(Solver::build().with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u)))
            .on(this->exec);

    auto solver = idr_factory->generate(this->mtx);
    auto precond =
        gko::as<gko::solver::Idr<value_type>>(solver->get_preconditioner());

    ASSERT_EQ(precond->get_size(), gko::dim<2>(3, 3));
    ASSERT_EQ(precond->get_system_matrix(), this->mtx);
}


TYPED_TEST(Idr, CanSetCriteriaAgain)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<gko::stop::CriterionFactory> init_crit =
        gko::stop::Iteration::build().with_max_iters(3u).on(this->exec);
    auto idr_factory = Solver::build().with_criteria(init_crit).on(this->exec);

    ASSERT_EQ((idr_factory->get_parameters().criteria).back(), init_crit);

    auto solver = idr_factory->generate(this->mtx);
    std::shared_ptr<gko::stop::CriterionFactory> new_crit =
        gko::stop::Iteration::build().with_max_iters(5u).on(this->exec);

    solver->set_stop_criterion_factory(new_crit);
    auto new_crit_fac = solver->get_stop_criterion_factory();
    auto niter = gko::as<gko::stop::Iteration::Factory>(new_crit_fac.get())
                     ->get_parameters()
                     .max_iters;

    ASSERT_EQ(niter, 5);
}


TYPED_TEST(Idr, CanSetPreconditionerInFactory)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> idr_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(this->mtx);

    auto idr_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_generated_preconditioner(idr_precond)
            .on(this->exec);
    auto solver = idr_factory->generate(this->mtx);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), idr_precond.get());
}


TYPED_TEST(Idr, ThrowsOnWrongPreconditionerInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> wrong_sized_mtx =
        Mtx::create(this->exec, gko::dim<2>{2, 2});
    std::shared_ptr<Solver> idr_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(wrong_sized_mtx);

    auto idr_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_generated_preconditioner(idr_precond)
            .on(this->exec);

    ASSERT_THROW(idr_factory->generate(this->mtx), gko::DimensionMismatch);
}


TYPED_TEST(Idr, CanSetPreconditioner)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> idr_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(this->mtx);

    auto idr_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec);
    auto solver = idr_factory->generate(this->mtx);
    solver->set_preconditioner(idr_precond);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), idr_precond.get());
}


TYPED_TEST(Idr, CanSetSubspaceDim)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto idr_factory =
        Solver::build()
            .with_subspace_dim(8u)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(4u))
            .on(this->exec);
    auto solver = idr_factory->generate(this->mtx);
    auto subspace_dim = solver->get_subspace_dim();

    ASSERT_EQ(subspace_dim, 8u);
}


TYPED_TEST(Idr, CanSetSubspaceDimAgain)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<gko::stop::CriterionFactory> init_crit =
        gko::stop::Iteration::build().with_max_iters(3u).on(this->exec);
    auto idr_factory =
        Solver::build().with_criteria(init_crit).with_subspace_dim(10u).on(
            this->exec);

    ASSERT_EQ(idr_factory->get_parameters().subspace_dim, 10);

    auto solver = idr_factory->generate(this->mtx);

    solver->set_subspace_dim(20);

    ASSERT_EQ(solver->get_subspace_dim(), 20);
}


TYPED_TEST(Idr, CanSetKappa)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;
    auto idr_factory =
        Solver::build()
            .with_kappa(real_type{0.05})
            .with_criteria(gko::stop::Iteration::build().with_max_iters(4u))
            .on(this->exec);
    auto solver = idr_factory->generate(this->mtx);
    auto kappa = solver->get_kappa();

    ASSERT_EQ(kappa, real_type{0.05});
}


TYPED_TEST(Idr, CanSetKappaAgain)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;
    std::shared_ptr<gko::stop::CriterionFactory> init_crit =
        gko::stop::Iteration::build().with_max_iters(3u).on(this->exec);
    auto idr_factory = Solver::build()
                           .with_criteria(init_crit)
                           .with_kappa(real_type{0.05})
                           .on(this->exec);

    ASSERT_EQ(idr_factory->get_parameters().kappa, real_type{0.05});

    auto solver = idr_factory->generate(this->mtx);

    solver->set_kappa(real_type{0.3});

    ASSERT_EQ(solver->get_kappa(), real_type{0.3});
}


TYPED_TEST(Idr, CanSetDeterministic)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto idr_factory =
        Solver::build()
            .with_deterministic(true)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(4u))
            .on(this->exec);
    auto solver = idr_factory->generate(this->mtx);
    auto deterministic = solver->get_deterministic();

    ASSERT_EQ(deterministic, true);
}


TYPED_TEST(Idr, CanSetDeterministicAgain)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    std::shared_ptr<gko::stop::CriterionFactory> init_crit =
        gko::stop::Iteration::build().with_max_iters(3u).on(this->exec);
    auto idr_factory =
        Solver::build().with_criteria(init_crit).with_deterministic(true).on(
            this->exec);

    ASSERT_EQ(idr_factory->get_parameters().deterministic, true);

    auto solver = idr_factory->generate(this->mtx);

    solver->set_deterministic(false);

    ASSERT_EQ(solver->get_deterministic(), false);
}


TYPED_TEST(Idr, CanSetComplexSubspace)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto idr_factory =
        Solver::build()
            .with_complex_subspace(true)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(4u))
            .on(this->exec);
    auto solver = idr_factory->generate(this->mtx);
    auto complex_subspace = solver->get_complex_subspace();

    ASSERT_EQ(complex_subspace, true);
}


TYPED_TEST(Idr, CanSetComplexSubspaceAgain)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    std::shared_ptr<gko::stop::CriterionFactory> init_crit =
        gko::stop::Iteration::build().with_max_iters(3u).on(this->exec);
    auto idr_factory =
        Solver::build().with_criteria(init_crit).with_complex_subspace(true).on(
            this->exec);

    ASSERT_EQ(idr_factory->get_parameters().complex_subspace, true);

    auto solver = idr_factory->generate(this->mtx);

    solver->set_complex_subspace(false);

    ASSERT_EQ(solver->get_complex_subspace(), false);
}


TYPED_TEST(Idr, PassExplicitFactory)
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
