// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <typeinfo>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/chebyshev.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


template <typename T>
class Chebyshev : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Chebyshev<value_type>;

    Chebyshev()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          chebyshev_factory(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(3u),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value))
                  .on(exec)),
          solver(chebyshev_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<typename Solver::Factory> chebyshev_factory;
    std::unique_ptr<gko::LinOp> solver;
};

TYPED_TEST_SUITE(Chebyshev, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Chebyshev, ChebyshevFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->chebyshev_factory->get_executor(), this->exec);
}


TYPED_TEST(Chebyshev, ChebyshevFactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    auto solver = static_cast<Solver*>(this->solver.get());

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(3, 3));
    ASSERT_NE(solver->get_system_matrix(), nullptr);
    ASSERT_EQ(solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(Chebyshev, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->chebyshev_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver.get());

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(static_cast<const Mtx*>(copy_mtx.get()),
                        this->mtx.get(), 0.0);
}


TYPED_TEST(Chebyshev, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->chebyshev_factory->generate(Mtx::create(this->exec));

    copy->move_from(this->solver);

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(static_cast<const Mtx*>(copy_mtx.get()),
                        this->mtx.get(), 0.0);
}


TYPED_TEST(Chebyshev, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;

    auto clone = this->solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    auto clone_mtx = static_cast<Solver*>(clone.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(static_cast<const Mtx*>(clone_mtx.get()),
                        this->mtx.get(), 0.0);
}


TYPED_TEST(Chebyshev, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;

    this->solver->clear();

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
    auto solver_mtx =
        static_cast<Solver*>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(Chebyshev, DefaultApplyUsesInitialGuess)
{
    ASSERT_TRUE(this->solver->apply_uses_initial_guess());
}


TYPED_TEST(Chebyshev, CanSetEigenRegion)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    using coeff_type = gko::solver::detail::coeff_type<value_type>;

    std::shared_ptr<Solver> chebyshev_solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_foci(coeff_type{0.2}, coeff_type{1.2})
            .on(this->exec)
            ->generate(this->mtx);

    ASSERT_EQ(chebyshev_solver->get_parameters().foci,
              std::make_pair(coeff_type{0.2}, coeff_type{1.2}));
}


TYPED_TEST(Chebyshev, CanSetInnerSolverInFactory)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;

    auto chebyshev_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_preconditioner(Solver::build().with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u)))
            .on(this->exec);
    auto solver = chebyshev_factory->generate(this->mtx);
    auto preconditioner = dynamic_cast<const Solver*>(
        static_cast<Solver*>(solver.get())->get_preconditioner().get());

    ASSERT_NE(preconditioner, nullptr);
    ASSERT_EQ(preconditioner->get_size(), gko::dim<2>(3, 3));
    ASSERT_EQ(preconditioner->get_system_matrix(), this->mtx);
}


TYPED_TEST(Chebyshev, CanSetGeneratedInnerSolverInFactory)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> chebyshev_solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(this->mtx);

    auto chebyshev_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_generated_preconditioner(chebyshev_solver)
            .on(this->exec);
    auto solver = chebyshev_factory->generate(this->mtx);
    auto preconditioner = solver->get_preconditioner();

    ASSERT_NE(preconditioner.get(), nullptr);
    ASSERT_EQ(preconditioner.get(), chebyshev_solver.get());
}


TYPED_TEST(Chebyshev, CanSetCriteriaAgain)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<gko::stop::CriterionFactory> init_crit =
        gko::stop::Iteration::build().with_max_iters(3u).on(this->exec);
    auto chebyshev_factory =
        Solver::build().with_criteria(init_crit).on(this->exec);

    ASSERT_EQ((chebyshev_factory->get_parameters().criteria).back(), init_crit);

    auto solver = chebyshev_factory->generate(this->mtx);
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


TYPED_TEST(Chebyshev, ThrowsOnWrongInnerSolverInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> wrong_sized_mtx =
        Mtx::create(this->exec, gko::dim<2>{2, 2});
    std::shared_ptr<Solver> chebyshev_solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(wrong_sized_mtx);

    auto chebyshev_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_generated_preconditioner(chebyshev_solver)
            .on(this->exec);

    ASSERT_THROW(chebyshev_factory->generate(this->mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(Chebyshev, CanSetInnerSolver)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> chebyshev_solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(this->mtx);

    auto chebyshev_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec);
    auto solver = chebyshev_factory->generate(this->mtx);
    solver->set_preconditioner(chebyshev_solver);
    auto preconditioner = solver->get_preconditioner();

    ASSERT_NE(preconditioner.get(), nullptr);
    ASSERT_EQ(preconditioner.get(), chebyshev_solver.get());
}


TYPED_TEST(Chebyshev, CanSetApplyWithInitialGuessMode)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    using initial_guess_mode = gko::solver::initial_guess_mode;

    for (auto guess : {initial_guess_mode::provided, initial_guess_mode::rhs,
                       initial_guess_mode::zero}) {
        auto chebyshev_factory =
            Solver::build()
                .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
                .with_default_initial_guess(guess)
                .on(this->exec);
        auto solver = chebyshev_factory->generate(this->mtx);

        ASSERT_EQ(solver->apply_uses_initial_guess(),
                  guess == gko::solver::initial_guess_mode::provided);
    }
}


TYPED_TEST(Chebyshev, ThrowOnWrongInnerSolverSet)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> wrong_sized_mtx =
        Mtx::create(this->exec, gko::dim<2>{2, 2});
    std::shared_ptr<Solver> chebyshev_solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(wrong_sized_mtx);

    auto chebyshev_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec);
    auto solver = chebyshev_factory->generate(this->mtx);

    ASSERT_THROW(solver->set_preconditioner(chebyshev_solver),
                 gko::DimensionMismatch);
}


TYPED_TEST(Chebyshev, ThrowsOnRectangularMatrixInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> rectangular_mtx =
        Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->chebyshev_factory->generate(rectangular_mtx),
                 gko::DimensionMismatch);
}
