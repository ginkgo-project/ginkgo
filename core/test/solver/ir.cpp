// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/ir.hpp>


#include <typeinfo>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/log/profiler_hook.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Ir : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Ir<value_type>;

    Ir()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          ir_factory(Solver::build()
                         .with_criteria(
                             gko::stop::Iteration::build().with_max_iters(3u),
                             gko::stop::ResidualNorm<value_type>::build()
                                 .with_reduction_factor(r<value_type>::value))
                         .on(exec)),
          solver(ir_factory->generate(mtx))
    {}

    std::shared_ptr<gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<typename Solver::Factory> ir_factory;
    std::unique_ptr<gko::LinOp> solver;
};

TYPED_TEST_SUITE(Ir, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Ir, IrFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->ir_factory->get_executor(), this->exec);
}


TYPED_TEST(Ir, IrFactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(3, 3));
    auto cg_solver = static_cast<Solver*>(this->solver.get());
    ASSERT_NE(cg_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(cg_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(Ir, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->ir_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver);

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_mtx), this->mtx, 0.0);
}


TYPED_TEST(Ir, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->ir_factory->generate(Mtx::create(this->exec));

    copy->move_from(this->solver);

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_mtx), this->mtx, 0.0);
}


TYPED_TEST(Ir, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto clone = this->solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    auto clone_mtx = static_cast<Solver*>(clone.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(clone_mtx), this->mtx, 0.0);
}


TYPED_TEST(Ir, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;
    this->solver->clear();

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
    auto solver_mtx =
        static_cast<Solver*>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(Ir, DefaultApplyUsesInitialGuess)
{
    ASSERT_TRUE(this->solver->apply_uses_initial_guess());
}


TYPED_TEST(Ir, CanSetInnerSolverInFactory)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto ir_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(r<value_type>::value))
            .with_solver(Solver::build().with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u)))
            .on(this->exec);
    auto solver = ir_factory->generate(this->mtx);
    auto inner_solver = dynamic_cast<const Solver*>(
        static_cast<Solver*>(solver.get())->get_solver().get());

    ASSERT_NE(inner_solver, nullptr);
    ASSERT_EQ(inner_solver->get_size(), gko::dim<2>(3, 3));
    ASSERT_EQ(inner_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(Ir, CanSetGeneratedInnerSolverInFactory)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> ir_solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(this->mtx);

    auto ir_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_generated_solver(ir_solver)
            .on(this->exec);
    auto solver = ir_factory->generate(this->mtx);
    auto inner_solver = solver->get_solver();

    ASSERT_NE(inner_solver.get(), nullptr);
    ASSERT_EQ(inner_solver.get(), ir_solver.get());
}


TYPED_TEST(Ir, CanSetCriteriaAgain)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<gko::stop::CriterionFactory> init_crit =
        gko::stop::Iteration::build().with_max_iters(3u).on(this->exec);
    auto ir_factory = Solver::build().with_criteria(init_crit).on(this->exec);

    ASSERT_EQ((ir_factory->get_parameters().criteria).back(), init_crit);

    auto solver = ir_factory->generate(this->mtx);
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


TYPED_TEST(Ir, ThrowsOnWrongInnerSolverInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> wrong_sized_mtx =
        Mtx::create(this->exec, gko::dim<2>{2, 2});
    std::shared_ptr<Solver> ir_solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(wrong_sized_mtx);

    auto ir_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_generated_solver(ir_solver)
            .on(this->exec);

    ASSERT_THROW(ir_factory->generate(this->mtx), gko::DimensionMismatch);
}


TYPED_TEST(Ir, CanSetInnerSolver)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> ir_solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(this->mtx);

    auto ir_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec);
    auto solver = ir_factory->generate(this->mtx);
    solver->set_solver(ir_solver);
    auto inner_solver = solver->get_solver();

    ASSERT_NE(inner_solver.get(), nullptr);
    ASSERT_EQ(inner_solver.get(), ir_solver.get());
}


TYPED_TEST(Ir, CanSetApplyWithInitialGuessMode)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    using initial_guess_mode = gko::solver::initial_guess_mode;
    for (auto guess : {initial_guess_mode::provided, initial_guess_mode::rhs,
                       initial_guess_mode::zero}) {
        auto ir_factory =
            Solver::build()
                .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
                .with_default_initial_guess(guess)
                .on(this->exec);
        auto solver = ir_factory->generate(this->mtx);

        ASSERT_EQ(solver->apply_uses_initial_guess(),
                  guess == gko::solver::initial_guess_mode::provided);
    }
}


TYPED_TEST(Ir, ThrowOnWrongInnerSolverSet)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> wrong_sized_mtx =
        Mtx::create(this->exec, gko::dim<2>{2, 2});
    std::shared_ptr<Solver> ir_solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(wrong_sized_mtx);

    auto ir_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec);
    auto solver = ir_factory->generate(this->mtx);

    ASSERT_THROW(solver->set_solver(ir_solver), gko::DimensionMismatch);
}


TYPED_TEST(Ir, ThrowsOnRectangularMatrixInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> rectangular_mtx =
        Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->ir_factory->generate(rectangular_mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(Ir, DefaultRelaxationFactor)
{
    using value_type = typename TestFixture::value_type;
    const value_type relaxation_factor{0.5};

    auto richardson =
        gko::solver::Richardson<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(r<value_type>::value))
            .on(this->exec)
            ->generate(this->mtx);

    ASSERT_EQ(richardson->get_parameters().relaxation_factor, value_type{1});
}


TYPED_TEST(Ir, UseAsRichardson)
{
    using value_type = typename TestFixture::value_type;
    const value_type relaxation_factor{0.5};

    auto richardson =
        gko::solver::Richardson<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(r<value_type>::value))
            .with_relaxation_factor(relaxation_factor)
            .on(this->exec)
            ->generate(this->mtx);

    ASSERT_EQ(richardson->get_parameters().relaxation_factor, value_type{0.5});
}


TYPED_TEST(Ir, DefaultSmootherBuildWithSolver)
{
    using value_type = typename TestFixture::value_type;
    using Solver = typename TestFixture::Solver;
    auto solver = gko::as<Solver>(share(std::move(this->solver)));

    auto smoother_factory = gko::solver::build_smoother<value_type>(solver);
    auto criteria =
        std::dynamic_pointer_cast<const gko::stop::Iteration::Factory>(
            smoother_factory->get_parameters().criteria.at(0));

    ASSERT_EQ(smoother_factory->get_parameters().relaxation_factor,
              value_type{0.9});
    ASSERT_NE(criteria.get(), nullptr);
    ASSERT_EQ(criteria->get_parameters().max_iters, 1);
    ASSERT_EQ(smoother_factory->get_parameters().generated_solver.get(),
              solver.get());
}


TYPED_TEST(Ir, DefaultSmootherBuildWithFactory)
{
    using value_type = typename TestFixture::value_type;
    using Solver = typename TestFixture::Solver;
    auto factory = this->ir_factory;

    auto smoother_factory = gko::solver::build_smoother<value_type>(factory);
    auto criteria =
        std::dynamic_pointer_cast<const gko::stop::Iteration::Factory>(
            smoother_factory->get_parameters().criteria.at(0));

    ASSERT_EQ(smoother_factory->get_parameters().relaxation_factor,
              value_type{0.9});
    ASSERT_NE(criteria.get(), nullptr);
    ASSERT_EQ(criteria->get_parameters().max_iters, 1);
    ASSERT_EQ(smoother_factory->get_parameters().solver.get(), factory.get());
}


TYPED_TEST(Ir, SmootherBuildWithSolver)
{
    using value_type = typename TestFixture::value_type;
    using Solver = typename TestFixture::Solver;
    auto solver = gko::as<Solver>(gko::share(std::move(this->solver)));

    auto smoother_factory =
        gko::solver::build_smoother<value_type>(solver, 3, value_type{0.5});
    auto criteria =
        std::dynamic_pointer_cast<const gko::stop::Iteration::Factory>(
            smoother_factory->get_parameters().criteria.at(0));

    ASSERT_EQ(smoother_factory->get_parameters().relaxation_factor,
              value_type{0.5});
    ASSERT_NE(criteria.get(), nullptr);
    ASSERT_EQ(criteria->get_parameters().max_iters, 3);
    ASSERT_EQ(smoother_factory->get_parameters().generated_solver.get(),
              solver.get());
}


TYPED_TEST(Ir, SmootherBuildWithFactory)
{
    using value_type = typename TestFixture::value_type;
    using Solver = typename TestFixture::Solver;
    auto factory = this->ir_factory;

    auto smoother_factory =
        gko::solver::build_smoother<value_type>(factory, 3, value_type{0.5});
    auto criteria =
        std::dynamic_pointer_cast<const gko::stop::Iteration::Factory>(
            smoother_factory->get_parameters().criteria.at(0));

    ASSERT_EQ(smoother_factory->get_parameters().relaxation_factor,
              value_type{0.5});
    ASSERT_NE(criteria.get(), nullptr);
    ASSERT_EQ(criteria->get_parameters().max_iters, 3);
    ASSERT_EQ(smoother_factory->get_parameters().solver.get(), factory.get());
}


struct TestSummaryWriter : gko::log::ProfilerHook::SummaryWriter {
    void write(const std::vector<gko::log::ProfilerHook::summary_entry>& e,
               std::chrono::nanoseconds overhead) override
    {
        int matched = 0;
        for (const auto& data : e) {
            if (data.name == "residual_norm::residual_norm") {
                matched++;
                // Contains make_residual_norm 3 times: The last 4-th iteration
                // exits due to iteration limit.
                EXPECT_EQ(data.count, 3);
            }
        }
        // ensure matching once
        EXPECT_EQ(matched, 1);
    }
};


TYPED_TEST(Ir, RunResidualNormCheckCorrectTimes)
{
    using value_type = typename TestFixture::value_type;
    using Solver = typename TestFixture::Solver;
    using Mtx = typename TestFixture::Mtx;
    auto b = gko::initialize<Mtx>({2, -1.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);
    auto logger = gko::share(gko::log::ProfilerHook::create_summary(
        std::make_shared<gko::CpuTimer>(),
        std::make_unique<TestSummaryWriter>()));
    this->exec->add_logger(logger);

    // solver reaches the iteration limit
    this->solver->apply(b, x);

    // The assertions happen in the destructor of `logger`
}


TYPED_TEST(Ir, PassExplicitFactory)
{
    using Solver = typename TestFixture::Solver;
    auto stop_factory = gko::share(
        gko::stop::Iteration::build().with_max_iters(1u).on(this->exec));
    auto inner_solver_factory = gko::share(Solver::build().on(this->exec));

    auto factory = Solver::build()
                       .with_criteria(stop_factory)
                       .with_solver(inner_solver_factory)
                       .on(this->exec);

    ASSERT_EQ(factory->get_parameters().criteria.front(), stop_factory);
    ASSERT_EQ(factory->get_parameters().solver, inner_solver_factory);
}


}  // namespace
