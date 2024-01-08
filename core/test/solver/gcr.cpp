// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/gcr.hpp>


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
class Gcr : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Gcr<value_type>;
    using Big_solver = gko::solver::Gcr<double>;

    static constexpr gko::remove_complex<T> reduction_factor =
        gko::remove_complex<T>(1e-6);

    Gcr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{1.0, 2.0, 3.0}, {3.0, 2.0, -1.0}, {0.0, -1.0, 2}}, exec)),
          gcr_factory(Solver::build()
                          .with_criteria(
                              gko::stop::Iteration::build().with_max_iters(3u),
                              gko::stop::ResidualNorm<value_type>::build()
                                  .with_reduction_factor(reduction_factor))
                          .on(exec)),
          solver(gcr_factory->generate(mtx)),
          gcr_big_factory(
              Big_solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(128u),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(reduction_factor))
                  .on(exec)),
          big_solver(gcr_big_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> gcr_factory;
    std::unique_ptr<gko::LinOp> solver;
    std::unique_ptr<Big_solver::Factory> gcr_big_factory;
    std::unique_ptr<gko::LinOp> big_solver;

    static void assert_same_matrices(const Mtx* m1, const Mtx* m2)
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

template <typename T>
constexpr gko::remove_complex<T> Gcr<T>::reduction_factor;

TYPED_TEST_SUITE(Gcr, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Gcr, GcrFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->gcr_factory->get_executor(), this->exec);
}


TYPED_TEST(Gcr, GcrFactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    auto gcr_solver = static_cast<Solver*>(this->solver.get());

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(3, 3));
    ASSERT_NE(gcr_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(gcr_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(Gcr, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->gcr_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver.get());

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx*>(copy_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(Gcr, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->gcr_factory->generate(Mtx::create(this->exec));

    copy->move_from(this->solver);

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx*>(copy_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(Gcr, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto clone = this->solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    auto clone_mtx = static_cast<Solver*>(clone.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx*>(clone_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(Gcr, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;
    this->solver->clear();

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
    auto solver_mtx =
        static_cast<Solver*>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(Gcr, ApplyUsesInitialGuessReturnsTrue)
{
    ASSERT_TRUE(this->solver->apply_uses_initial_guess());
}


TYPED_TEST(Gcr, CanSetPreconditionerGenerator)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto gcr_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(TestFixture::reduction_factor))
            .with_preconditioner(Solver::build().with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u)))
            .on(this->exec);
    auto solver = gcr_factory->generate(this->mtx);
    auto precond = dynamic_cast<const gko::solver::Gcr<value_type>*>(
        static_cast<gko::solver::Gcr<value_type>*>(solver.get())
            ->get_preconditioner()
            .get());

    ASSERT_NE(precond, nullptr);
    ASSERT_EQ(precond->get_size(), gko::dim<2>(3, 3));
    ASSERT_EQ(precond->get_system_matrix(), this->mtx);
}


TYPED_TEST(Gcr, CanSetCriteriaAgain)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<gko::stop::CriterionFactory> init_crit =
        gko::stop::Iteration::build().with_max_iters(3u).on(this->exec);
    auto gcr_factory = Solver::build().with_criteria(init_crit).on(this->exec);

    ASSERT_EQ((gcr_factory->get_parameters().criteria).back(), init_crit);

    auto solver = gcr_factory->generate(this->mtx);
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


TYPED_TEST(Gcr, CanSetKrylovDim)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto gcr_factory =
        Solver::build()
            .with_krylov_dim(4u)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(4u),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(TestFixture::reduction_factor))
            .on(this->exec);
    auto solver = gcr_factory->generate(this->mtx);
    auto krylov_dim = solver->get_krylov_dim();

    ASSERT_EQ(krylov_dim, 4);
}


TYPED_TEST(Gcr, CanSetKrylovDimAgain)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<gko::stop::CriterionFactory> init_crit =
        gko::stop::Iteration::build().with_max_iters(3u).on(this->exec);
    auto gcr_factory =
        Solver::build().with_criteria(init_crit).with_krylov_dim(10u).on(
            this->exec);

    ASSERT_EQ(gcr_factory->get_parameters().krylov_dim, 10);

    auto solver = gcr_factory->generate(this->mtx);

    solver->set_krylov_dim(20);

    ASSERT_EQ(solver->get_krylov_dim(), 20);
}


TYPED_TEST(Gcr, CanSetPreconditionerInFactory)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> gcr_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(this->mtx);

    auto gcr_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_generated_preconditioner(gcr_precond)
            .on(this->exec);
    auto solver = gcr_factory->generate(this->mtx);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), gcr_precond.get());
}


TYPED_TEST(Gcr, ThrowsOnWrongPreconditionerInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> wrong_sized_mtx =
        Mtx::create(this->exec, gko::dim<2>{2, 2});
    std::shared_ptr<Solver> gcr_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(wrong_sized_mtx);

    auto gcr_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_generated_preconditioner(gcr_precond)
            .on(this->exec);

    ASSERT_THROW(gcr_factory->generate(this->mtx), gko::DimensionMismatch);
}


TYPED_TEST(Gcr, ThrowsOnRectangularMatrixInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> rectangular_mtx =
        Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->gcr_factory->generate(rectangular_mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(Gcr, CanSetPreconditioner)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> gcr_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec)
            ->generate(this->mtx);

    auto gcr_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(this->exec);
    auto solver = gcr_factory->generate(this->mtx);
    solver->set_preconditioner(gcr_precond);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), gcr_precond.get());
}


TYPED_TEST(Gcr, PassExplicitFactory)
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
