// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/multigrid.hpp>


#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/direct.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"


namespace {


class DummyLinOp : public gko::EnableLinOp<DummyLinOp>,
                   public gko::EnableCreateMethod<DummyLinOp> {
public:
    DummyLinOp(std::shared_ptr<const gko::Executor> exec,
               gko::dim<2> size = gko::dim<2>{})
        : EnableLinOp<DummyLinOp>(exec, size)
    {}

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {}

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {}
};


template <typename ValueType, bool uses_initial_guess = true>
class DummyLinOpWithFactory
    : public gko::EnableLinOp<
          DummyLinOpWithFactory<ValueType, uses_initial_guess>>,
      public gko::multigrid::EnableMultigridLevel<ValueType> {
public:
    using Mtx = gko::matrix::Dense<ValueType>;

    DummyLinOpWithFactory(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<DummyLinOpWithFactory>(exec)
    {}

    bool apply_uses_initial_guess() const override
    {
        return uses_initial_guess;
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        int GKO_FACTORY_PARAMETER_SCALAR(value, 5);
    };
    GKO_ENABLE_LIN_OP_FACTORY(DummyLinOpWithFactory, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    DummyLinOpWithFactory(const Factory* factory,
                          std::shared_ptr<const gko::LinOp> op)
        : gko::EnableLinOp<DummyLinOpWithFactory>(factory->get_executor(),
                                                  op->get_size()),
          gko::multigrid::EnableMultigridLevel<ValueType>(op),
          parameters_{factory->get_parameters()},
          op_{op},
          n_{op->get_size()[0]}
    {
        this->set_multigrid_level(
            std::make_shared<DummyLinOp>(this->get_executor(),
                                         gko::dim<2>{n_, n_ - 1}),
            gko::share(gko::test::generate_random_dense_matrix<ValueType>(
                n_ - 1, n_ - 1,
                std::uniform_real_distribution<gko::remove_complex<ValueType>>(
                    0, 1),
                std::default_random_engine{}, factory->get_executor())),
            std::make_shared<DummyLinOp>(this->get_executor(),
                                         gko::dim<2>{n_ - 1, n_}));
    }

    std::shared_ptr<const gko::LinOp> op_;
    gko::size_type n_;

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {}

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {}
};


template <typename T>
class Multigrid : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Multigrid;
    using DummyRPFactory = DummyLinOpWithFactory<value_type>;
    using DummyFactory = DummyLinOpWithFactory<value_type>;

    Multigrid()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>({{2, -1.0, 0.0, 0.0},
                                    {-1.0, 2, -1.0, 0.0},
                                    {0.0, -1.0, 2, -1.0},
                                    {0.0, 0.0, -1.0, 2}},
                                   exec)),
          rp_factory(DummyRPFactory::build().on(exec)),
          lo_factory(DummyFactory::build().on(exec)),
          rp_factory2(DummyRPFactory::build().with_value(2).on(exec)),
          lo_factory2(DummyFactory::build().with_value(2).on(exec)),
          criterion(gko::stop::Iteration::build().with_max_iters(1u).on(exec))
    {
        multigrid_factory =
            Solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(3u),
                    gko::stop::ResidualNorm<value_type>::build()
                        .with_baseline(gko::stop::mode::initial_resnorm)
                        .with_reduction_factor(gko::remove_complex<T>{1e-6}))
                .with_max_levels(2u)
                .with_coarsest_solver(lo_factory)
                .with_pre_smoother(lo_factory)
                .with_mid_smoother(lo_factory)
                .with_post_smoother(lo_factory)
                .with_post_uses_pre(false)
                .with_mid_case(
                    gko::solver::multigrid::mid_smooth_type::standalone)
                .with_mg_level(rp_factory)
                .with_min_coarse_rows(2u)
                .with_default_initial_guess(
                    gko::solver::initial_guess_mode::provided)
                .on(exec);
        solver = multigrid_factory->generate(mtx);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> multigrid_factory;
    std::unique_ptr<gko::LinOp> solver;
    std::shared_ptr<typename DummyRPFactory::Factory> rp_factory;
    std::shared_ptr<typename DummyRPFactory::Factory> rp_factory2;
    std::shared_ptr<typename DummyFactory::Factory> lo_factory;
    std::shared_ptr<typename DummyFactory::Factory> lo_factory2;
    std::shared_ptr<const gko::stop::CriterionFactory> criterion;

    static int get_value(
        gko::ptr_param<const gko::multigrid::MultigridLevel> rp)
    {
        return dynamic_cast<const DummyRPFactory*>(rp.get())
            ->get_parameters()
            .value;
    }

    static int get_value(gko::ptr_param<const gko::LinOp> lo)
    {
        return dynamic_cast<const DummyFactory*>(lo.get())
            ->get_parameters()
            .value;
    }
};

TYPED_TEST_SUITE(Multigrid, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Multigrid, MultigridFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->multigrid_factory->get_executor(), this->exec);
}


TYPED_TEST(Multigrid, MultigridFactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(4, 4));
    auto multigrid_solver = static_cast<Solver*>(this->solver.get());
    ASSERT_NE(multigrid_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(multigrid_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(Multigrid, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->multigrid_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver);

    ASSERT_EQ(copy->get_size(), gko::dim<2>(4, 4));
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_mtx), this->mtx, 0.0);
}


TYPED_TEST(Multigrid, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->multigrid_factory->generate(Mtx::create(this->exec));

    copy->move_from(this->solver);

    ASSERT_EQ(copy->get_size(), gko::dim<2>(4, 4));
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_mtx), this->mtx, 0.0);
}


TYPED_TEST(Multigrid, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto clone = this->solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(4, 4));
    auto clone_mtx = static_cast<Solver*>(clone.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(clone_mtx), this->mtx, 0.0);
}


TYPED_TEST(Multigrid, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;

    this->solver->clear();

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
    auto solver_mtx =
        static_cast<Solver*>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
    auto mg_level =
        static_cast<Solver*>(this->solver.get())->get_mg_level_list();
    ASSERT_EQ(mg_level.size(), 0);
}


TYPED_TEST(Multigrid, ApplyUsesInitialGuessReturnsTrue)
{
    ASSERT_TRUE(this->solver->apply_uses_initial_guess());
}


TYPED_TEST(Multigrid, ApplyUsesInitialGuessReturnsFalseWhenZeroGuess)
{
    using Solver = typename TestFixture::Solver;
    auto multigrid_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_max_levels(2u)
            .with_coarsest_solver(this->lo_factory)
            .with_pre_smoother(this->lo_factory)
            .with_mg_level(this->rp_factory)
            .with_min_coarse_rows(2u)
            .with_default_initial_guess(gko::solver::initial_guess_mode::zero)
            .on(this->exec);

    auto solver = multigrid_factory->generate(this->mtx);

    ASSERT_FALSE(solver->apply_uses_initial_guess());
}


TYPED_TEST(Multigrid, CanChangeCycle)
{
    using Solver = typename TestFixture::Solver;
    auto solver = static_cast<Solver*>(this->solver.get());
    auto original = solver->get_cycle();

    solver->set_cycle(gko::solver::multigrid::cycle::w);

    ASSERT_EQ(original, gko::solver::multigrid::cycle::v);
    ASSERT_EQ(solver->get_cycle(), gko::solver::multigrid::cycle::w);
}


TYPED_TEST(Multigrid, EachLevelAreDistinct)
{
    using Solver = typename TestFixture::Solver;
    auto solver = static_cast<Solver*>(this->solver.get());
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    ASSERT_EQ(mg_level.size(), 2);
    ASSERT_NE(mg_level.at(0), mg_level.at(1));
    ASSERT_EQ(this->get_value(mg_level.at(0)), 5);
    ASSERT_EQ(this->get_value(mg_level.at(1)), 5);
    ASSERT_EQ(pre_smoother.size(), 2);
    ASSERT_NE(pre_smoother.at(0), pre_smoother.at(1));
    ASSERT_EQ(this->get_value(pre_smoother.at(0)), 5);
    ASSERT_EQ(this->get_value(pre_smoother.at(1)), 5);
    ASSERT_EQ(mid_smoother.size(), 2);
    ASSERT_NE(mid_smoother.at(0), mid_smoother.at(1));
    ASSERT_EQ(this->get_value(mid_smoother.at(0)), 5);
    ASSERT_EQ(this->get_value(mid_smoother.at(1)), 5);
    ASSERT_EQ(post_smoother.size(), 2);
    ASSERT_NE(post_smoother.at(0), post_smoother.at(1));
    ASSERT_EQ(this->get_value(post_smoother.at(0)), 5);
    ASSERT_EQ(this->get_value(post_smoother.at(1)), 5);
    ASSERT_NE(coarsest_solver, nullptr);
}


TYPED_TEST(Multigrid, DefaultBehavior)
{
    using value_type = typename TestFixture::value_type;
    using Solver = typename TestFixture::Solver;
    auto solver = Solver::build()
                      .with_max_levels(1u)
                      .with_mg_level(this->rp_factory)
                      .with_criteria(this->criterion)
                      .with_min_coarse_rows(2u)
                      .on(this->exec)
                      ->generate(this->mtx);
    auto coarsest_solver = solver->get_coarsest_solver();
    auto direct = dynamic_cast<
        const gko::experimental::solver::Direct<value_type, gko::int32>*>(
        coarsest_solver.get());
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    auto pre_ir = dynamic_cast<const gko::solver::Ir<value_type>*>(
        pre_smoother.at(0).get());
    auto pre_jac = dynamic_cast<const gko::preconditioner::Jacobi<value_type>*>(
        pre_ir->get_solver().get());
    auto post_ir = dynamic_cast<const gko::solver::Ir<value_type>*>(
        post_smoother.at(0).get());
    auto post_jac =
        dynamic_cast<const gko::preconditioner::Jacobi<value_type>*>(
            post_ir->get_solver().get());
    auto mid_ir = dynamic_cast<const gko::solver::Ir<value_type>*>(
        mid_smoother.at(0).get());
    auto mid_jac = dynamic_cast<const gko::preconditioner::Jacobi<value_type>*>(
        mid_ir->get_solver().get());

    ASSERT_NE(pre_ir, nullptr);
    ASSERT_NE(pre_jac, nullptr);
    ASSERT_NE(post_ir, nullptr);
    ASSERT_NE(post_jac, nullptr);
    ASSERT_NE(mid_ir, nullptr);
    ASSERT_NE(mid_jac, nullptr);
    ASSERT_NE(direct, nullptr);
}


TYPED_TEST(Multigrid, DefaultBehaviorGivenNullptrs)
{
    using Solver = typename TestFixture::Solver;
    auto solver = Solver::build()
                      .with_max_levels(1u)
                      .with_min_coarse_rows(2u)
                      .with_mg_level(this->rp_factory)
                      .with_pre_smoother(nullptr)
                      .with_mid_smoother(nullptr)
                      .with_post_smoother(nullptr)
                      .with_criteria(this->criterion)
                      .on(this->exec)
                      ->generate(this->mtx);
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();

    ASSERT_EQ(pre_smoother.at(0), nullptr);
    ASSERT_EQ(mid_smoother.at(0), nullptr);
    ASSERT_EQ(post_smoother.at(0), nullptr);
}


TYPED_TEST(Multigrid, ThrowWhenNullMgLevel)
{
    using Solver = typename TestFixture::Solver;
    auto factory = Solver::build()
                       .with_max_levels(1u)
                       .with_min_coarse_rows(2u)
                       .with_criteria(this->criterion)
                       .on(this->exec);

    ASSERT_THROW(factory->generate(this->mtx), gko::NotSupported);
}


TYPED_TEST(Multigrid, ThrowWhenMgLevelContainsNullptr)
{
    using Solver = typename TestFixture::Solver;
    auto factory = Solver::build()
                       .with_max_levels(1u)
                       .with_min_coarse_rows(2u)
                       .with_criteria(this->criterion)
                       .with_mg_level(this->rp_factory, nullptr)
                       .on(this->exec);

    ASSERT_THROW(factory->generate(this->mtx), gko::NotSupported);
}


TYPED_TEST(Multigrid, ThrowWhenEmptyMgLevelList)
{
    using Solver = typename TestFixture::Solver;
    auto factory =
        Solver::build()
            .with_max_levels(1u)
            .with_min_coarse_rows(2u)
            .with_mg_level(
                std::vector<std::shared_ptr<const gko::LinOpFactory>>{})
            .with_criteria(this->criterion)
            .on(this->exec);

    ASSERT_THROW(factory->generate(this->mtx), gko::NotSupported);
}


TYPED_TEST(Multigrid, ThrowWhenInconsistentSizeOfPreSmoother)
{
    using Solver = typename TestFixture::Solver;
    auto factory =
        Solver::build()
            .with_max_levels(1u)
            .with_min_coarse_rows(2u)
            .with_criteria(this->criterion)
            .with_mg_level(this->rp_factory, this->rp_factory, this->rp_factory)
            .with_pre_smoother(this->lo_factory, this->lo_factory)
            .on(this->exec);

    ASSERT_THROW(factory->generate(this->mtx), gko::NotSupported);
}


TYPED_TEST(Multigrid, ThrowWhenInconsistentSizeOfMidSmoother)
{
    using Solver = typename TestFixture::Solver;
    auto factory =
        Solver::build()
            .with_max_levels(1u)
            .with_min_coarse_rows(2u)
            .with_criteria(this->criterion)
            .with_mg_level(this->rp_factory, this->rp_factory, this->rp_factory)
            .with_mid_case(gko::solver::multigrid::mid_smooth_type::standalone)
            .with_mid_smoother(this->lo_factory, this->lo_factory)
            .on(this->exec);

    ASSERT_THROW(factory->generate(this->mtx), gko::NotSupported);
}


TYPED_TEST(Multigrid, ThrowWhenInconsistentSizeOfPostSmoother)
{
    using Solver = typename TestFixture::Solver;
    auto factory =
        Solver::build()
            .with_max_levels(1u)
            .with_min_coarse_rows(2u)
            .with_criteria(this->criterion)
            .with_mg_level(this->rp_factory, this->rp_factory, this->rp_factory)
            .with_post_uses_pre(false)
            .with_post_smoother(this->lo_factory, this->lo_factory2)
            .on(this->exec);

    ASSERT_THROW(factory->generate(this->mtx), gko::NotSupported);
}


TYPED_TEST(Multigrid, TwoMgLevel)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;

    auto solver =
        Solver::build()
            .with_max_levels(2u)
            .with_min_coarse_rows(2u)
            .with_mg_level(this->rp_factory, this->rp_factory2)
            .with_pre_smoother(this->lo_factory, this->lo_factory2)
            .with_mid_smoother(this->lo_factory, this->lo_factory2)
            .with_post_smoother(this->lo_factory2, this->lo_factory)
            .with_post_uses_pre(false)
            .with_mid_case(gko::solver::multigrid::mid_smooth_type::standalone)
            .with_criteria(this->criterion)
            .on(this->exec)
            ->generate(this->mtx);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();
    auto direct = dynamic_cast<
        const gko::experimental::solver::Direct<value_type, gko::int32>*>(
        coarsest_solver.get());

    ASSERT_EQ(mg_level.size(), 2);
    ASSERT_NE(mg_level.at(0), mg_level.at(1));
    ASSERT_EQ(this->get_value(mg_level.at(0)), 5);
    ASSERT_EQ(this->get_value(mg_level.at(1)), 2);
    ASSERT_EQ(this->get_value(pre_smoother.at(0)), 5);
    ASSERT_EQ(this->get_value(pre_smoother.at(1)), 2);
    ASSERT_EQ(this->get_value(mid_smoother.at(0)), 5);
    ASSERT_EQ(this->get_value(mid_smoother.at(1)), 2);
    ASSERT_EQ(this->get_value(post_smoother.at(0)), 2);
    ASSERT_EQ(this->get_value(post_smoother.at(1)), 5);
    // coarset_solver is direct LU by default
    ASSERT_NE(direct, nullptr);
}


TYPED_TEST(Multigrid, TwoMgLevelWithOneSmootherRelaxation)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;

    auto solver =
        Solver::build()
            .with_max_levels(2u)
            .with_min_coarse_rows(2u)
            .with_mg_level(this->rp_factory, this->rp_factory2)
            .with_pre_smoother(this->lo_factory)
            .with_mid_smoother(this->lo_factory)
            .with_post_smoother(this->lo_factory2)
            .with_post_uses_pre(false)
            .with_mid_case(gko::solver::multigrid::mid_smooth_type::standalone)
            .with_criteria(this->criterion)
            .on(this->exec)
            ->generate(this->mtx);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();
    auto direct = dynamic_cast<
        const gko::experimental::solver::Direct<value_type, gko::int32>*>(
        coarsest_solver.get());

    ASSERT_EQ(mg_level.size(), 2);
    ASSERT_NE(mg_level.at(0), mg_level.at(1));
    ASSERT_EQ(this->get_value(mg_level.at(0)), 5);
    ASSERT_EQ(this->get_value(mg_level.at(1)), 2);
    ASSERT_EQ(this->get_value(pre_smoother.at(0)), 5);
    ASSERT_EQ(this->get_value(pre_smoother.at(1)), 5);
    ASSERT_EQ(this->get_value(mid_smoother.at(0)), 5);
    ASSERT_EQ(this->get_value(mid_smoother.at(1)), 5);
    ASSERT_EQ(this->get_value(post_smoother.at(0)), 2);
    ASSERT_EQ(this->get_value(post_smoother.at(1)), 2);
    ASSERT_NE(direct, nullptr);
}


TYPED_TEST(Multigrid, CustomSelectorWithSameSize)
{
    using Solver = typename TestFixture::Solver;
    auto selector = [](const gko::size_type level, const gko::LinOp* matrix) {
        return (level == 1) ? 0 : 1;
    };
    auto solver =
        Solver::build()
            .with_max_levels(2u)
            .with_min_coarse_rows(2u)
            .with_mg_level(this->rp_factory, this->rp_factory2)
            .with_pre_smoother(this->lo_factory, this->lo_factory2)
            .with_mid_smoother(this->lo_factory2, this->lo_factory)
            .with_post_smoother(this->lo_factory, this->lo_factory2)
            .with_post_uses_pre(false)
            .with_mid_case(gko::solver::multigrid::mid_smooth_type::standalone)
            .with_level_selector(selector)
            .with_criteria(this->criterion)
            .on(this->exec)
            ->generate(this->mtx);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();

    ASSERT_EQ(mg_level.size(), 2);
    ASSERT_EQ(this->get_value(mg_level.at(0)), 2);
    ASSERT_EQ(this->get_value(mg_level.at(1)), 5);
    // pre_smoother use the same index as mg_level
    ASSERT_EQ(pre_smoother.size(), 2);
    ASSERT_EQ(this->get_value(pre_smoother.at(0)), 2);
    ASSERT_EQ(this->get_value(pre_smoother.at(1)), 5);
    // pre_smoother use the same index as mg_level
    ASSERT_EQ(mid_smoother.size(), 2);
    ASSERT_EQ(this->get_value(mid_smoother.at(0)), 5);
    ASSERT_EQ(this->get_value(mid_smoother.at(1)), 2);
    // post_smoother has the same index as mg_level
    ASSERT_EQ(post_smoother.size(), 2);
    ASSERT_EQ(this->get_value(post_smoother.at(0)), 2);
    ASSERT_EQ(this->get_value(post_smoother.at(1)), 5);
}


TYPED_TEST(Multigrid, CustomSelectorWithOneSmootherRelaxation)
{
    using Solver = typename TestFixture::Solver;
    auto selector = [](const gko::size_type level, const gko::LinOp* matrix) {
        return (level == 1) ? 0 : 1;
    };
    auto solver =
        Solver::build()
            .with_max_levels(2u)
            .with_min_coarse_rows(2u)
            .with_mg_level(this->rp_factory, this->rp_factory2)
            .with_pre_smoother(this->lo_factory)
            .with_mid_smoother(this->lo_factory)
            .with_post_smoother(this->lo_factory2)
            .with_post_uses_pre(false)
            .with_mid_case(gko::solver::multigrid::mid_smooth_type::standalone)
            .with_level_selector(selector)
            .with_criteria(this->criterion)
            .on(this->exec)
            ->generate(this->mtx);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();

    ASSERT_EQ(mg_level.size(), 2);
    ASSERT_EQ(this->get_value(mg_level.at(0)), 2);
    ASSERT_EQ(this->get_value(mg_level.at(1)), 5);
    // pre_smoother always uses the same factory
    ASSERT_EQ(pre_smoother.size(), 2);
    ASSERT_EQ(this->get_value(pre_smoother.at(0)), 5);
    ASSERT_EQ(this->get_value(pre_smoother.at(1)), 5);
    // mid_smoother always uses the same factory
    ASSERT_EQ(mid_smoother.size(), 2);
    ASSERT_EQ(this->get_value(mid_smoother.at(0)), 5);
    ASSERT_EQ(this->get_value(mid_smoother.at(1)), 5);
    // post_smoother always uses the same factory
    ASSERT_EQ(post_smoother.size(), 2);
    ASSERT_EQ(this->get_value(post_smoother.at(0)), 2);
    ASSERT_EQ(this->get_value(post_smoother.at(1)), 2);
}


TYPED_TEST(Multigrid, CustomSelectorWithMix)
{
    using Solver = typename TestFixture::Solver;
    auto selector = [](const gko::size_type level, const gko::LinOp* matrix) {
        return (level == 1) ? 0 : 1;
    };
    auto solver =
        Solver::build()
            .with_max_levels(2u)
            .with_min_coarse_rows(2u)
            .with_mg_level(this->rp_factory, this->rp_factory2)
            .with_pre_smoother(this->lo_factory)
            .with_post_smoother(this->lo_factory2, this->lo_factory)
            .with_post_uses_pre(false)
            .with_mid_case(gko::solver::multigrid::mid_smooth_type::standalone)
            .with_level_selector(selector)
            .with_criteria(this->criterion)
            .on(this->exec)
            ->generate(this->mtx);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();

    ASSERT_EQ(mg_level.size(), 2);
    ASSERT_EQ(this->get_value(mg_level.at(0)), 2);
    ASSERT_EQ(this->get_value(mg_level.at(1)), 5);
    // pre_smoother always uses the same factory
    ASSERT_EQ(pre_smoother.size(), 2);
    ASSERT_EQ(this->get_value(pre_smoother.at(0)), 5);
    ASSERT_EQ(this->get_value(pre_smoother.at(1)), 5);
    // mid_smoother uses Jacobi by default
    ASSERT_EQ(mid_smoother.size(), 2);
    ASSERT_NE(mid_smoother.at(0), nullptr);
    ASSERT_NE(mid_smoother.at(1), nullptr);
    // post_smoother uses the same index as mg_level
    ASSERT_EQ(post_smoother.size(), 2);
    ASSERT_EQ(this->get_value(post_smoother.at(0)), 5);
    ASSERT_EQ(this->get_value(post_smoother.at(1)), 2);
}


TYPED_TEST(Multigrid, PostUsesPre)
{
    using Solver = typename TestFixture::Solver;
    // post setting should be ignored
    auto solver = Solver::build()
                      .with_max_levels(2u)
                      .with_min_coarse_rows(2u)
                      .with_mg_level(this->rp_factory, this->rp_factory2)
                      .with_pre_smoother(this->lo_factory, this->lo_factory2)
                      .with_post_smoother(this->lo_factory2, this->lo_factory)
                      .with_criteria(this->criterion)
                      .on(this->exec)
                      ->generate(this->mtx);
    auto pre_smoother = solver->get_pre_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();

    ASSERT_EQ(this->get_value(pre_smoother.at(0)), 5);
    ASSERT_EQ(this->get_value(pre_smoother.at(1)), 2);
    // post_smoother ignore the manual setting because the post_uses_pre = true
    // the elements are copied from pre_smoother, so the pointers are the same
    ASSERT_EQ(post_smoother.size(), 2);
    ASSERT_EQ(post_smoother.at(0).get(), pre_smoother.at(0).get());
    ASSERT_EQ(post_smoother.at(1).get(), pre_smoother.at(1).get());
}


TYPED_TEST(Multigrid, MidUsesPre)
{
    using Solver = typename TestFixture::Solver;
    // post setting should be ignored
    auto solver = Solver::build()
                      .with_max_levels(2u)
                      .with_min_coarse_rows(2u)
                      .with_mg_level(this->rp_factory, this->rp_factory2)
                      .with_pre_smoother(this->lo_factory, this->lo_factory2)
                      .with_mid_smoother(this->lo_factory2, this->lo_factory)
                      .with_mid_case(
                          gko::solver::multigrid::mid_smooth_type::pre_smoother)
                      .with_criteria(this->criterion)
                      .on(this->exec)
                      ->generate(this->mtx);
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();

    ASSERT_EQ(this->get_value(pre_smoother.at(0)), 5);
    ASSERT_EQ(this->get_value(pre_smoother.at(1)), 2);
    // mid is handled by the pre smoother of next level, so the mid_smoother is
    // empty mid_smoother ignores the manual setting because
    // multigrid::mid_smooth_type::pre_smoother
    ASSERT_EQ(mid_smoother.size(), 0);
}


TYPED_TEST(Multigrid, MidUsesPost)
{
    using Solver = typename TestFixture::Solver;
    // post setting should be ignored
    auto solver =
        Solver::build()
            .with_max_levels(2u)
            .with_min_coarse_rows(2u)
            .with_mg_level(this->rp_factory, this->rp_factory2)
            .with_post_smoother(this->lo_factory, this->lo_factory2)
            .with_mid_smoother(this->lo_factory2, this->lo_factory)
            .with_post_uses_pre(false)
            .with_mid_case(
                gko::solver::multigrid::mid_smooth_type::post_smoother)
            .with_criteria(this->criterion)
            .on(this->exec)
            ->generate(this->mtx);
    auto post_smoother = solver->get_post_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();

    ASSERT_EQ(this->get_value(post_smoother.at(0)), 5);
    ASSERT_EQ(this->get_value(post_smoother.at(1)), 2);
    // mid is handled by the post smoother of previous level, so the
    // mid_smoother is empty mid_smoother ignores the manual setting because
    // multigrid::mid_smooth_type::post_smoother
    ASSERT_EQ(mid_smoother.size(), 0);
}


TYPED_TEST(Multigrid, PostUsesPreAndMidUsesPre)
{
    using Solver = typename TestFixture::Solver;
    // post setting should be ignored
    auto solver = Solver::build()
                      .with_max_levels(2u)
                      .with_min_coarse_rows(2u)
                      .with_mg_level(this->rp_factory, this->rp_factory2)
                      .with_pre_smoother(this->lo_factory, this->lo_factory2)
                      .with_mid_smoother(this->lo_factory2, this->lo_factory)
                      .with_post_smoother(this->lo_factory2, this->lo_factory)
                      .with_mid_case(
                          gko::solver::multigrid::mid_smooth_type::pre_smoother)
                      .with_criteria(this->criterion)
                      .on(this->exec)
                      ->generate(this->mtx);
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();

    ASSERT_EQ(pre_smoother.size(), 2);
    ASSERT_EQ(this->get_value(pre_smoother.at(0)), 5);
    ASSERT_EQ(this->get_value(pre_smoother.at(1)), 2);
    // post uses pre
    ASSERT_EQ(post_smoother.size(), 2);
    ASSERT_EQ(post_smoother.at(0).get(), pre_smoother.at(0).get());
    ASSERT_EQ(post_smoother.at(1).get(), pre_smoother.at(1).get());
    // mid is handled by the pre smoother of next cycle, so the mid_smoother is
    // empty
    ASSERT_EQ(mid_smoother.size(), 0);
}


TYPED_TEST(Multigrid, PostUsesPreAndMidUsesPost)
{
    using Solver = typename TestFixture::Solver;
    // post setting should be ignored
    auto solver =
        Solver::build()
            .with_max_levels(2u)
            .with_min_coarse_rows(2u)
            .with_mg_level(this->rp_factory, this->rp_factory2)
            .with_pre_smoother(this->lo_factory, this->lo_factory2)
            .with_mid_smoother(this->lo_factory2, this->lo_factory)
            .with_post_smoother(this->lo_factory2, this->lo_factory)
            .with_mid_case(
                gko::solver::multigrid::mid_smooth_type::post_smoother)
            .with_criteria(this->criterion)
            .on(this->exec)
            ->generate(this->mtx);
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();

    ASSERT_EQ(pre_smoother.size(), 2);
    ASSERT_EQ(this->get_value(pre_smoother.at(0)), 5);
    ASSERT_EQ(this->get_value(pre_smoother.at(1)), 2);
    // post uses pre
    ASSERT_EQ(post_smoother.size(), 2);
    ASSERT_EQ(post_smoother.at(0).get(), pre_smoother.at(0).get());
    ASSERT_EQ(post_smoother.at(1).get(), pre_smoother.at(1).get());
    // mid is handled by the post smoother of previous cycle, so the
    // mid_smoother is empty
    ASSERT_EQ(mid_smoother.size(), 0);
}


TYPED_TEST(Multigrid, DefaultCoarsestSolverSelectorUsesTheFirstOne)
{
    using Solver = typename TestFixture::Solver;
    auto solver = Solver::build()
                      .with_max_levels(2u)
                      .with_min_coarse_rows(2u)
                      .with_mg_level(this->rp_factory)
                      .with_pre_smoother(this->lo_factory)
                      .with_criteria(this->criterion)
                      .with_coarsest_solver(this->lo_factory, this->lo_factory2)
                      .on(this->exec)
                      ->generate(this->mtx);
    auto coarsest_solver = solver->get_coarsest_solver();

    ASSERT_EQ(this->get_value(coarsest_solver), 5);
}


TYPED_TEST(Multigrid, CustomCoarsestSolverSelector)
{
    using Solver = typename TestFixture::Solver;
    auto selector = [](const gko::size_type level, const gko::LinOp* matrix) {
        return (level == 2) ? 1 : 0;
    };
    auto solver = Solver::build()
                      .with_max_levels(2u)
                      .with_min_coarse_rows(2u)
                      .with_mg_level(this->rp_factory)
                      .with_pre_smoother(this->lo_factory)
                      .with_criteria(this->criterion)
                      .with_coarsest_solver(this->lo_factory, this->lo_factory2)
                      .with_solver_selector(selector)
                      .on(this->exec)
                      ->generate(this->mtx);
    auto coarsest_solver = solver->get_coarsest_solver();

    ASSERT_EQ(this->get_value(coarsest_solver), 2);
}


TYPED_TEST(Multigrid, DeferredFactoryParameter)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;

    auto solver = Solver::build()
                      .with_mg_level(DummyRPFactory::build())
                      .with_pre_smoother(DummyFactory::build())
                      .with_mid_smoother(DummyFactory::build())
                      .with_post_smoother(DummyFactory::build())
                      .with_criteria(gko::stop::Iteration::build())
                      .with_coarsest_solver(DummyFactory::build())
                      .on(this->exec);

    GKO_ASSERT_DYNAMIC_TYPE(solver->get_parameters().mg_level[0],
                            typename DummyRPFactory::Factory);
    GKO_ASSERT_DYNAMIC_TYPE(solver->get_parameters().pre_smoother[0],
                            typename DummyFactory::Factory);
    GKO_ASSERT_DYNAMIC_TYPE(solver->get_parameters().mid_smoother[0],
                            typename DummyFactory::Factory);
    GKO_ASSERT_DYNAMIC_TYPE(solver->get_parameters().post_smoother[0],
                            typename DummyFactory::Factory);
    GKO_ASSERT_DYNAMIC_TYPE(solver->get_parameters().coarsest_solver[0],
                            typename DummyFactory::Factory);
}


}  // namespace
