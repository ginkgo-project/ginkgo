// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/multigrid.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/test/utils.hpp"


namespace {


/**
 * global_step is a global value to label each step of multigrid operator
 * advanced_apply: global_step *= real(alpha), push(global_step), global_step++
 * others:         push(global_step), global_step++
 * Need to initialize global_step before applying multigrid.
 * DummyLinOp only increases global_step but it does not store global_step.
 */
int global_step = 0;


void assert_same_vector(std::vector<int> v1, std::vector<int> v2)
{
    ASSERT_EQ(v1.size(), v2.size());
    for (gko::size_type i = 0; i < v1.size(); i++) {
        ASSERT_EQ(v1.at(i), v2.at(i));
    }
}


class DummyLinOp : public gko::EnableLinOp<DummyLinOp>,
                   public gko::EnableCreateMethod<DummyLinOp> {
public:
    DummyLinOp(std::shared_ptr<const gko::Executor> exec,
               gko::dim<2> size = gko::dim<2>{})
        : EnableLinOp<DummyLinOp>(exec, size)
    {}

    bool apply_uses_initial_guess() const override { return true; }

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        global_step++;
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {}
};

class DummyRestrictOp : public gko::EnableLinOp<DummyRestrictOp>,
                        public gko::EnableCreateMethod<DummyRestrictOp> {
public:
    const std::vector<int> get_rstr_step() const { return rstr_step; }

    DummyRestrictOp(std::shared_ptr<const gko::Executor> exec,
                    gko::dim<2> size = gko::dim<2>{})
        : EnableLinOp<DummyRestrictOp>(exec, size)
    {}

    bool apply_uses_initial_guess() const override { return true; }

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        rstr_step.push_back(global_step);
        global_step++;
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {}

    mutable std::vector<int> rstr_step;
};


class DummyProlongOp : public gko::EnableLinOp<DummyProlongOp>,
                       public gko::EnableCreateMethod<DummyProlongOp> {
public:
    const std::vector<int> get_prlg_step() const { return prlg_step; }

    DummyProlongOp(std::shared_ptr<const gko::Executor> exec,
                   gko::dim<2> size = gko::dim<2>{})
        : EnableLinOp<DummyProlongOp>(exec, size)
    {}

    bool apply_uses_initial_guess() const override { return true; }

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {}

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {
        prlg_step.push_back(global_step);
        global_step++;
    }

    mutable std::vector<int> prlg_step;
};


template <typename ValueType>
class DummyLinOpWithFactory
    : public gko::EnableLinOp<DummyLinOpWithFactory<ValueType>> {
public:
    const std::vector<int> get_step() const { return step; }

    bool apply_uses_initial_guess() const override { return true; }

    DummyLinOpWithFactory(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<DummyLinOpWithFactory>(exec)
    {}

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory){};
    GKO_ENABLE_LIN_OP_FACTORY(DummyLinOpWithFactory, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    DummyLinOpWithFactory(const Factory* factory,
                          std::shared_ptr<const gko::LinOp> op)
        : gko::EnableLinOp<DummyLinOpWithFactory>(factory->get_executor(),
                                                  transpose(op->get_size())),
          parameters_{factory->get_parameters()},
          op_{op}
    {}

    std::shared_ptr<const gko::LinOp> op_;

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        step.push_back(global_step);
        global_step++;
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {
        auto alpha_value =
            gko::as<gko::matrix::Dense<ValueType>>(alpha)->at(0, 0);
        gko::remove_complex<ValueType> scale = std::real(alpha_value);
        global_step *= static_cast<int>(scale);
        step.push_back(global_step);
        global_step++;
    }

    mutable std::vector<int> step;
};

template <typename ValueType>
class DummyMultigridLevelWithFactory
    : public gko::EnableLinOp<DummyMultigridLevelWithFactory<ValueType>>,
      public gko::multigrid::EnableMultigridLevel<ValueType> {
public:
    const std::vector<int> get_rstr_step() const
    {
        return restrict_->get_rstr_step();
    }

    const std::vector<int> get_prlg_step() const
    {
        return prolong_->get_prlg_step();
    }

    DummyMultigridLevelWithFactory(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<DummyMultigridLevelWithFactory>(exec)
    {}


    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory){};
    GKO_ENABLE_LIN_OP_FACTORY(DummyMultigridLevelWithFactory, parameters,
                              Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    DummyMultigridLevelWithFactory(const Factory* factory,
                                   std::shared_ptr<const gko::LinOp> op)
        : gko::EnableLinOp<DummyMultigridLevelWithFactory>(
              factory->get_executor(), op->get_size()),
          gko::multigrid::EnableMultigridLevel<ValueType>(op),
          parameters_{factory->get_parameters()},
          op_{op}
    {
        auto exec = this->get_executor();
        gko::size_type n = op_->get_size()[0] - 1;
        coarse_ = DummyLinOp::create(exec, gko::dim<2>{n});
        restrict_ = DummyRestrictOp::create(exec, gko::dim<2>{n, n + 1});
        prolong_ = DummyProlongOp::create(exec, gko::dim<2>{n + 1, n});
        this->set_multigrid_level(prolong_, coarse_, restrict_);
    }

    std::shared_ptr<const gko::LinOp> op_;
    std::shared_ptr<const gko::LinOp> coarse_;
    std::shared_ptr<const DummyRestrictOp> restrict_;
    std::shared_ptr<const DummyProlongOp> prolong_;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {}

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {}
};


template <typename ValueIndexType>
class Multigrid : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Csr = gko::matrix::Csr<value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Multigrid;
    using Coarse = gko::multigrid::Pgm<value_type>;
    using CoarseNext = gko::multigrid::Pgm<gko::next_precision<value_type>>;
    using Smoother = gko::solver::Ir<value_type>;
    using InnerSolver = gko::preconditioner::Jacobi<value_type>;
    using CoarsestSolver = gko::solver::Cg<value_type>;
    using CoarsestNextSolver = gko::solver::Cg<gko::next_precision<value_type>>;
    using DummyRPFactory = DummyMultigridLevelWithFactory<value_type>;
    using DummyFactory = DummyLinOpWithFactory<value_type>;
    Multigrid()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Csr>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          mtx2(gko::initialize<Csr>({{2, -1.0, 0.0, 0.0, 0.0, 0.0},
                                     {-1.0, 2, -1.0, 0.0, 0.0, 0.0},
                                     {0.0, -1.0, 2, -1.0, 0.0, 0.0},
                                     {0.0, 0.0, -1.0, 2, -1.0, 0.0},
                                     {0.0, 0.0, 0.0, -1.0, 2, -1.0},
                                     {0.0, 0.0, 0.0, 0.0, -1.0, 2}},
                                    exec)),
          coarse_factory(Coarse::build()
                             .with_max_iterations(2u)
                             .with_max_unassigned_ratio(0.1)
                             .on(exec)),
          coarsenext_factory(CoarseNext::build()
                                 .with_max_iterations(2u)
                                 .with_max_unassigned_ratio(0.1)
                                 .on(exec)),
          smoother_factory(gko::give(
              Smoother::build()
                  .with_solver(InnerSolver::build().with_max_block_size(1u))
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(1u))
                  .on(exec))),
          coarsest_factory(
              CoarsestSolver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(4u),
                      gko::stop::Time::build().with_time_limit(
                          std::chrono::seconds(6)),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_baseline(gko::stop::mode::initial_resnorm)
                          .with_reduction_factor(r<value_type>::value))
                  .on(exec)),
          coarsestnext_factory(
              CoarsestNextSolver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(4u),
                      gko::stop::Time::build().with_time_limit(
                          std::chrono::seconds(6)))
                  .on(exec)),
          rp_factory(DummyRPFactory::build().on(exec)),
          lo_factory(DummyFactory::build().on(exec)),
          b(gko::initialize<Mtx>({I<value_type>({1, 0}), I<value_type>({0, 2}),
                                  I<value_type>({-1, -2})},
                                 exec)),
          x(gko::initialize<Mtx>(
              {I<value_type>({-1, 0}), I<value_type>({-2, 1}),
               I<value_type>({1, 1})},
              exec)),
          b2(gko::initialize<Mtx>({-1.0, 3.0, 0.0, -3.0, 3.0, 1.0}, exec)),
          x2(gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, exec))
    {}

    static void assert_same_step(gko::ptr_param<const gko::LinOp> lo,
                                 std::vector<int> v2)
    {
        auto v1 = dynamic_cast<const DummyFactory*>(lo.get())->get_step();
        assert_same_vector(v1, v2);
    }

    static void assert_same_step(
        gko::ptr_param<const gko::multigrid::MultigridLevel> rp,
        std::vector<int> rstr, std::vector<int> prlg)
    {
        auto dummy = dynamic_cast<const DummyRPFactory*>(rp.get());
        auto v = dummy->get_rstr_step();
        assert_same_vector(v, rstr);
        v = dummy->get_prlg_step();
        assert_same_vector(v, prlg);
    }

    std::unique_ptr<typename Solver::Factory> get_multigrid_factory(
        gko::solver::multigrid::cycle cycle)
    {
        return std::move(
            Solver::build()
                .with_pre_smoother(smoother_factory)
                .with_coarsest_solver(coarsest_factory)
                .with_max_levels(2u)
                .with_post_uses_pre(true)
                .with_mid_case(gko::solver::multigrid::mid_smooth_type::both)
                .with_mg_level(coarse_factory)
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(4u),
                    gko::stop::Time::build().with_time_limit(
                        std::chrono::seconds(6)),
                    gko::stop::ResidualNorm<value_type>::build()
                        .with_baseline(gko::stop::mode::initial_resnorm)
                        .with_reduction_factor(r<value_type>::value))
                .with_cycle(cycle)
                .with_min_coarse_rows(1u)
                .on(exec));
    }

    std::unique_ptr<typename Solver::Factory> get_mixed_multigrid_factory(
        gko::solver::multigrid::cycle cycle)
    {
        return std::move(
            Solver::build()
                .with_pre_smoother(smoother_factory)
                .with_coarsest_solver(coarsestnext_factory)
                .with_max_levels(2u)
                .with_post_uses_pre(true)
                .with_mid_case(gko::solver::multigrid::mid_smooth_type::both)
                .with_mg_level(coarse_factory, coarsenext_factory)
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(200u),
                    gko::stop::Time::build().with_time_limit(
                        std::chrono::seconds(100)),
                    gko::stop::ResidualNorm<value_type>::build()
                        .with_baseline(gko::stop::mode::initial_resnorm)
                        .with_reduction_factor(r<value_type>::value))
                .with_cycle(cycle)
                .with_min_coarse_rows(1u)
                .on(exec));
    }


    std::unique_ptr<typename Solver::Factory> get_factory_individual(
        gko::solver::multigrid::cycle cycle, gko::size_type max_levels = 2u,
        gko::solver::multigrid::mid_smooth_type mid_case =
            gko::solver::multigrid::mid_smooth_type::both)
    {
        return std::move(
            Solver::build()
                .with_max_levels(max_levels)
                .with_mg_level(this->rp_factory, this->rp_factory,
                               this->rp_factory)
                .with_pre_smoother(nullptr, this->lo_factory, this->lo_factory)
                .with_mid_smoother(nullptr, nullptr, this->lo_factory)
                .with_post_smoother(this->lo_factory, nullptr, this->lo_factory)
                .with_coarsest_solver(
                    gko::matrix::IdentityFactory<value_type>::create(exec))
                .with_post_uses_pre(false)
                .with_mid_case(mid_case)
                .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                .with_cycle(cycle)
                .with_min_coarse_rows(1u)
                .on(this->exec));
    }

    std::unique_ptr<typename Solver::Factory> get_factory_same(
        gko::solver::multigrid::cycle cycle, gko::size_type max_levels = 2u,
        gko::solver::multigrid::mid_smooth_type mid_case =
            gko::solver::multigrid::mid_smooth_type::both)
    {
        return std::move(
            Solver::build()
                .with_max_levels(max_levels)
                .with_mg_level(this->rp_factory, this->rp_factory,
                               this->rp_factory)
                .with_pre_smoother(nullptr, this->lo_factory, this->lo_factory)
                .with_coarsest_solver(this->lo_factory)
                .with_post_uses_pre(true)
                .with_mid_case(mid_case)
                .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                .with_cycle(cycle)
                .with_min_coarse_rows(1u)
                .on(this->exec));
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Csr> mtx;
    std::shared_ptr<Csr> mtx2;
    std::shared_ptr<typename Coarse::Factory> coarse_factory;
    std::shared_ptr<typename CoarseNext::Factory> coarsenext_factory;
    std::shared_ptr<typename Smoother::Factory> smoother_factory;
    std::shared_ptr<typename CoarsestSolver::Factory> coarsest_factory;
    std::shared_ptr<typename CoarsestNextSolver::Factory> coarsestnext_factory;
    std::shared_ptr<typename DummyRPFactory::Factory> rp_factory;
    std::shared_ptr<typename DummyFactory::Factory> lo_factory;
    std::shared_ptr<Mtx> b;
    std::shared_ptr<Mtx> x;
    std::shared_ptr<Mtx> b2;
    std::shared_ptr<Mtx> x2;
};

TYPED_TEST_SUITE(Multigrid, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(Multigrid, VCycleIndividual)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_individual(gko::solver::multigrid::cycle::v)
                      ->generate(this->mtx);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();

    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre: pass, 2
    // mid: pass, pass (not used)
    // post: 5, pass
    // coarset_solver: Identity
    //  +               -
    //    \           /
    //      +       -
    //        \   /
    //          v
    //  +               5
    //    0           4
    //      1       -
    //        2   3
    //          v
    global_step = 0;
    solver->apply(this->b, this->x);

    this->assert_same_step(mg_level.at(0), {0}, {4});
    this->assert_same_step(mg_level.at(1), {2}, {3});
    this->assert_same_step(pre_smoother.at(1), {1});
    this->assert_same_step(post_smoother.at(0), {5});
}


TYPED_TEST(Multigrid, VCycleSame)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_same(gko::solver::multigrid::cycle::v)
                      ->generate(this->mtx);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre, mid, post: pass, 2
    // coarset_solver: dummy_operator
    //  +               -
    //    \           /
    //      +       -
    //        \   /
    //          v
    //  +               -
    //    0           6
    //      1       5
    //        2   4
    //          3
    global_step = 0;
    solver->apply(this->b, this->x);

    this->assert_same_step(mg_level.at(0), {0}, {6});
    this->assert_same_step(mg_level.at(1), {2}, {4});
    // all uses pre_smoother
    this->assert_same_step(pre_smoother.at(1), {1, 5});
    this->assert_same_step(coarsest_solver, {3});
}


TYPED_TEST(Multigrid, WCycleIndividual)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver =
        this->get_factory_individual(gko::solver::multigrid::cycle::w, 3u)
            ->generate(this->mtx2);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre: pass, 2, 2
    // mid: pass, pass, 3
    // post: 5, pass, 5
    // coarset_solver: Identity
    // +                                                             -
    //   \                                                         /
    //     +                         - +                         -
    //       \                     /     \                     /
    //         +       - +       -         +       - +       -
    //           \   /     \   /             \   /     \   /
    //             v         v                 v         v
    // +                                                             24
    //   0                                                         23
    //     1                         - 12                        -
    //       2                     11    13                    22
    //         3       6 7       10        14      1718      21
    //           4   5     8   9             15  16    19  20
    //             v         v                 v         v
    global_step = 0;
    solver->apply(this->b2, this->x2);

    this->assert_same_step(mg_level.at(0), {0}, {23});
    this->assert_same_step(mg_level.at(1), {2, 13}, {11, 22});
    this->assert_same_step(mg_level.at(2), {4, 8, 15, 19}, {5, 9, 16, 20});
    this->assert_same_step(pre_smoother.at(1), {1, 12});
    this->assert_same_step(pre_smoother.at(2), {3, 7, 14, 18});
    this->assert_same_step(post_smoother.at(0), {24});
    this->assert_same_step(post_smoother.at(2), {6, 10, 17, 21});
}


TYPED_TEST(Multigrid, WCycleIndividualMidUsePost)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver =
        this->get_factory_individual(
                gko::solver::multigrid::cycle::w, 3u,
                gko::solver::multigrid::mid_smooth_type::post_smoother)
            ->generate(this->mtx2);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre: pass, 2, 2
    // mid: pass, pass, 3
    // post: 5, pass, 5
    // coarset_solver: Identity
    // +                                                       -
    //   \                                                   /
    //     +                       -                       -
    //       \                   /   \                   /
    //         +       -       -       +       -       -
    //           \   /   \   /           \   /   \   /
    //             v       v               v       v
    // +                                                       21
    //   0                                                   20
    //     1                       -                       -
    //       2                   10  11                  19
    //         3       6       9       12      15      18
    //           4   5   7   8           13  14  16  17
    //             v       v               v       v
    global_step = 0;
    solver->apply(this->b2, this->x2);

    this->assert_same_step(mg_level.at(0), {0}, {20});
    this->assert_same_step(mg_level.at(1), {2, 11}, {10, 19});
    this->assert_same_step(mg_level.at(2), {4, 7, 13, 16}, {5, 8, 14, 17});
    this->assert_same_step(pre_smoother.at(1), {1});
    this->assert_same_step(pre_smoother.at(2), {3, 12});
    this->assert_same_step(post_smoother.at(0), {21});
    this->assert_same_step(post_smoother.at(2), {6, 9, 15, 18});
}


TYPED_TEST(Multigrid, WCycleIndividualMidUsePre)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_individual(
                          gko::solver::multigrid::cycle::w, 3u,
                          gko::solver::multigrid::mid_smooth_type::pre_smoother)
                      ->generate(this->mtx2);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre: pass, 2, 2
    // mid: pass, pass, 3
    // post: 5, pass, 5
    // coarset_solver: Identity
    // +                                                       -
    //   \                                                   /
    //     +                       +                       -
    //       \                   /   \                   /
    //         +       +       -       +       +       -
    //           \   /   \   /           \   /   \   /
    //             v       v               v       v
    // +                                                       22
    //   0                                                   21
    //     1                       11                      -
    //       2                   10  12                  20
    //         3       6       9       13      16      19
    //           4   5   7   8           14  15  17  18
    //             v       v               v       v
    global_step = 0;
    solver->apply(this->b2, this->x2);

    this->assert_same_step(mg_level.at(0), {0}, {21});
    this->assert_same_step(mg_level.at(1), {2, 12}, {10, 20});
    this->assert_same_step(mg_level.at(2), {4, 7, 14, 17}, {5, 8, 15, 18});
    this->assert_same_step(pre_smoother.at(1), {1, 11});
    this->assert_same_step(pre_smoother.at(2), {3, 6, 13, 16});
    this->assert_same_step(post_smoother.at(0), {22});
    this->assert_same_step(post_smoother.at(2), {9, 19});
}


TYPED_TEST(Multigrid, WCycleIndividualMidUseStandalone)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_individual(
                          gko::solver::multigrid::cycle::w, 3u,
                          gko::solver::multigrid::mid_smooth_type::standalone)
                      ->generate(this->mtx2);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre: pass, 2, 2
    // mid: pass, pass, 3
    // post: 5, pass, 5
    // coarset_solver: Identity
    // +                                                       -
    //   \                                                   /
    //     +                       *                       -
    //       \                   /   \                   /
    //         +       *       -       +       *       -
    //           \   /   \   /           \   /   \   /
    //             v       v               v       v
    // +                                                       21
    //   0                                                   20
    //     1                       *                       -
    //       2                   10  11                  19
    //         3       6       9       12      15      18
    //           4   5   7   8           13  14  16  17
    //             v       v               v       v
    global_step = 0;
    solver->apply(this->b2, this->x2);

    this->assert_same_step(mg_level.at(0), {0}, {20});
    this->assert_same_step(mg_level.at(1), {2, 11}, {10, 19});
    this->assert_same_step(mg_level.at(2), {4, 7, 13, 16}, {5, 8, 14, 17});
    this->assert_same_step(pre_smoother.at(1), {1});
    this->assert_same_step(pre_smoother.at(2), {3, 12});
    this->assert_same_step(post_smoother.at(0), {21});
    this->assert_same_step(post_smoother.at(2), {9, 18});
    this->assert_same_step(mid_smoother.at(2), {6, 15});
}


TYPED_TEST(Multigrid, WCycleSame)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_same(gko::solver::multigrid::cycle::w, 3u)
                      ->generate(this->mtx2);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre, mid, post: pass, 2, 2
    // coarset_solver: dummy_operator
    // +                                                             -
    //   \                                                         /
    //     +                         - +                         -
    //       \                     /     \                     /
    //         +       - +       -         +       - +       -
    //           \   /     \   /             \   /     \   /
    //             v         v                 v         v
    // +                                                             -
    //   0                                                         29
    //     1                         1415                        28
    //       2                     13    16                    27
    //         3       7 8       12        17      2122      26
    //           4   6     9   11            18  20    23  25
    //             5         10                19        24
    global_step = 0;
    solver->apply(this->b2, this->x2);

    this->assert_same_step(mg_level.at(0), {0}, {29});
    this->assert_same_step(mg_level.at(1), {2, 16}, {13, 27});
    this->assert_same_step(mg_level.at(2), {4, 9, 18, 23}, {6, 11, 20, 25});
    // all uses pre_smoother
    this->assert_same_step(pre_smoother.at(1), {1, 14, 15, 28});
    this->assert_same_step(pre_smoother.at(2), {3, 7, 8, 12, 17, 21, 22, 26});
    this->assert_same_step(coarsest_solver, {5, 10, 19, 24});
}


TYPED_TEST(Multigrid, WCycleSameMidUsePost)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver =
        this->get_factory_same(
                gko::solver::multigrid::cycle::w, 3u,
                gko::solver::multigrid::mid_smooth_type::post_smoother)
            ->generate(this->mtx2);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre, mid, post: pass, 2, 2
    // coarset_solver: dummy_operator
    // +                                                       -
    //   \                                                   /
    //     +                       -                       -
    //       \                   /   \                   /
    //         +       -       -       +       -       -
    //           \   /   \   /           \   /   \   /
    //             v       v               v       v
    // +                                                       -
    //   0                                                   26
    //     1                       13                      25
    //       2                   12  14                  24
    //         3       7       11      15      19      23
    //           4   6   8   10          16  18  20  22
    //             5       9               17      21
    global_step = 0;
    solver->apply(this->b2, this->x2);

    this->assert_same_step(mg_level.at(0), {0}, {26});
    this->assert_same_step(mg_level.at(1), {2, 14}, {12, 24});
    this->assert_same_step(mg_level.at(2), {4, 8, 16, 20}, {6, 10, 18, 22});
    // all use pre_smoother
    this->assert_same_step(pre_smoother.at(1), {1, 13, 25});
    this->assert_same_step(pre_smoother.at(2), {3, 7, 11, 15, 19, 23});
    this->assert_same_step(coarsest_solver, {5, 9, 17, 21});
}


TYPED_TEST(Multigrid, WCycleSameMidUsePre)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_same(
                          gko::solver::multigrid::cycle::w, 3u,
                          gko::solver::multigrid::mid_smooth_type::pre_smoother)
                      ->generate(this->mtx2);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre, mid, post: pass, 2, 2
    // coarset_solver: dummy_operator
    // +                                                       -
    //   \                                                   /
    //     +                       +                       -
    //       \                   /   \                   /
    //         +       +       -       +       +       -
    //           \   /   \   /           \   /   \   /
    //             v       v               v       v
    // +                                                       -
    //   0                                                   26
    //     1                       13                      25
    //       2                   12  14                  24
    //         3       7       11      15      19      23
    //           4   6   8   10          16  18  20  22
    //             5       9               17      21
    global_step = 0;
    solver->apply(this->b2, this->x2);

    this->assert_same_step(mg_level.at(0), {0}, {26});
    this->assert_same_step(mg_level.at(1), {2, 14}, {12, 24});
    this->assert_same_step(mg_level.at(2), {4, 8, 16, 20}, {6, 10, 18, 22});
    // all use pre_smoother
    this->assert_same_step(pre_smoother.at(1), {1, 13, 25});
    this->assert_same_step(pre_smoother.at(2), {3, 7, 11, 15, 19, 23});
    this->assert_same_step(coarsest_solver, {5, 9, 17, 21});
}


TYPED_TEST(Multigrid, FCycleIndividual)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver =
        this->get_factory_individual(gko::solver::multigrid::cycle::f, 3u)
            ->generate(this->mtx2);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();

    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre: pass, 2, 2
    // mid: pass, pass, 3
    // post: 5, pass, 5
    // coarset_solver: Identity
    // +                                                   -
    //   \                                               /
    //     +                         - +               -
    //       \                     /     \           /
    //         +       - +       -         +       -
    //           \   /     \   /             \   /
    //             v         v                 v
    // +                                                   20
    //   0                                               19
    //     1                         - 12              -
    //       2                     11    13          18
    //         3       6 7       10        14      17
    //           4   5     8   9             15  16
    //             v         v                 v
    global_step = 0;
    solver->apply(this->b2, this->x2);

    this->assert_same_step(mg_level.at(0), {0}, {19});
    this->assert_same_step(mg_level.at(1), {2, 13}, {11, 18});
    this->assert_same_step(mg_level.at(2), {4, 8, 15}, {5, 9, 16});
    this->assert_same_step(pre_smoother.at(1), {1, 12});
    this->assert_same_step(pre_smoother.at(2), {3, 7, 14});
    // this->assert_same_step(mid_smoother.at(0), {7});
    this->assert_same_step(post_smoother.at(0), {20});
    this->assert_same_step(post_smoother.at(2), {6, 10, 17});
}


TYPED_TEST(Multigrid, FCycleIndividualMidUsePost)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver =
        this->get_factory_individual(
                gko::solver::multigrid::cycle::f, 3u,
                gko::solver::multigrid::mid_smooth_type::post_smoother)
            ->generate(this->mtx2);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre: pass, 2, 2
    // mid: pass, pass, 3
    // post: 5, pass, 5
    // coarset_solver: Identity
    // +                                               -
    //   \                                           /
    //     +                       -               -
    //       \                   /   \           /
    //         +       -       -       +       -
    //           \   /   \   /           \   /
    //             v       v               v
    // +                                               18
    //   0                                           17
    //     1                       -               -
    //       2                   10  11          16
    //         3       6       9       12      15
    //           4   5   7   8           13  14
    //             v       v               v
    global_step = 0;
    solver->apply(this->b2, this->x2);

    this->assert_same_step(mg_level.at(0), {0}, {17});
    this->assert_same_step(mg_level.at(1), {2, 11}, {10, 16});
    this->assert_same_step(mg_level.at(2), {4, 7, 13}, {5, 8, 14});
    this->assert_same_step(pre_smoother.at(1), {1});
    this->assert_same_step(pre_smoother.at(2), {3, 12});
    this->assert_same_step(post_smoother.at(0), {18});
    this->assert_same_step(post_smoother.at(2), {6, 9, 15});
}


TYPED_TEST(Multigrid, FCycleIndividualMidUsePre)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_individual(
                          gko::solver::multigrid::cycle::f, 3u,
                          gko::solver::multigrid::mid_smooth_type::pre_smoother)
                      ->generate(this->mtx2);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre: pass, 2, 2
    // mid: pass, pass, 3
    // post: 5, pass, 5
    // coarset_solver: Identity
    // +                                               -
    //   \                                           /
    //     +                       +               -
    //       \                   /   \           /
    //         +       +       -       +       -
    //           \   /   \   /           \   /
    //             v       v               v
    // +                                               19
    //   0                                           18
    //     1                       11              -
    //       2                   10  12          17
    //         3       6       9       13      16
    //           4   5   7   8           14  15
    //             v       v               v
    global_step = 0;
    solver->apply(this->b2, this->x2);

    this->assert_same_step(mg_level.at(0), {0}, {18});
    this->assert_same_step(mg_level.at(1), {2, 12}, {10, 17});
    this->assert_same_step(mg_level.at(2), {4, 7, 14}, {5, 8, 15});
    this->assert_same_step(pre_smoother.at(1), {1, 11});
    this->assert_same_step(pre_smoother.at(2), {3, 6, 13});
    this->assert_same_step(post_smoother.at(0), {19});
    this->assert_same_step(post_smoother.at(2), {9, 16});
}


TYPED_TEST(Multigrid, FCycleIndividualMidUseStandalone)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_individual(
                          gko::solver::multigrid::cycle::f, 3u,
                          gko::solver::multigrid::mid_smooth_type::standalone)
                      ->generate(this->mtx2);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre: pass, 2, 2
    // mid: pass, pass, 3
    // post: 5, pass, 5
    // coarset_solver: Identity
    // +                                               -
    //   \                                           /
    //     +                       *               -
    //       \                   /   \           /
    //         +       *       -       +       -
    //           \   /   \   /           \   /
    //             v       v               v
    // +                                               18
    //   0                                           17
    //     1                       *               -
    //       2                   10  11          16
    //         3       6       9       12      15
    //           4   5   7   8           13  14
    //             v       v               v
    global_step = 0;
    solver->apply(this->b2, this->x2);

    this->assert_same_step(mg_level.at(0), {0}, {17});
    this->assert_same_step(mg_level.at(1), {2, 11}, {10, 16});
    this->assert_same_step(mg_level.at(2), {4, 7, 13}, {5, 8, 14});
    this->assert_same_step(pre_smoother.at(1), {1});
    this->assert_same_step(pre_smoother.at(2), {3, 12});
    this->assert_same_step(post_smoother.at(0), {18});
    this->assert_same_step(post_smoother.at(2), {9, 15});
    this->assert_same_step(mid_smoother.at(2), {6});
}


TYPED_TEST(Multigrid, FCycleSame)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_same(gko::solver::multigrid::cycle::f, 3u)
                      ->generate(this->mtx2);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre, mid, post: pass, 2, 2
    // coarset_solver: dummy_operator
    // +                                                   -
    //   \                                               /
    //     +                         - +               -
    //       \                     /     \           /
    //         +       - +       -         +       -
    //           \   /     \   /             \   /
    //             v         v                 v
    // +                                                   -
    //   0                                               24
    //     1                         1415              23
    //       2                     13    16          22
    //         3       7 8       12        17      21
    //           4   6     9   11            18  20
    //             5         10                19
    global_step = 0;
    solver->apply(this->b2, this->x2);

    this->assert_same_step(mg_level.at(0), {0}, {24});
    this->assert_same_step(mg_level.at(1), {2, 16}, {13, 22});
    this->assert_same_step(mg_level.at(2), {4, 9, 18}, {6, 11, 20});
    this->assert_same_step(pre_smoother.at(1), {1, 14, 15, 23});
    this->assert_same_step(pre_smoother.at(2), {3, 7, 8, 12, 17, 21});
    this->assert_same_step(coarsest_solver, {5, 10, 19});
}


TYPED_TEST(Multigrid, FCycleSameMidUsePost)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver =
        this->get_factory_same(
                gko::solver::multigrid::cycle::f, 3u,
                gko::solver::multigrid::mid_smooth_type::post_smoother)
            ->generate(this->mtx2);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre, mid, post: pass, 2, 2
    // coarset_solver: dummy_operator
    // +                                               -
    //   \                                           /
    //     +                       -               -
    //       \                   /   \           /
    //         +       -       -       +       -
    //           \   /   \   /           \   /
    //             v       v               v
    // +                                               -
    //   0                                           22
    //     1                       13              21
    //       2                   12  14          20
    //         3       7       11      15      19
    //           4   6   8   10          16  18
    //             5       9               17
    global_step = 0;
    solver->apply(this->b2, this->x2);

    this->assert_same_step(mg_level.at(0), {0}, {22});
    this->assert_same_step(mg_level.at(1), {2, 14}, {12, 20});
    this->assert_same_step(mg_level.at(2), {4, 8, 16}, {6, 10, 18});
    // all use pre_smoother
    this->assert_same_step(pre_smoother.at(1), {1, 13, 21});
    this->assert_same_step(pre_smoother.at(2), {3, 7, 11, 15, 19});
    this->assert_same_step(coarsest_solver, {5, 9, 17});
}


TYPED_TEST(Multigrid, FCycleSameMidUsePre)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_same(
                          gko::solver::multigrid::cycle::f, 3u,
                          gko::solver::multigrid::mid_smooth_type::pre_smoother)
                      ->generate(this->mtx2);
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre, mid, post: pass, 2, 2
    // coarset_solver: dummy_operator
    // +                                               -
    //   \                                           /
    //     +                       +               -
    //       \                   /   \           /
    //         +       +       -       +       -
    //           \   /   \   /           \   /
    //             v       v               v
    // +                                               -
    //   0                                           22
    //     1                       13              21
    //       2                   12  14          20
    //         3       7       11      15      19
    //           4   6   8   10          16  18
    //             5       9               17
    global_step = 0;
    solver->apply(this->b2, this->x2);

    this->assert_same_step(mg_level.at(0), {0}, {22});
    this->assert_same_step(mg_level.at(1), {2, 14}, {12, 20});
    this->assert_same_step(mg_level.at(2), {4, 8, 16}, {6, 10, 18});
    // all use pre_smoother
    this->assert_same_step(pre_smoother.at(1), {1, 13, 21});
    this->assert_same_step(pre_smoother.at(2), {3, 7, 11, 15, 19});
    this->assert_same_step(coarsest_solver, {5, 9, 17});
}


TYPED_TEST(Multigrid, CanChangeCycle)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_same(gko::solver::multigrid::cycle::v)
                      ->generate(this->mtx);
    auto original = solver->get_cycle();
    auto mg_level = solver->get_mg_level_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    // change v cycle to f cycle
    global_step = 0;
    solver->set_cycle(gko::solver::multigrid::cycle::f);
    solver->apply(this->b, this->x);

    ASSERT_EQ(original, gko::solver::multigrid::cycle::v);
    ASSERT_EQ(solver->get_cycle(), gko::solver::multigrid::cycle::f);
    this->assert_same_step(mg_level.at(0), {0}, {11});
    this->assert_same_step(mg_level.at(1), {2, 7}, {4, 9});
    // all uses pre_smoother
    this->assert_same_step(pre_smoother.at(1), {1, 5, 6, 10});
    this->assert_same_step(coarsest_solver, {3, 8});
}


TYPED_TEST(Multigrid, ZeroGuessIgnoresInput)
{
    using Solver = typename TestFixture::Solver;
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto common_part =
        Solver::build()
            .with_pre_smoother(this->smoother_factory)
            .with_coarsest_solver(this->coarsest_factory)
            .with_max_levels(2u)
            .with_mg_level(this->coarse_factory)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_min_coarse_rows(1u);
    auto normal_mg = common_part
                         .with_default_initial_guess(
                             gko::solver::initial_guess_mode::provided)
                         .on(this->exec)
                         ->generate(this->mtx);
    auto zeroguess_mg =
        common_part
            .with_default_initial_guess(gko::solver::initial_guess_mode::zero)
            .on(this->exec)
            ->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({-1.0, -3.0, -2.0}, this->exec);
    auto zero = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    // putting zero to normal multigrid is the same behavior as using zero guess
    // in multigrid
    normal_mg->apply(b, zero);
    zeroguess_mg->apply(b, x);

    GKO_ASSERT_MTX_NEAR(zero, x, r<value_type>::value);
}


TYPED_TEST(Multigrid, SolvesStencilSystemByVCycle)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto multigrid_factory =
        this->get_multigrid_factory(gko::solver::multigrid::cycle::v);
    auto solver = multigrid_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value);
}


TYPED_TEST(Multigrid, SolvesStencilSystemByWCycle)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto multigrid_factory =
        this->get_multigrid_factory(gko::solver::multigrid::cycle::w);
    auto solver = multigrid_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value);
}


TYPED_TEST(Multigrid, SolvesStencilSystemByFCycle)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto multigrid_factory =
        this->get_multigrid_factory(gko::solver::multigrid::cycle::f);
    auto solver = multigrid_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value);
}


}  // namespace
