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

#include <ginkgo/core/solver/multigrid.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/amgx_pgm.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/test/utils.hpp"


namespace {


/**
 * global_step is a global value to label each step of multigrid operator
 * advanced_apply: global_step *= real(alpha), push(global_step), global_step++
 * others:         push(global_step), global_step++
 * Need to initialize global_step before applying multigrid.
 */
int global_step = 0;


void assert_same_vector(std::vector<int> v1, std::vector<int> v2)
{
    ASSERT_EQ(v1.size(), v2.size());
    for (gko::size_type i = 0; i < v1.size(); i++) {
        ASSERT_EQ(v1.at(i), v2.at(i));
    }
}


template <typename ValueType>
class DummyLinOpWithFactory
    : public gko::EnableLinOp<DummyLinOpWithFactory<ValueType>> {
public:
    const std::vector<int> get_step() const { return step; }

    DummyLinOpWithFactory(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<DummyLinOpWithFactory>(exec)
    {}

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory){};
    GKO_ENABLE_LIN_OP_FACTORY(DummyLinOpWithFactory, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    DummyLinOpWithFactory(const Factory *factory,
                          std::shared_ptr<const gko::LinOp> op)
        : gko::EnableLinOp<DummyLinOpWithFactory>(factory->get_executor(),
                                                  transpose(op->get_size())),
          parameters_{factory->get_parameters()},
          op_{op}
    {}

    std::shared_ptr<const gko::LinOp> op_;

protected:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        step.push_back(global_step);
        global_step++;
    }

    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override
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


class DummyRestrictProlongOpWithFactory
    : public gko::multigrid::EnableRestrictProlong<
          DummyRestrictProlongOpWithFactory> {
public:
    const std::vector<int> get_rstr_step() const { return rstr_step; }

    const std::vector<int> get_prlg_step() const { return prlg_step; }

    DummyRestrictProlongOpWithFactory(std::shared_ptr<const gko::Executor> exec)
        : gko::multigrid::EnableRestrictProlong<
              DummyRestrictProlongOpWithFactory>(exec)
    {}

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory){};
    GKO_ENABLE_RESTRICT_PROLONG_FACTORY(DummyRestrictProlongOpWithFactory,
                                        parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    DummyRestrictProlongOpWithFactory(const Factory *factory,
                                      std::shared_ptr<const gko::LinOp> op)
        : gko::multigrid::EnableRestrictProlong<
              DummyRestrictProlongOpWithFactory>(factory->get_executor()),
          parameters_{factory->get_parameters()},
          op_{op}
    {
        this->set_coarse_fine(op_, op_->get_size()[0]);
    }

    std::shared_ptr<const gko::LinOp> op_;

protected:
    void restrict_apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        rstr_step.push_back(global_step);
        global_step++;
    }

    void prolong_applyadd_impl(const gko::LinOp *b,
                               gko::LinOp *x) const override
    {
        prlg_step.push_back(global_step);
        global_step++;
    }

    mutable std::vector<int> rstr_step;
    mutable std::vector<int> prlg_step;
};


template <typename T>
class Multigrid : public ::testing::Test {
protected:
    using value_type = T;
    using Csr = gko::matrix::Csr<value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Multigrid<value_type>;
    using Coarse = gko::multigrid::AmgxPgm<value_type>;
    using Smoother = gko::preconditioner::Jacobi<value_type>;
    using CoarsestSolver = gko::solver::Cg<value_type>;
    using DummyRPFactory = DummyRestrictProlongOpWithFactory;
    using DummyFactory = DummyLinOpWithFactory<value_type>;
    Multigrid()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Csr>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          coarse_factory(Coarse::build()
                             .with_max_iterations(2u)
                             .with_max_unassigned_percentage(0.1)
                             .on(exec)),
          smoother_factory(
              gko::give(Smoother::build().with_max_block_size(1u).on(exec))),
          coarsest_factory(
              CoarsestSolver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(4u).on(exec),
                      gko::stop::Time::build()
                          .with_time_limit(std::chrono::seconds(6))
                          .on(exec),
                      gko::stop::ResidualNormReduction<value_type>::build()
                          .with_reduction_factor(r<value_type>::value)
                          .on(exec))
                  .on(exec)),
          rp_factory(DummyRPFactory::build().on(exec)),
          lo_factory(DummyFactory::build().on(exec)),
          b(gko::initialize<Mtx>({I<value_type>({1, 0}), I<value_type>({0, 2}),
                                  I<value_type>({-1, -2})},
                                 exec)),
          x(gko::initialize<Mtx>(
              {I<value_type>({-1, 0}), I<value_type>({-2, 1}),
               I<value_type>({1, 1})},
              exec))
    {}

    static void assert_same_step(const gko::LinOp *lo, std::vector<int> v2)
    {
        auto v1 = dynamic_cast<const DummyFactory *>(lo)->get_step();
        assert_same_vector(v1, v2);
    }

    static void assert_same_step(const gko::multigrid::RestrictProlong *rp,
                                 std::vector<int> rstr, std::vector<int> prlg)
    {
        auto dummy = dynamic_cast<const DummyRPFactory *>(rp);
        auto v = dummy->get_rstr_step();
        assert_same_vector(v, rstr);
        v = dummy->get_prlg_step();
        assert_same_vector(v, prlg);
    }

    std::unique_ptr<typename Solver::Factory> get_multigrid_factory(
        gko::solver::multigrid_cycle cycle)
    {
        return std::move(
            Solver::build()
                .with_pre_smoother(smoother_factory)
                .with_coarsest_solver(coarsest_factory)
                .with_max_levels(2u)
                .with_post_uses_pre(true)
                .with_mid_case(gko::solver::multigrid_mid_uses::pre)
                .with_rstr_prlg(coarse_factory)
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(4u).on(exec),
                    gko::stop::Time::build()
                        .with_time_limit(std::chrono::seconds(6))
                        .on(exec),
                    gko::stop::ResidualNormReduction<value_type>::build()
                        .with_reduction_factor(r<value_type>::value)
                        .on(exec))
                .with_cycle(cycle)
                .on(exec));
    }


    std::unique_ptr<typename Solver::Factory> get_factory_individual(
        gko::solver::multigrid_cycle cycle)
    {
        return std::move(
            Solver::build()
                .with_max_levels(2u)
                .with_rstr_prlg(this->rp_factory, this->rp_factory)
                .with_pre_smoother(nullptr, this->lo_factory)
                .with_mid_smoother(this->lo_factory, nullptr)
                .with_post_smoother(this->lo_factory, nullptr)
                .with_pre_relaxation(gko::Array<value_type>(this->exec, {1, 2}))
                .with_mid_relaxation(gko::Array<value_type>(this->exec, {3, 4}))
                .with_post_relaxation(
                    gko::Array<value_type>(this->exec, {5, 6}))
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(1u).on(
                        this->exec))
                .with_cycle(cycle)
                .on(this->exec));
    }


    std::unique_ptr<typename Solver::Factory> get_factory_same(
        gko::solver::multigrid_cycle cycle)
    {
        return std::move(
            Solver::build()
                .with_max_levels(2u)
                .with_rstr_prlg(this->rp_factory, this->rp_factory)
                .with_pre_smoother(nullptr, this->lo_factory)
                .with_coarsest_solver(this->lo_factory)
                .with_pre_relaxation(gko::Array<value_type>(this->exec, {2}))
                .with_post_uses_pre(true)
                .with_mid_case(gko::solver::multigrid_mid_uses::pre)
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(1u).on(
                        this->exec))
                .with_cycle(cycle)
                .on(this->exec));
    }


    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Csr> mtx;
    std::shared_ptr<typename Coarse::Factory> coarse_factory;
    std::shared_ptr<typename Smoother::Factory> smoother_factory;
    std::shared_ptr<typename CoarsestSolver::Factory> coarsest_factory;
    std::shared_ptr<DummyRPFactory::Factory> rp_factory;
    std::shared_ptr<typename DummyFactory::Factory> lo_factory;
    std::shared_ptr<Mtx> b;
    std::shared_ptr<Mtx> x;
};

TYPED_TEST_CASE(Multigrid, gko::test::ValueTypes);


TYPED_TEST(Multigrid, VCycleIndividual)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_individual(gko::solver::multigrid_cycle::v)
                      ->generate(this->mtx);
    auto rstr_prlg = solver->get_rstr_prlg_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();

    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre: pass, 2
    // mid: 3, pass (not used)
    // post: 5, pass
    // coarset_solver: Identity
    //  +               -
    //    \           /
    //      +       -
    //        \   /
    //          v
    //  +               30
    //    0           5
    //      2       -
    //        3   4
    //          v
    global_step = 0;
    solver->apply(gko::lend(this->b), gko::lend(this->x));

    this->assert_same_step(rstr_prlg.at(0).get(), {0}, {5});
    this->assert_same_step(rstr_prlg.at(1).get(), {3}, {4});
    this->assert_same_step(pre_smoother.at(1).get(), {2});
    this->assert_same_step(post_smoother.at(0).get(), {30});
}


TYPED_TEST(Multigrid, VCycleSame)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_same(gko::solver::multigrid_cycle::v)
                      ->generate(this->mtx);
    auto rstr_prlg = solver->get_rstr_prlg_list();
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
    //    0           13
    //      2       12
    //        3   5
    //          4
    global_step = 0;
    solver->apply(gko::lend(this->b), gko::lend(this->x));

    this->assert_same_step(rstr_prlg.at(0).get(), {0}, {13});
    this->assert_same_step(rstr_prlg.at(1).get(), {3}, {5});
    // all uses pre_smoother
    this->assert_same_step(pre_smoother.at(1).get(), {2, 12});
    this->assert_same_step(coarsest_solver.get(), {4});
}


TYPED_TEST(Multigrid, WCycleIndividual)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_individual(gko::solver::multigrid_cycle::w)
                      ->generate(this->mtx);
    auto rstr_prlg = solver->get_rstr_prlg_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();
    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre: pass, 2
    // mid: 3, pass
    // post: 5, pass
    // coarset_solver: Identity
    //  +                       *                       -
    //    \                   /   \                   /
    //      +       *       -       +       *       -
    //        \   /   \   /           \   /   \   /
    //          v       v               v       v
    //  +                       24                      290
    //    0                   7   25                  57
    //      2       *       -       52      *       -
    //        3   4   5   6           53  54  55  56
    //          v       v               v       v
    global_step = 0;
    solver->apply(gko::lend(this->b), gko::lend(this->x));

    this->assert_same_step(rstr_prlg.at(0).get(), {0, 25}, {7, 57});
    this->assert_same_step(rstr_prlg.at(1).get(), {3, 5, 53, 55},
                           {4, 6, 54, 56});
    this->assert_same_step(pre_smoother.at(1).get(), {2, 52});
    this->assert_same_step(mid_smoother.at(0).get(), {24});
    this->assert_same_step(post_smoother.at(0).get(), {290});
}


TYPED_TEST(Multigrid, WCycleSame)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_same(gko::solver::multigrid_cycle::w)
                      ->generate(this->mtx);
    auto rstr_prlg = solver->get_rstr_prlg_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();
    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre, mid, post: pass, 2
    // coarset_solver: dummy_operator
    //  +                       *                       -
    //    \                   /   \                   /
    //      +       *       -       +       *       -
    //        \   /   \   /           \   /   \   /
    //          v       v               v       v
    //  +                       *                       -
    //    0                   33  34                  305
    //      2       12      32      70      148     304
    //        3   5   13  15          71  73  149 151
    //          4       14              72      150
    global_step = 0;
    solver->apply(gko::lend(this->b), gko::lend(this->x));

    this->assert_same_step(rstr_prlg.at(0).get(), {0, 34}, {33, 305});
    this->assert_same_step(rstr_prlg.at(1).get(), {3, 13, 71, 149},
                           {5, 15, 73, 151});
    // all uses pre_smoother
    this->assert_same_step(pre_smoother.at(1).get(), {2, 12, 32, 70, 148, 304});
    this->assert_same_step(coarsest_solver.get(), {4, 14, 72, 150});
}


TYPED_TEST(Multigrid, FCycleIndividual)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_individual(gko::solver::multigrid_cycle::f)
                      ->generate(this->mtx);
    auto rstr_prlg = solver->get_rstr_prlg_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto mid_smoother = solver->get_mid_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre: pass, 2
    // mid: 3, pass
    // post: 5, pass
    // coarset_solver: Identity
    //  +                       *               -
    //    \                   /   \           /
    //      +       *       -       +       -
    //        \   /   \   /           \   /
    //          v       v               v
    //  +                       24              280
    //    0                   7   25          55
    //      2       *       -       52      -
    //        3   4   5   6           53  54
    //          v       v               v
    global_step = 0;
    solver->apply(gko::lend(this->b), gko::lend(this->x));

    this->assert_same_step(rstr_prlg.at(0).get(), {0, 25}, {7, 55});
    this->assert_same_step(rstr_prlg.at(1).get(), {3, 5, 53}, {4, 6, 54});
    // all uses pre_smoother
    this->assert_same_step(pre_smoother.at(1).get(), {2, 52});
    this->assert_same_step(mid_smoother.at(0).get(), {24});
    this->assert_same_step(post_smoother.at(0).get(), {280});
}


TYPED_TEST(Multigrid, FCycleSame)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_same(gko::solver::multigrid_cycle::f)
                      ->generate(this->mtx);
    auto rstr_prlg = solver->get_rstr_prlg_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();
    // \: restrict,  /: prolong,   v: coarsest_solve
    // +: presmooth, *: midsmooth, -: postsmooth
    // alpha setting
    // pre, mid, post: pass, 2
    // coarset_solver: dummy_operator
    //  +                       *               -
    //    \                   /   \           /
    //      +       *       -       +       -
    //        \   /   \   /           \   /
    //          v       v               v
    //  +                       *               -
    //    0                   33  34          149
    //      2       12      32      70      148
    //        3   5   13  15          71  73
    //          4       14              72
    global_step = 0;
    solver->apply(gko::lend(this->b), gko::lend(this->x));

    this->assert_same_step(rstr_prlg.at(0).get(), {0, 34}, {33, 149});
    this->assert_same_step(rstr_prlg.at(1).get(), {3, 13, 71}, {5, 15, 73});
    // all uses pre_smoother
    this->assert_same_step(pre_smoother.at(1).get(), {2, 12, 32, 70, 148});
    this->assert_same_step(coarsest_solver.get(), {4, 14, 72});
}


TYPED_TEST(Multigrid, CanChangeCycle)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto solver = this->get_factory_same(gko::solver::multigrid_cycle::v)
                      ->generate(this->mtx);
    auto original = solver->get_cycle();
    auto rstr_prlg = solver->get_rstr_prlg_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    // change v cycle to f cycle
    global_step = 0;
    solver->set_cycle(gko::solver::multigrid_cycle::f);
    solver->apply(gko::lend(this->b), gko::lend(this->x));

    ASSERT_EQ(original, gko::solver::multigrid_cycle::v);
    ASSERT_EQ(solver->get_cycle(), gko::solver::multigrid_cycle::f);
    this->assert_same_step(rstr_prlg.at(0).get(), {0, 34}, {33, 149});
    this->assert_same_step(rstr_prlg.at(1).get(), {3, 13, 71}, {5, 15, 73});
    // all uses pre_smoother
    this->assert_same_step(pre_smoother.at(1).get(), {2, 12, 32, 70, 148});
    this->assert_same_step(coarsest_solver.get(), {4, 14, 72});
}


TYPED_TEST(Multigrid, SolvesStencilSystemByVCycle)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;

    auto multigrid_factory =
        this->get_multigrid_factory(gko::solver::multigrid_cycle::v);
    auto solver = multigrid_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value);
}


TYPED_TEST(Multigrid, SolvesStencilSystemByWCycle)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;

    auto multigrid_factory =
        this->get_multigrid_factory(gko::solver::multigrid_cycle::w);
    auto solver = multigrid_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value);
}


TYPED_TEST(Multigrid, SolvesStencilSystemByFCycle)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;

    auto multigrid_factory =
        this->get_multigrid_factory(gko::solver::multigrid_cycle::f);
    auto solver = multigrid_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value);
}


}  // namespace
