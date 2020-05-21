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


#include <typeinfo>


#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/restrict_prolong.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>


#include "core/test/utils.hpp"


namespace {


class DummyLinOpWithFactory : public gko::EnableLinOp<DummyLinOpWithFactory> {
public:
    DummyLinOpWithFactory(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<DummyLinOpWithFactory>(exec)
    {}

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        int GKO_FACTORY_PARAMETER(value, 5);
    };
    GKO_ENABLE_LIN_OP_FACTORY(DummyLinOpWithFactory, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    DummyLinOpWithFactory(const Factory *factory,
                          std::shared_ptr<const gko::LinOp> op)
        : gko::EnableLinOp<DummyLinOpWithFactory>(factory->get_executor()),
          parameters_{factory->get_parameters()},
          op_{op}
    {}

    std::shared_ptr<const gko::LinOp> op_;

protected:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override {}

    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override
    {}
};


class DummyRestrictProlongOpWithFactory
    : public gko::multigrid::EnableRestrictProlong<
          DummyRestrictProlongOpWithFactory> {
public:
    DummyRestrictProlongOpWithFactory(std::shared_ptr<const gko::Executor> exec)
        : gko::multigrid::EnableRestrictProlong<
              DummyRestrictProlongOpWithFactory>(exec)
    {}

    int get_value() const { return parameters_.value; }
    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        int GKO_FACTORY_PARAMETER(value, 5);
    };
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
    {}

    void prolong_applyadd_impl(const gko::LinOp *b,
                               gko::LinOp *x) const override
    {}
};


template <typename T>
class Multigrid : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Multigrid<value_type>;
    using DummyRPFactory = DummyRestrictProlongOpWithFactory;
    using DummyFactory = DummyLinOpWithFactory;

    Multigrid()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          rp_factory(DummyRPFactory::build().on(exec)),
          lo_factory(DummyFactory::build().on(exec)),
          rp_factory2(DummyRPFactory::build().with_value(2).on(exec))
    {
        multigrid_factory =
            Solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(3u).on(exec),
                    gko::stop::ResidualNormReduction<value_type>::build()
                        .with_reduction_factor(gko::remove_complex<T>{1e-6})
                        .on(exec))
                .with_max_levels(2u)
                .with_coarsest_solver(lo_factory)
                .with_pre_smoother(lo_factory)
                .with_post_smoother(lo_factory)
                .with_rstr_prlg(rp_factory)
                .on(exec);
        solver = multigrid_factory->generate(mtx);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> multigrid_factory;
    std::unique_ptr<gko::LinOp> solver;
    std::shared_ptr<DummyRPFactory::Factory> rp_factory;
    std::shared_ptr<DummyRPFactory::Factory> rp_factory2;
    std::shared_ptr<DummyFactory::Factory> lo_factory;

    static void assert_same_matrices(const Mtx *m1, const Mtx *m2)
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

TYPED_TEST_CASE(Multigrid, gko::test::ValueTypes);


TYPED_TEST(Multigrid, MultigridFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->multigrid_factory->get_executor(), this->exec);
}


TYPED_TEST(Multigrid, MultigridFactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(3, 3));
    auto multigrid_solver = static_cast<Solver *>(this->solver.get());
    ASSERT_NE(multigrid_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(multigrid_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(Multigrid, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->multigrid_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver.get());

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver *>(copy.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(Multigrid, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->multigrid_factory->generate(Mtx::create(this->exec));

    copy->copy_from(std::move(this->solver));

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver *>(copy.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(Multigrid, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto clone = this->solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    auto clone_mtx = static_cast<Solver *>(clone.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx *>(clone_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(Multigrid, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;
    this->solver->clear();

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
    auto solver_mtx =
        static_cast<Solver *>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
    auto rstr_prlg =
        static_cast<Solver *>(this->solver.get())->get_rstr_prlg_list();
    ASSERT_EQ(rstr_prlg.size(), 0);
}


TYPED_TEST(Multigrid, ApplyUsesInitialGuessReturnsTrue)
{
    ASSERT_TRUE(this->solver->apply_uses_initial_guess());
}

TYPED_TEST(Multigrid, EachLevelAreDistinct)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    using DummyFactory = typename TestFixture::DummyFactory;

    auto solver = static_cast<Solver *>(this->solver.get());
    auto rstr_prlg = solver->get_rstr_prlg_list();
    auto pre_smoother = solver->get_pre_smoother_list();
    auto post_smoother = solver->get_post_smoother_list();
    auto coarsest_solver = solver->get_coarsest_solver();

    ASSERT_EQ(rstr_prlg.size(), 2);
    ASSERT_NE(rstr_prlg.at(0), rstr_prlg.at(1));
    ASSERT_EQ(static_cast<DummyRPFactory *>(rstr_prlg.at(0).get())
                  ->get_parameters()
                  .value,
              5);
    ASSERT_EQ(static_cast<DummyRPFactory *>(rstr_prlg.at(1).get())
                  ->get_parameters()
                  .value,
              5);
    ASSERT_EQ(pre_smoother.size(), 2);
    ASSERT_NE(pre_smoother.at(0), pre_smoother.at(1));
    ASSERT_EQ(static_cast<DummyFactory *>(pre_smoother.at(0).get())
                  ->get_parameters()
                  .value,
              5);
    ASSERT_EQ(static_cast<DummyFactory *>(pre_smoother.at(1).get())
                  ->get_parameters()
                  .value,
              5);
    ASSERT_EQ(post_smoother.size(), 2);
    ASSERT_NE(post_smoother.at(0), post_smoother.at(1));
    ASSERT_EQ(static_cast<DummyFactory *>(post_smoother.at(0).get())
                  ->get_parameters()
                  .value,
              5);
    ASSERT_EQ(static_cast<DummyFactory *>(post_smoother.at(1).get())
                  ->get_parameters()
                  .value,
              5);
    ASSERT_NE(coarsest_solver, nullptr);
}

TYPED_TEST(Multigrid, TwoRstrPrlg)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;

    auto solver = Solver::build()
                      .with_max_levels(2u)
                      .with_coarsest_solver(this->lo_factory)
                      .with_rstr_prlg(this->rp_factory, this->rp_factory2)
                      .on(this->exec)
                      ->generate(this->mtx);
    auto rstr_prlg = solver->get_rstr_prlg_list();

    ASSERT_EQ(rstr_prlg.size(), 2);
    ASSERT_NE(rstr_prlg.at(0), rstr_prlg.at(1));
    ASSERT_EQ(static_cast<DummyRPFactory *>(rstr_prlg.at(0).get())
                  ->get_parameters()
                  .value,
              5);
    ASSERT_EQ(static_cast<DummyRPFactory *>(rstr_prlg.at(1).get())
                  ->get_parameters()
                  .value,
              2);
}


TYPED_TEST(Multigrid, CustomSelector)
{
    using Solver = typename TestFixture::Solver;
    using DummyRPFactory = typename TestFixture::DummyRPFactory;
    std::function<gko::size_type(const gko::size_type, const gko::size_type)>
        selector =
            [](const gko::size_type num_rows, const gko::size_type level) {
                return (level == 2) ? 1 : 0;
            };
    auto solver = Solver::build()
                      .with_max_levels(4u)
                      .with_coarsest_solver(this->lo_factory)
                      .with_rstr_prlg(this->rp_factory, this->rp_factory2)
                      .with_rstr_prlg_index(selector)
                      .on(this->exec)
                      ->generate(this->mtx);
    auto rstr_prlg = solver->get_rstr_prlg_list();

    ASSERT_EQ(rstr_prlg.size(), 4);
    ASSERT_EQ(static_cast<DummyRPFactory *>(rstr_prlg.at(0).get())
                  ->get_parameters()
                  .value,
              5);
    ASSERT_EQ(static_cast<DummyRPFactory *>(rstr_prlg.at(1).get())
                  ->get_parameters()
                  .value,
              5);
    ASSERT_EQ(static_cast<DummyRPFactory *>(rstr_prlg.at(2).get())
                  ->get_parameters()
                  .value,
              2);
    ASSERT_EQ(static_cast<DummyRPFactory *>(rstr_prlg.at(3).get())
                  ->get_parameters()
                  .value,
              5);
}


}  // namespace
