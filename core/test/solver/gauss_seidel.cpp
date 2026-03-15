// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/solver/gauss_seidel.hpp>
#include <ginkgo/core/stop/iteration.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class FwdGaussSeidel : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Ell<value_type, index_type>;
    using Solver = gko::solver::FwdGaussSeidel<value_type, index_type>;

    FwdGaussSeidel()
        : exec(gko::ReferenceExecutor::create()),
          // 4x4 symmetric positive-definite matrix (two-color ordering):
          //   color 0: rows 0, 1  color 1: rows 2, 3
          mtx(gko::initialize<Mtx>(
              // clang-format off
              {{2.0, 0.0, 1.0, 0.0},
               {0.0, 3.0, 0.0, 1.0},
               {1.0, 0.0, 4.0, 0.0},
               {0.0, 1.0, 0.0, 5.0}},
              // clang-format on
              exec)),
          gs_factory(Solver::build()
                         .with_criteria(
                             gko::stop::Iteration::build().with_max_iters(1u))
                         .with_color_ptrs(std::vector<index_type>{0, 2, 4})
                         .on(exec)),
          solver(gs_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> gs_factory;
    std::unique_ptr<gko::LinOp> solver;
};

TYPED_TEST_SUITE(FwdGaussSeidel, gko::test::ValueIndexTypesBase,
                 PairTypenameNameGenerator);


TYPED_TEST(FwdGaussSeidel, FactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->gs_factory->get_executor(), this->exec);
}


TYPED_TEST(FwdGaussSeidel, FactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(4, 4));
    auto gs = static_cast<Solver*>(this->solver.get());
    ASSERT_NE(gs->get_system_matrix(), nullptr);
    ASSERT_EQ(gs->get_system_matrix(), this->mtx);
}


TYPED_TEST(FwdGaussSeidel, ApplyUsesInitialGuessReturnsTrue)
{
    ASSERT_TRUE(this->solver->apply_uses_initial_guess());
}


TYPED_TEST(FwdGaussSeidel, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto empty = this->gs_factory->generate(Mtx::create(this->exec));

    empty->copy_from(this->solver);

    ASSERT_EQ(empty->get_size(), gko::dim<2>(4, 4));
    auto copy_mtx = static_cast<Solver*>(empty.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_mtx), this->mtx, 0.0);
}


TYPED_TEST(FwdGaussSeidel, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto empty = this->gs_factory->generate(Mtx::create(this->exec));

    empty->move_from(this->solver);

    ASSERT_EQ(empty->get_size(), gko::dim<2>(4, 4));
    auto moved_mtx = static_cast<Solver*>(empty.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(moved_mtx), this->mtx, 0.0);
}


TYPED_TEST(FwdGaussSeidel, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;

    auto clone = this->solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(4, 4));
    auto clone_mtx = static_cast<Solver*>(clone.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(clone_mtx), this->mtx, 0.0);
}


TYPED_TEST(FwdGaussSeidel, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;

    this->solver->clear();

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(static_cast<Solver*>(this->solver.get())->get_system_matrix(),
              nullptr);
}


TYPED_TEST(FwdGaussSeidel, ColorPtrsAreStoredFromParameters)
{
    using Solver = typename TestFixture::Solver;
    using index_type = typename TestFixture::index_type;

    auto gs = static_cast<const Solver*>(this->solver.get());
    const auto& stored = gs->get_parameters().color_ptrs;

    ASSERT_EQ(stored.size(), 3u);
    EXPECT_EQ(stored[0], index_type{0});
    EXPECT_EQ(stored[1], index_type{2});
    EXPECT_EQ(stored[2], index_type{4});
}


TYPED_TEST(FwdGaussSeidel, SolvesToKnownExactSolutionAfterFiveIterations)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Solver = typename TestFixture::Solver;
    using Vec = gko::matrix::Dense<value_type>;

    // RHS with mixed signs; exact solution x* = {4/7, -1, 6/7, 0}.
    // Error in x[0] contracts by factor 1/8 per sweep: after 5 iterations
    // max component error ≈ 1e-4, well within the 1e-3 tolerance below.
    auto b = gko::initialize<Vec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x = gko::initialize<Vec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    auto exact = gko::initialize<Vec>({value_type{4.0 / 7.0}, value_type{-1.0},
                                       value_type{6.0 / 7.0}, value_type{0.0}},
                                      this->exec);

    auto solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(5u))
            .with_color_ptrs(std::vector<index_type>{0, 2, 4})
            .on(this->exec)
            ->generate(this->mtx);
    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, exact, 1e-3);
}


}  // namespace
