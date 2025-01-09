/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/solver/chebyshev.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Chebyshev : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Chebyshev<value_type>;
    Chebyshev()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{0.9, -1.0, 3.0}, {0.0, 1.0, 3.0}, {0.0, 0.0, 1.1}}, exec)),
          // Eigenvalues of mtx are 0.9, 1.0 and 1.1
          chebyshev_factory(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(30u).on(
                          exec),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value)
                          .on(exec))
                  .with_foci(value_type{0.9}, value_type{1.1})
                  .on(exec))
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> chebyshev_factory;
};

TYPED_TEST_SUITE(Chebyshev, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Chebyshev, CheckDefaultNumAlphaBetaWithoutIteration)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto upper = value_type{1.1};
    auto lower = value_type{0.9};
    auto factory =
        Solver::build()
            .with_criteria(gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(r<value_type>::value)
                               .on(this->exec))
            .with_foci(lower, upper)
            .on(this->exec);
    auto solver = factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    auto alpha = gko::as<gko::matrix::Dense<value_type>>(
        solver->get_workspace_op(gko::solver::workspace_traits<Solver>::alpha));
    auto beta = gko::as<gko::matrix::Dense<value_type>>(
        solver->get_workspace_op(gko::solver::workspace_traits<Solver>::beta));
    // if the stop criterion does not contain iteration limit, it will use the
    // default value.
    ASSERT_EQ(alpha->get_size(), (gko::dim<2>{1, 4}));
    ASSERT_EQ(beta->get_size(), (gko::dim<2>{1, 4}));
}


TYPED_TEST(Chebyshev, CheckDefaultNumAlphaBetaWithLessIteration)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto upper = value_type{1.1};
    auto lower = value_type{0.9};
    auto factory =
        Solver::build()
            .with_criteria(
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(r<value_type>::value)
                    .on(this->exec),
                gko::stop::Iteration::build().with_max_iters(1u).on(this->exec))
            .with_foci(lower, upper)
            .on(this->exec);
    auto solver = factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    auto alpha = gko::as<gko::matrix::Dense<value_type>>(
        solver->get_workspace_op(gko::solver::workspace_traits<Solver>::alpha));
    auto beta = gko::as<gko::matrix::Dense<value_type>>(
        solver->get_workspace_op(gko::solver::workspace_traits<Solver>::beta));
    // if the iteration limit less than the default value, it will use the
    // default value.
    ASSERT_EQ(alpha->get_size(), (gko::dim<2>{1, 4}));
    ASSERT_EQ(beta->get_size(), (gko::dim<2>{1, 4}));
}


TYPED_TEST(Chebyshev, CheckStoredAlphaBeta)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto upper = value_type{1.1};
    auto lower = value_type{0.9};
    auto factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(6u).on(this->exec))
            .with_foci(lower, upper)
            .on(this->exec);
    auto solver = factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    auto alpha = gko::as<gko::matrix::Dense<value_type>>(
        solver->get_workspace_op(gko::solver::workspace_traits<Solver>::alpha));
    auto beta = gko::as<gko::matrix::Dense<value_type>>(
        solver->get_workspace_op(gko::solver::workspace_traits<Solver>::beta));
    // the iteration is more than default
    ASSERT_EQ(alpha->get_size(), (gko::dim<2>{1, 7}));
    ASSERT_EQ(beta->get_size(), (gko::dim<2>{1, 7}));
    // check the num_keep alpha, beta
    auto d = (upper + lower) / value_type{2};
    auto c = (upper - lower) / value_type{2};
    EXPECT_EQ(alpha->at(0, 0), value_type{1} / d);
    EXPECT_EQ(beta->at(0, 0), value_type{0});
    EXPECT_EQ(beta->at(0, 1),
              value_type{0.5} * (c * alpha->at(0, 0)) * (c * alpha->at(0, 0)));
    EXPECT_EQ(alpha->at(0, 1),
              value_type{1} / (d - beta->at(0, 1) / alpha->at(0, 0)));
    EXPECT_EQ(beta->at(0, 2), (c * alpha->at(0, 1) / value_type{2}) *
                                  (c * alpha->at(0, 1) / value_type{2}));
    EXPECT_EQ(alpha->at(0, 2),
              value_type{1} / (d - beta->at(0, 2) / alpha->at(0, 1)));
}


TYPED_TEST(Chebyshev, NumAlphaBetaFromChangingCriterion)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto upper = value_type{1.1};
    auto lower = value_type{0.9};
    auto factory =
        Solver::build()
            .with_criteria(
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(r<value_type>::value)
                    .on(this->exec),
                gko::stop::Iteration::build().with_max_iters(6u).on(this->exec))
            .with_foci(lower, upper)
            .on(this->exec);
    auto solver = factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    // same as previous test, but it works with combined factory
    solver->apply(b.get(), x.get());

    auto alpha = gko::as<gko::matrix::Dense<value_type>>(
        solver->get_workspace_op(gko::solver::workspace_traits<Solver>::alpha));
    auto beta = gko::as<gko::matrix::Dense<value_type>>(
        solver->get_workspace_op(gko::solver::workspace_traits<Solver>::beta));
    // if the iteration limit is less than the default value, it will use the
    // default value.
    ASSERT_EQ(alpha->get_size(), (gko::dim<2>{1, 7}));
    ASSERT_EQ(beta->get_size(), (gko::dim<2>{1, 7}));
    {
        // Set less iteration limit
        solver->set_stop_criterion_factory(
            gko::stop::Iteration::build().with_max_iters(4u).on(this->exec));

        solver->apply(b.get(), x.get());

        auto alpha_tmp =
            gko::as<gko::matrix::Dense<value_type>>(solver->get_workspace_op(
                gko::solver::workspace_traits<Solver>::alpha));
        auto beta_tmp =
            gko::as<gko::matrix::Dense<value_type>>(solver->get_workspace_op(
                gko::solver::workspace_traits<Solver>::beta));
        // if the iteration limit is less than the previous one, it keeps the
        // storage.
        ASSERT_EQ(alpha_tmp->get_size(), (gko::dim<2>{1, 7}));
        ASSERT_EQ(beta_tmp->get_size(), (gko::dim<2>{1, 7}));
        ASSERT_EQ(alpha_tmp->get_const_values(), alpha->get_const_values());
        ASSERT_EQ(beta_tmp->get_const_values(), beta->get_const_values());
    }
    {
        // Set more iteration limit
        solver->set_stop_criterion_factory(
            gko::stop::Iteration::build().with_max_iters(10u).on(this->exec));

        solver->apply(b.get(), x.get());

        auto alpha_tmp =
            gko::as<gko::matrix::Dense<value_type>>(solver->get_workspace_op(
                gko::solver::workspace_traits<Solver>::alpha));
        auto beta_tmp =
            gko::as<gko::matrix::Dense<value_type>>(solver->get_workspace_op(
                gko::solver::workspace_traits<Solver>::beta));
        // if the iteration limit is less than the previous one, it keeps the
        // storage.
        ASSERT_EQ(alpha_tmp->get_size(), (gko::dim<2>{1, 11}));
        ASSERT_EQ(beta_tmp->get_size(), (gko::dim<2>{1, 11}));
        ASSERT_NE(alpha_tmp->get_const_values(), alpha->get_const_values());
        ASSERT_NE(beta_tmp->get_const_values(), beta->get_const_values());
    }
}


TYPED_TEST(Chebyshev, SolvesTriangularSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->chebyshev_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Chebyshev, SolvesTriangularSystemMixed)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->chebyshev_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}),
                        (r_mixed<value_type, TypeParam>()) * 1e1);
}


TYPED_TEST(Chebyshev, SolvesTriangularSystemComplex)
{
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->chebyshev_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {value_type{3.9, -7.8}, value_type{9.0, -18.0}, value_type{2.2, -4.4}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.0, 0.0}, value_type{0.0, 0.0}, value_type{0.0, 0.0}},
        this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.0, -2.0}, value_type{3.0, -6.0},
                           value_type{2.0, -4.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Chebyshev, SolvesTriangularSystemMixedComplex)
{
    using value_type =
        gko::to_complex<gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->chebyshev_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {value_type{3.9, -7.8}, value_type{9.0, -18.0}, value_type{2.2, -4.4}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.0, 0.0}, value_type{0.0, 0.0}, value_type{0.0, 0.0}},
        this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.0, -2.0}, value_type{3.0, -6.0},
                           value_type{2.0, -4.0}}),
                        (r_mixed<value_type, TypeParam>()) * 1e1);
}


TYPED_TEST(Chebyshev, SolvesTriangularSystemWithIterativeInnerSolver)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;

    const gko::remove_complex<value_type> inner_reduction_factor = 1e-2;
    auto inner_solver_factory = gko::share(
        gko::solver::Gmres<value_type>::build()
            .with_criteria(gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(inner_reduction_factor)
                               .on(this->exec))
            .on(this->exec));

    auto solver_factory =
        gko::solver::Chebyshev<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(30u).on(
                               this->exec),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(r<value_type>::value)
                               .on(this->exec))
            .with_solver(inner_solver_factory)
            .with_foci(value_type{0.9}, value_type{1.1})
            .on(this->exec);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver_factory->generate(this->mtx)->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Chebyshev, SolvesMultipleTriangularSystems)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto solver = this->chebyshev_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {I<T>{3.9, 2.9}, I<T>{9.0, 4.0}, I<T>{2.2, 1.1}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({{1.0, 1.0}, {3.0, 1.0}, {2.0, 1.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Chebyshev, SolvesTriangularSystemUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->chebyshev_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.5, 5.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Chebyshev, SolvesTriangularSystemUsingAdvancedApplyMixed)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->chebyshev_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.5, 5.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Chebyshev, SolvesTriangularSystemUsingAdvancedApplyComplex)
{
    using Scalar = typename TestFixture::Mtx;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->chebyshev_factory->generate(this->mtx);
    auto alpha = gko::initialize<Scalar>({2.0}, this->exec);
    auto beta = gko::initialize<Scalar>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>(
        {value_type{3.9, -7.8}, value_type{9.0, -18.0}, value_type{2.2, -4.4}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.5, -1.0}, value_type{1.0, -2.0}, value_type{2.0, -4.0}},
        this->exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.5, -3.0}, value_type{5.0, -10.0},
                           value_type{2.0, -4.0}}),
                        (r_mixed<value_type, TypeParam>()) * 1e1);
}


TYPED_TEST(Chebyshev, SolvesTriangularSystemUsingAdvancedApplyMixedComplex)
{
    using Scalar = gko::matrix::Dense<
        gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->chebyshev_factory->generate(this->mtx);
    auto alpha = gko::initialize<Scalar>({2.0}, this->exec);
    auto beta = gko::initialize<Scalar>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>(
        {value_type{3.9, -7.8}, value_type{9.0, -18.0}, value_type{2.2, -4.4}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.5, -1.0}, value_type{1.0, -2.0}, value_type{2.0, -4.0}},
        this->exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.5, -3.0}, value_type{5.0, -10.0},
                           value_type{2.0, -4.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Chebyshev, SolvesMultipleStencilSystemsUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto solver = this->chebyshev_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>(
        {I<T>{3.9, 2.9}, I<T>{9.0, 4.0}, I<T>{2.2, 1.1}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.5, 1.0}, I<T>{1.0, 2.0}, I<T>{2.0, 3.0}}, this->exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({{1.5, 1.0}, {5.0, 0.0}, {2.0, -1.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Chebyshev, SolvesTransposedTriangularSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->chebyshev_factory->generate(this->mtx->transpose());
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->transpose()->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Chebyshev, SolvesConjTransposedTriangularSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver =
        this->chebyshev_factory->generate(this->mtx->conj_transpose());
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->conj_transpose()->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Chebyshev, ApplyWithGivenInitialGuessModeIsEquivalentToRef)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using initial_guess_mode = gko::solver::initial_guess_mode;
    auto ref_solver =
        gko::solver::Chebyshev<value_type>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u).on(this->exec))
            .with_foci(value_type{0.9}, value_type{1.1})
            .on(this->exec)
            ->generate(this->mtx);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    for (auto guess : {initial_guess_mode::provided, initial_guess_mode::rhs,
                       initial_guess_mode::zero}) {
        auto solver =
            gko::solver::Chebyshev<value_type>::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(1u).on(
                        this->exec))
                .with_foci(value_type{0.9}, value_type{1.1})
                .with_default_initial_guess(guess)
                .on(this->exec)
                ->generate(this->mtx);
        auto x = gko::initialize<Mtx>({1.0, -1.0, 1.0}, this->exec);
        std::shared_ptr<Mtx> ref_x = nullptr;
        if (guess == initial_guess_mode::provided) {
            ref_x = x->clone();
        } else if (guess == initial_guess_mode::rhs) {
            ref_x = b->clone();
        } else {
            ref_x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);
        }
        solver->apply(b, x);
        ref_solver->apply(b, ref_x);

        GKO_ASSERT_MTX_NEAR(x, ref_x, 0.0);
    }
}


}  // namespace
