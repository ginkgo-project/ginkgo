// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/chebyshev_kernels.hpp"

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/multivector.hpp>
#include <ginkgo/core/solver/chebyshev.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/test/utils.hpp"

template <typename T>
class Chebyshev : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::MultiVector<value_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Chebyshev<value_type>;
    using coeff_type = gko::solver::detail::coeff_type<value_type>;
    Chebyshev()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Dense>(
              {{0.9, -1.0, 3.0}, {0.0, 1.0, 3.0}, {0.0, 0.0, 1.1}}, exec)),
          // Eigenvalues of mtx are 0.9, 1.0 and 1.1
          chebyshev_factory(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(30u),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value))
                  .with_foci(coeff_type{0.9}, coeff_type{1.1})
                  .on(exec)),
          alpha(0.5),
          beta(0.25),
          inner_sol(gko::initialize<Mtx>(
              {{0.5, 0.125, -0.125}, {0.25, 0.5, -1.0}, {1.5, -0.25, 1.5}},
              this->exec)),
          update_sol(gko::initialize<Mtx>(
              {{1.0, 0.0, -0.5}, {0.5, 1.0, -1.0}, {-1.5, -0.5, 1.0}},
              this->exec)),
          output(gko::initialize<Mtx>(
              {{-1.0, 0.5, -0.0}, {0.75, 0.25, -1.25}, {1.0, -1.25, 3.0}},
              this->exec))
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Dense> mtx;
    std::unique_ptr<typename Solver::Factory> chebyshev_factory;
    coeff_type alpha;
    coeff_type beta;
    std::shared_ptr<Mtx> inner_sol;
    std::shared_ptr<Mtx> update_sol;
    std::shared_ptr<Mtx> output;
};

TYPED_TEST_SUITE(Chebyshev, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Chebyshev, KernelInitUpdate)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;

    gko::kernels::reference::chebyshev::init_update(
        this->exec, this->alpha, this->inner_sol->get_const_device_view(),
        this->update_sol->get_device_view(), this->output->get_device_view());

    GKO_ASSERT_MTX_NEAR(this->update_sol, this->inner_sol, 0);
    GKO_ASSERT_MTX_NEAR(this->output,
                        gko::initialize<Mtx>({{-0.75, 0.5625, -0.0625},
                                              {0.875, 0.5, -1.75},
                                              {1.75, -1.375, 3.75}},
                                             this->exec),
                        r<value_type>::value);
}


TYPED_TEST(Chebyshev, KernelUpdate)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;

    gko::kernels::reference::chebyshev::update(
        this->exec, this->alpha, this->beta, this->inner_sol->get_device_view(),
        this->update_sol->get_device_view(), this->output->get_device_view());

    GKO_ASSERT_MTX_NEAR(this->update_sol, this->inner_sol, 0);
    GKO_ASSERT_MTX_NEAR(
        this->inner_sol,
        gko::initialize<Mtx>(
            {{0.75, 0.125, -0.25}, {0.375, 0.75, -1.25}, {1.125, -0.375, 1.75}},
            this->exec),
        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(this->output,
                        gko::initialize<Mtx>({{-0.625, 0.5625, -0.125},
                                              {0.9375, 0.625, -1.875},
                                              {1.5625, -1.4375, 3.875}},
                                             this->exec),
                        r<value_type>::value);
}


TYPED_TEST(Chebyshev, SolvesTriangularSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->chebyshev_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Chebyshev, SolvesTriangularSystemMixed)
{
    using mixed_type = gko::next_precision<typename TestFixture::value_type>;
    using MixedMtx = gko::matrix::MultiVector<mixed_type>;
    auto solver = this->chebyshev_factory->generate(this->mtx);
    auto b = gko::initialize<MixedMtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<MixedMtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}),
                        (r_mixed<mixed_type, TypeParam>()) * 1e1);
}


TYPED_TEST(Chebyshev, SolvesTriangularSystemWithIterativeInnerSolver)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using coeff_type = typename TestFixture::coeff_type;
    const gko::remove_complex<value_type> inner_reduction_factor = 1e-2;
    auto precond_factory = gko::share(
        gko::solver::Gmres<value_type>::build()
            .with_criteria(gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(inner_reduction_factor))
            .on(this->exec));
    auto solver_factory =
        gko::solver::Chebyshev<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(30u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(r<value_type>::value))
            .with_preconditioner(precond_factory)
            .with_foci(coeff_type{0.9}, coeff_type{1.1})
            .on(this->exec);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver_factory->generate(this->mtx)->apply(b, x);

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

    solver->apply(b, x);

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

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.5, 5.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Chebyshev, SolvesTriangularSystemUsingAdvancedApplyMixed)
{
    using mixed_type = gko::next_precision<typename TestFixture::value_type>;
    using MixedMtx = gko::matrix::MultiVector<mixed_type>;
    auto solver = this->chebyshev_factory->generate(this->mtx);
    auto alpha = gko::initialize<MixedMtx>({2.0}, this->exec);
    auto beta = gko::initialize<MixedMtx>({-1.0}, this->exec);
    auto b = gko::initialize<MixedMtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<MixedMtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.5, 5.0, 2.0}),
                        (r_mixed<mixed_type, TypeParam>()) * 1e1);
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

    solver->apply(alpha, b, beta, x);

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

    solver->transpose()->apply(b, x);

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

    solver->conj_transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Chebyshev, ApplyWithGivenInitialGuessModeIsEquivalentToRef)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using initial_guess_mode = gko::solver::initial_guess_mode;
    using coeff_type = typename TestFixture::coeff_type;
    auto ref_solver =
        gko::solver::Chebyshev<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_foci(coeff_type{0.9}, coeff_type{1.1})
            .on(this->exec)
            ->generate(this->mtx);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    for (auto guess : {initial_guess_mode::provided, initial_guess_mode::rhs,
                       initial_guess_mode::zero}) {
        auto solver =
            gko::solver::Chebyshev<value_type>::build()
                .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                .with_foci(coeff_type{0.9}, coeff_type{1.1})
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
