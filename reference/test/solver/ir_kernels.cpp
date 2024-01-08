// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/ir.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/solver/ir_kernels.hpp"
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
              {{0.9, -1.0, 3.0}, {0.0, 1.0, 3.0}, {0.0, 0.0, 1.1}}, exec)),
          // Eigenvalues of mtx are 0.9, 1.0 and 1.1
          // Richardson iteration, converges since
          // | relaxation_factor * lambda - 1 | < 1
          ir_factory(Solver::build()
                         .with_criteria(
                             gko::stop::Iteration::build().with_max_iters(30u),
                             gko::stop::ResidualNorm<value_type>::build()
                                 .with_reduction_factor(r<value_type>::value))
                         .on(exec))
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> ir_factory;
};

TYPED_TEST_SUITE(Ir, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Ir, KernelInitialize)
{
    gko::stopping_status stopped{};
    gko::stopping_status non_stopped{};
    auto stop = gko::array<gko::stopping_status>(this->exec, 2);
    stopped.stop(1);
    non_stopped.reset();
    std::fill_n(stop.get_data(), stop.get_size(), non_stopped);

    gko::kernels::reference::ir::initialize(this->exec, &stop);

    ASSERT_EQ(stop.get_data()[0], non_stopped);
    ASSERT_EQ(stop.get_data()[1], non_stopped);
}


TYPED_TEST(Ir, SolvesTriangularSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->ir_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Ir, SolvesTriangularSystemMixed)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->ir_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}),
                        (r_mixed<value_type, TypeParam>()) * 1e1);
}


TYPED_TEST(Ir, SolvesTriangularSystemComplex)
{
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->ir_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {value_type{3.9, -7.8}, value_type{9.0, -18.0}, value_type{2.2, -4.4}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.0, 0.0}, value_type{0.0, 0.0}, value_type{0.0, 0.0}},
        this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.0, -2.0}, value_type{3.0, -6.0},
                           value_type{2.0, -4.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Ir, SolvesTriangularSystemMixedComplex)
{
    using value_type =
        gko::to_complex<gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->ir_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {value_type{3.9, -7.8}, value_type{9.0, -18.0}, value_type{2.2, -4.4}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.0, 0.0}, value_type{0.0, 0.0}, value_type{0.0, 0.0}},
        this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.0, -2.0}, value_type{3.0, -6.0},
                           value_type{2.0, -4.0}}),
                        (r_mixed<value_type, TypeParam>()) * 1e1);
}


TYPED_TEST(Ir, SolvesTriangularSystemWithIterativeInnerSolver)
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
        gko::solver::Ir<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(30u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(r<value_type>::value))
            .with_solver(inner_solver_factory)
            .on(this->exec);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver_factory->generate(this->mtx)->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Ir, SolvesMultipleTriangularSystems)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto solver = this->ir_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {I<T>{3.9, 2.9}, I<T>{9.0, 4.0}, I<T>{2.2, 1.1}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({{1.0, 1.0}, {3.0, 1.0}, {2.0, 1.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Ir, SolvesTriangularSystemUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->ir_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.5, 5.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Ir, SolvesTriangularSystemUsingAdvancedApplyMixed)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->ir_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.5, 5.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Ir, SolvesTriangularSystemUsingAdvancedApplyComplex)
{
    using Scalar = typename TestFixture::Mtx;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->ir_factory->generate(this->mtx);
    auto alpha = gko::initialize<Scalar>({2.0}, this->exec);
    auto beta = gko::initialize<Scalar>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>(
        {value_type{3.9, -7.8}, value_type{9.0, -18.0}, value_type{2.2, -4.4}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.5, -1.0}, value_type{1.0, -2.0}, value_type{2.0, -4.0}},
        this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.5, -3.0}, value_type{5.0, -10.0},
                           value_type{2.0, -4.0}}),
                        (r_mixed<value_type, TypeParam>()) * 1e1);
}


TYPED_TEST(Ir, SolvesTriangularSystemUsingAdvancedApplyMixedComplex)
{
    using Scalar = gko::matrix::Dense<
        gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->ir_factory->generate(this->mtx);
    auto alpha = gko::initialize<Scalar>({2.0}, this->exec);
    auto beta = gko::initialize<Scalar>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>(
        {value_type{3.9, -7.8}, value_type{9.0, -18.0}, value_type{2.2, -4.4}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.5, -1.0}, value_type{1.0, -2.0}, value_type{2.0, -4.0}},
        this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.5, -3.0}, value_type{5.0, -10.0},
                           value_type{2.0, -4.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Ir, SolvesMultipleStencilSystemsUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto solver = this->ir_factory->generate(this->mtx);
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


TYPED_TEST(Ir, SolvesTransposedTriangularSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->ir_factory->generate(this->mtx->transpose());
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Ir, SolvesConjTransposedTriangularSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->ir_factory->generate(this->mtx->conj_transpose());
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->conj_transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Ir, RichardsonSolvesTriangularSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver =
        gko::solver::Ir<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(100u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(r<value_type>::value)
                               .on(this->exec))
            .with_relaxation_factor(value_type{0.9})
            .on(this->exec)
            ->generate(this->mtx);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Ir, RichardsonSolvesTriangularSystemWithIterativeInnerSolver)
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
        gko::solver::Ir<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(100u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(r<value_type>::value)
                               .on(this->exec))
            .with_relaxation_factor(value_type{0.9})
            .with_solver(inner_solver_factory)
            .on(this->exec);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver_factory->generate(this->mtx)->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Ir, RichardsonTransposedSolvesTriangularSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver =
        gko::solver::Ir<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(30u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(r<value_type>::value)
                               .on(this->exec))
            .with_relaxation_factor(value_type{0.9})
            .on(this->exec)
            ->generate(this->mtx->transpose());
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Ir, RichardsonConjTransposedSolvesTriangularSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver =
        gko::solver::Ir<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(30u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(r<value_type>::value)
                               .on(this->exec))
            .with_relaxation_factor(value_type{0.9})
            .on(this->exec)
            ->generate(this->mtx->conj_transpose());
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->conj_transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Ir, ApplyWithGivenInitialGuessModeIsEquivalentToRef)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using initial_guess_mode = gko::solver::initial_guess_mode;
    auto ref_solver =
        gko::solver::Ir<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(this->exec)
            ->generate(this->mtx);
    auto b = gko::initialize<Mtx>({3.9, 9.0, 2.2}, this->exec);
    for (auto guess : {initial_guess_mode::provided, initial_guess_mode::rhs,
                       initial_guess_mode::zero}) {
        auto solver =
            gko::solver::Ir<value_type>::build()
                .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
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
