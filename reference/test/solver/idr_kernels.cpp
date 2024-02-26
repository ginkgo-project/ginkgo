// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/idr.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Idr : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Idr<value_type>;

    Idr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{1.0, -3.0, 0.0}, {-4.0, 1.0, -3.0}, {2.0, -1.0, 2.0}}, exec)),
          idr_factory(Solver::build()
                          .with_deterministic(true)
                          .with_criteria(
                              gko::stop::Iteration::build().with_max_iters(8u),
                              gko::stop::Time::build().with_time_limit(
                                  std::chrono::seconds(6)),
                              gko::stop::ResidualNorm<value_type>::build()
                                  .with_reduction_factor(r<value_type>::value))
                          .on(exec)),
          idr_factory_precision(
              Solver::build()
                  .with_deterministic(true)
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(50u),
                      gko::stop::Time::build().with_time_limit(
                          std::chrono::seconds(6)),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value))
                  .on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> idr_factory;
    std::unique_ptr<typename Solver::Factory> idr_factory_precision;
};

TYPED_TEST_SUITE(Idr, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Idr, SolvesDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->idr_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-4.0, -1.0, 4.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Idr, SolvesDenseSystemMixed)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->idr_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-4.0, -1.0, 4.0}),
                        (r_mixed<value_type, TypeParam>()) * 1e1);
}


TYPED_TEST(Idr, SolvesDenseSystemComplex)
{
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->idr_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {value_type{-1.0, 2.0}, value_type{3.0, -6.0}, value_type{1.0, -2.0}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.0, 0.0}, value_type{0.0, 0.0}, value_type{0.0, 0.0}},
        this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{-4.0, 8.0}, value_type{-1.0, 2.0},
                           value_type{4.0, -8.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Idr, SolvesDenseSystemMixedComplex)
{
    using value_type =
        gko::to_complex<gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->idr_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {value_type{-1.0, 2.0}, value_type{3.0, -6.0}, value_type{1.0, -2.0}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.0, 0.0}, value_type{0.0, 0.0}, value_type{0.0, 0.0}},
        this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{-4.0, 8.0}, value_type{-1.0, 2.0},
                           value_type{4.0, -8.0}}),
                        (r_mixed<value_type, TypeParam>()) * 1e1);
}


TYPED_TEST(Idr, SolvesDenseSystemWithComplexSubSpace)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using Solver = typename TestFixture::Solver;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto solver_factory =
        Solver::build()
            .with_complex_subspace(true)
            .with_deterministic(true)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(8u).on(this->exec),
                gko::stop::Time::build()
                    .with_time_limit(std::chrono::seconds(6))
                    .on(this->exec),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(r<value_type>::value)
                    .on(this->exec))
            .on(this->exec);
    auto solver = solver_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-4.0, -1.0, 4.0}), half_tol);
}


TYPED_TEST(Idr, SolvesMultipleDenseSystems)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto solver = this->idr_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {I<T>{-1.0, -5.0}, I<T>{3.0, 1.0}, I<T>{1.0, -2.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({{-4.0, 1.0}, {-1.0, 2.0}, {4.0, -1.0}}),
                        half_tol);
}


TYPED_TEST(Idr, SolvesMultipleDenseSystemsWithComplexSubspace)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    using Solver = typename TestFixture::Solver;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto solver_factory =
        Solver::build()
            .with_complex_subspace(true)
            .with_deterministic(true)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(8u).on(this->exec),
                gko::stop::Time::build()
                    .with_time_limit(std::chrono::seconds(6))
                    .on(this->exec),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(r<value_type>::value)
                    .on(this->exec))
            .on(this->exec);
    auto solver = solver_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {I<T>{-1.0, -5.0}, I<T>{3.0, 1.0}, I<T>{1.0, -2.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({{-4.0, 1.0}, {-1.0, 2.0}, {4.0, -1.0}}),
                        half_tol);
}


TYPED_TEST(Idr, SolvesDenseSystemUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->idr_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({-8.5, -3.0, 6.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Idr, SolvesDenseSystemUsingAdvancedApplyMixed)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->idr_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({-8.5, -3.0, 6.0}),
                        (r_mixed<value_type, TypeParam>()) * 1e1);
}


TYPED_TEST(Idr, SolvesDenseSystemUsingAdvancedApplyComplex)
{
    using Scalar = typename TestFixture::Mtx;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->idr_factory->generate(this->mtx);
    auto alpha = gko::initialize<Scalar>({2.0}, this->exec);
    auto beta = gko::initialize<Scalar>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>(
        {value_type{-1.0, 2.0}, value_type{3.0, -6.0}, value_type{1.0, -2.0}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.5, -1.0}, value_type{1.0, -2.0}, value_type{2.0, -4.0}},
        this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{-8.5, 17.0}, value_type{-3.0, 6.0},
                           value_type{6.0, -12.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Idr, SolvesDenseSystemUsingAdvancedApplyMixedComplex)
{
    using Scalar = gko::matrix::Dense<
        gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->idr_factory->generate(this->mtx);
    auto alpha = gko::initialize<Scalar>({2.0}, this->exec);
    auto beta = gko::initialize<Scalar>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>(
        {value_type{-1.0, 2.0}, value_type{3.0, -6.0}, value_type{1.0, -2.0}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.5, -1.0}, value_type{1.0, -2.0}, value_type{2.0, -4.0}},
        this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{-8.5, 17.0}, value_type{-3.0, 6.0},
                           value_type{6.0, -12.0}}),
                        (r_mixed<value_type, TypeParam>()) * 1e1);
}


TYPED_TEST(Idr, SolvesMultipleDenseSystemsUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto solver = this->idr_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>(
        {I<T>{-1.0, -5.0}, I<T>{3.0, 1.0}, I<T>{1.0, -2.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.5, 1.0}, I<T>{1.0, 2.0}, I<T>{2.0, 3.0}}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({{-8.5, 1.0}, {-3.0, 2.0}, {6.0, -5.0}}),
                        half_tol);
}


// The following test-data was generated and validated with MATLAB
TYPED_TEST(Idr, SolvesBigDenseSystemForDivergenceCheck1)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    std::shared_ptr<Mtx> locmtx =
        gko::initialize<Mtx>({{-19.0, 47.0, -41.0, 35.0, -21.0, 71.0},
                              {-8.0, -66.0, 29.0, -96.0, -95.0, -14.0},
                              {-93.0, -58.0, -9.0, -87.0, 15.0, 35.0},
                              {60.0, -86.0, 54.0, -40.0, -93.0, 56.0},
                              {53.0, 94.0, -54.0, 86.0, -61.0, 4.0},
                              {-42.0, 57.0, 32.0, 89.0, 89.0, -39.0}},
                             this->exec);
    auto solver = this->idr_factory_precision->generate(locmtx);
    auto b =
        gko::initialize<Mtx>({0.0, -9.0, -2.0, 8.0, -5.0, -6.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    auto one_op = gko::initialize<gko::matrix::Dense<value_type>>(
        {gko::one<value_type>()}, this->exec);
    auto neg_one_op = gko::initialize<gko::matrix::Dense<value_type>>(
        {-gko::one<value_type>()}, this->exec);
    auto resnorm = gko::matrix::Dense<gko::remove_complex<value_type>>::create(
        this->exec, gko::dim<2>{1, 1});
    locmtx->apply(neg_one_op, x, one_op, b);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({0.13853406350816114, -0.08147485210505287, -0.0450299311807042,
           -0.0051264177562865719, 0.11609654300797841, 0.1018688746740561}),
        half_tol * 5e-1);
}


TYPED_TEST(Idr, SolvesBigDenseSystemForDivergenceCheck2)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    std::shared_ptr<Mtx> locmtx =
        gko::initialize<Mtx>({{-19.0, 47.0, -41.0, 35.0, -21.0, 71.0},
                              {-8.0, -66.0, 29.0, -96.0, -95.0, -14.0},
                              {-93.0, -58.0, -9.0, -87.0, 15.0, 35.0},
                              {60.0, -86.0, 54.0, -40.0, -93.0, 56.0},
                              {53.0, 94.0, -54.0, 86.0, -61.0, 4.0},
                              {-42.0, 57.0, 32.0, 89.0, 89.0, -39.0}},
                             this->exec);
    auto solver = this->idr_factory_precision->generate(locmtx);
    auto b =
        gko::initialize<Mtx>({9.0, -4.0, -6.0, -10.0, 1.0, 10.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({0.13517641417299162, 0.75117689075221139, 0.47572853185155239,
           -0.50927993095367852, 0.13463333820848167, 0.23126768306576015}),
        half_tol * 1e-1);
}


TYPED_TEST(Idr, SolvesMultipleDenseSystemsDivergenceCheck)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    std::shared_ptr<Mtx> locmtx =
        gko::initialize<Mtx>({{-19.0, 47.0, -41.0, 35.0, -21.0, 71.0},
                              {-8.0, -66.0, 29.0, -96.0, -95.0, -14.0},
                              {-93.0, -58.0, -9.0, -87.0, 15.0, 35.0},
                              {60.0, -86.0, 54.0, -40.0, -93.0, 56.0},
                              {53.0, 94.0, -54.0, 86.0, -61.0, 4.0},
                              {-42.0, 57.0, 32.0, 89.0, 89.0, -39.0}},
                             this->exec);
    auto solver = this->idr_factory_precision->generate(locmtx);
    auto b1 =
        gko::initialize<Mtx>({0.0, -9.0, -2.0, 8.0, -5.0, -6.0}, this->exec);
    auto x1 = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);
    auto b2 =
        gko::initialize<Mtx>({9.0, -4.0, -6.0, -10.0, 1.0, 10.0}, this->exec);
    auto x2 = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);
    auto bc = gko::initialize<Mtx>({I<T>{0., 0.}, I<T>{0., 0.}, I<T>{0., 0.},
                                    I<T>{0., 0.}, I<T>{0., 0.}, I<T>{0., 0.}},
                                   this->exec);
    auto xc = gko::initialize<Mtx>({I<T>{0., 0.}, I<T>{0., 0.}, I<T>{0., 0.},
                                    I<T>{0., 0.}, I<T>{0., 0.}, I<T>{0., 0.}},
                                   this->exec);
    for (size_t i = 0; i < xc->get_size()[0]; ++i) {
        bc->at(i, 0) = b1->at(i);
        bc->at(i, 1) = b2->at(i);
        xc->at(i, 0) = x1->at(i);
        xc->at(i, 1) = x2->at(i);
    }

    solver->apply(b1, x1);
    solver->apply(b2, x2);
    solver->apply(bc, xc);
    auto testMtx =
        gko::initialize<Mtx>({I<T>{0., 0.}, I<T>{0., 0.}, I<T>{0., 0.},
                              I<T>{0., 0.}, I<T>{0., 0.}, I<T>{0., 0.}},
                             this->exec);

    for (size_t i = 0; i < testMtx->get_size()[0]; ++i) {
        testMtx->at(i, 0) = x1->at(i);
        testMtx->at(i, 1) = x2->at(i);
    }

    auto alpha = gko::initialize<Mtx>({1.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto residual1 = gko::initialize<Mtx>({0.}, this->exec);
    residual1->copy_from(b1);
    auto residual2 = gko::initialize<Mtx>({0.}, this->exec);
    residual2->copy_from(b2);
    auto residualC = gko::initialize<Mtx>({0.}, this->exec);
    residualC->copy_from(bc);

    locmtx->apply(alpha, x1, beta, residual1);
    locmtx->apply(alpha, x2, beta, residual2);
    locmtx->apply(alpha, xc, beta, residualC);

    auto normS1 = inf_norm(residual1);
    auto normS2 = inf_norm(residual2);
    auto normC1 = inf_norm(residualC, 0);
    auto normC2 = inf_norm(residualC, 1);
    auto normB1 = inf_norm(bc, 0);
    auto normB2 = inf_norm(bc, 1);

    // make sure that all combined solutions are as good or better than the
    // single solutions
    ASSERT_LE(normC1 / normB1, normS1 / normB1 + r<value_type>::value * 1e2);
    ASSERT_LE(normC2 / normB2, normS2 / normB2 + r<value_type>::value * 1e2);

    // Not sure if this is necessary, the assertions above should cover what
    // is needed.
    GKO_ASSERT_MTX_NEAR(xc, testMtx, r<value_type>::value * 1e2);
}


TYPED_TEST(Idr, SolvesTransposedDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto solver = this->idr_factory->generate(this->mtx->transpose());
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-4.0, -1.0, 4.0}), half_tol);
}


TYPED_TEST(Idr, SolvesConjTransposedDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto solver = this->idr_factory->generate(this->mtx->conj_transpose());
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->conj_transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-4.0, -1.0, 4.0}), half_tol);
}


}  // namespace
