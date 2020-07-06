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

#include <ginkgo/core/solver/bicgstab.hpp>


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
class Bicgstab : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Bicgstab<value_type>;

    Bicgstab()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{1.0, -3.0, 0.0}, {-4.0, 1.0, -3.0}, {2.0, -1.0, 2.0}}, exec)),
          bicgstab_factory(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(8u).on(exec),
                      gko::stop::Time::build()
                          .with_time_limit(std::chrono::seconds(6))
                          .on(exec),
                      gko::stop::ResidualNormReduction<value_type>::build()
                          .with_reduction_factor(r<value_type>::value)
                          .on(exec))
                  .on(exec)),
          bicgstab_factory_precision(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(50u).on(
                          exec),
                      gko::stop::Time::build()
                          .with_time_limit(std::chrono::seconds(6))
                          .on(exec),
                      gko::stop::ResidualNormReduction<value_type>::build()
                          .with_reduction_factor(r<value_type>::value)
                          .on(exec))
                  .on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> bicgstab_factory;
    std::unique_ptr<typename Solver::Factory> bicgstab_factory_precision;
};

TYPED_TEST_CASE(Bicgstab, gko::test::ValueTypes);


TYPED_TEST(Bicgstab, SolvesDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto solver = this->bicgstab_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({-4.0, -1.0, 4.0}), half_tol);
}


TYPED_TEST(Bicgstab, SolvesMultipleDenseSystems)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto solver = this->bicgstab_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {I<T>{-1.0, -5.0}, I<T>{3.0, 1.0}, I<T>{1.0, -2.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({{-4.0, 1.0}, {-1.0, 2.0}, {4.0, -1.0}}),
                        half_tol);
}


TYPED_TEST(Bicgstab, SolvesDenseSystemUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto solver = this->bicgstab_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());


    GKO_ASSERT_MTX_NEAR(x, l({-8.5, -3.0, 6.0}), half_tol);
}


TYPED_TEST(Bicgstab, SolvesMultipleDenseSystemsUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto solver = this->bicgstab_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>(
        {I<T>{-1.0, -5.0}, I<T>{3.0, 1.0}, I<T>{1.0, -2.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.5, 1.0}, I<T>{1.0, 2.0}, I<T>{2.0, 3.0}}, this->exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());


    GKO_ASSERT_MTX_NEAR(x, l({{-8.5, 1.0}, {-3.0, 2.0}, {6.0, -5.0}}),
                        half_tol);
}


// The following test-data was generated and validated with MATLAB
TYPED_TEST(Bicgstab, SolvesBigDenseSystemForDivergenceCheck1)
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
    auto solver = this->bicgstab_factory_precision->generate(locmtx);
    auto b =
        gko::initialize<Mtx>({0.0, -9.0, -2.0, 8.0, -5.0, -6.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x,
        l({0.13853406350816114, -0.08147485210505287, -0.0450299311807042,
           -0.0051264177562865719, 0.11609654300797841, 0.1018688746740561}),
        half_tol * 5e-1);
}


TYPED_TEST(Bicgstab, SolvesBigDenseSystemForDivergenceCheck2)
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
    auto solver = this->bicgstab_factory_precision->generate(locmtx);
    auto b =
        gko::initialize<Mtx>({9.0, -4.0, -6.0, -10.0, 1.0, 10.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x,
        l({0.13517641417299162, 0.75117689075221139, 0.47572853185155239,
           -0.50927993095367852, 0.13463333820848167, 0.23126768306576015}),
        half_tol * 1e-1);
}


template <typename T>
gko::remove_complex<T> infNorm(gko::matrix::Dense<T> *mat, size_t col = 0)
{
    using std::abs;
    using no_cpx_t = gko::remove_complex<T>;
    no_cpx_t norm = 0.0;
    for (size_t i = 0; i < mat->get_size()[0]; ++i) {
        no_cpx_t absEntry = abs(mat->at(i, col));
        if (norm < absEntry) norm = absEntry;
    }
    return norm;
}


TYPED_TEST(Bicgstab, SolvesMultipleDenseSystemsDivergenceCheck)
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
    auto solver = this->bicgstab_factory_precision->generate(locmtx);
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

    solver->apply(b1.get(), x1.get());
    solver->apply(b2.get(), x2.get());
    solver->apply(bc.get(), xc.get());
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
    residual1->copy_from(b1->clone());
    auto residual2 = gko::initialize<Mtx>({0.}, this->exec);
    residual2->copy_from(b2->clone());
    auto residualC = gko::initialize<Mtx>({0.}, this->exec);
    residualC->copy_from(bc->clone());

    locmtx->apply(alpha.get(), x1.get(), beta.get(), residual1.get());
    locmtx->apply(alpha.get(), x2.get(), beta.get(), residual2.get());
    locmtx->apply(alpha.get(), xc.get(), beta.get(), residualC.get());

    auto normS1 = infNorm(residual1.get());
    auto normS2 = infNorm(residual2.get());
    auto normC1 = infNorm(residualC.get(), 0);
    auto normC2 = infNorm(residualC.get(), 1);
    auto normB1 = infNorm(bc.get(), 0);
    auto normB2 = infNorm(bc.get(), 1);

    // make sure that all combined solutions are as good or better than the
    // single solutions
    ASSERT_LE(normC1 / normB1, normS1 / normB1 + r<value_type>::value * 1e2);
    ASSERT_LE(normC2 / normB2, normS2 / normB2 + r<value_type>::value * 1e2);

    // Not sure if this is necessary, the assertions above should cover what is
    // needed.
    GKO_ASSERT_MTX_NEAR(xc, testMtx, r<value_type>::value);
}


TYPED_TEST(Bicgstab, SolvesTransposedDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto solver = this->bicgstab_factory->generate(this->mtx->transpose());
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->transpose()->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({-4.0, -1.0, 4.0}), half_tol);
}


TYPED_TEST(Bicgstab, SolvesConjTransposedDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto solver = this->bicgstab_factory->generate(this->mtx->conj_transpose());
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->conj_transpose()->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({-4.0, -1.0, 4.0}), half_tol);
}


}  // namespace
