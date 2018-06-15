/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/solver/bicgstab.hpp>


#include <gtest/gtest.h>


#include <core/base/exception.hpp>
#include <core/base/executor.hpp>
#include <core/matrix/dense.hpp>
#include <core/stop/combined.hpp>
#include <core/stop/iteration.hpp>
#include <core/stop/relative_residual_norm.hpp>
#include <core/stop/time.hpp>
#include <core/test/utils.hpp>


namespace {


class Bicgstab : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using Solver = gko::solver::Bicgstab<>;

    Bicgstab()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{1.0, -3.0, 0.0}, {-4.0, 1.0, -3.0}, {2.0, -1.0, 2.0}}, exec)),
          bicgstab_factory(
              Solver::Factory::create()
                  .with_criterion(gko::stop::Combined::Factory::create(
                      exec, gko::stop::Iteration::Factory::create(exec, 8),
                      gko::stop::Time::Factory::create(exec, 6),
                      gko::stop::RelativeResidualNorm<>::Factory::create(
                          exec, 1e-15)))
                  .on_executor(exec)),
          bicgstab_factory_precision(
              gko::solver::Bicgstab<>::Factory::create()
                  .with_criterion(gko::stop::Combined::Factory::create(
                      exec, gko::stop::Iteration::Factory::create(exec, 50),
                      gko::stop::Time::Factory::create(exec, 6),
                      gko::stop::RelativeResidualNorm<>::Factory::create(
                          exec, 1e-15)))
                  .on_executor(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<gko::solver::Bicgstab<>::Factory> bicgstab_factory;
    std::unique_ptr<gko::solver::Bicgstab<>::Factory>
        bicgstab_factory_precision;
};


TEST_F(Bicgstab, SolvesDenseSystem)
{
    auto solver = bicgstab_factory->generate(mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, exec);

    solver->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({-4.0, -1.0, 4.0}), 1e-8);
}


TEST_F(Bicgstab, SolvesMultipleDenseSystems)
{
    auto solver = bicgstab_factory->generate(mtx);
    auto b =
        gko::initialize<Mtx>({{-1.0, -5.0}, {3.0, 1.0}, {1.0, -2.0}}, exec);
    auto x = gko::initialize<Mtx>({{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}, exec);

    solver->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({{-4.0, 1.0}, {-1.0, 2.0}, {4.0, -1.0}}), 1e-8);
}


TEST_F(Bicgstab, SolvesDenseSystemUsingAdvancedApply)
{
    auto solver = bicgstab_factory->generate(mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, exec);
    auto beta = gko::initialize<Mtx>({-1.0}, exec);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());


    ASSERT_MTX_NEAR(x, l({-8.5, -3.0, 6.0}), 1e-8);
}


TEST_F(Bicgstab, SolvesMultipleDenseSystemsUsingAdvancedApply)
{
    auto solver = bicgstab_factory->generate(mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, exec);
    auto beta = gko::initialize<Mtx>({-1.0}, exec);
    auto b =
        gko::initialize<Mtx>({{-1.0, -5.0}, {3.0, 1.0}, {1.0, -2.0}}, exec);
    auto x = gko::initialize<Mtx>({{0.5, 1.0}, {1.0, 2.0}, {2.0, 3.0}}, exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());


    ASSERT_MTX_NEAR(x, l({{-8.5, 1.0}, {-3.0, 2.0}, {6.0, -5.0}}), 1e-8);
}


// The following test-data was generated and validated with MATLAB
TEST_F(Bicgstab, SolvesBigDenseSystemForDivergenceCheck1)
{
    std::shared_ptr<Mtx> locmtx =
        gko::initialize<Mtx>({{-19.0, 47.0, -41.0, 35.0, -21.0, 71.0},
                              {-8.0, -66.0, 29.0, -96.0, -95.0, -14.0},
                              {-93.0, -58.0, -9.0, -87.0, 15.0, 35.0},
                              {60.0, -86.0, 54.0, -40.0, -93.0, 56.0},
                              {53.0, 94.0, -54.0, 86.0, -61.0, 4.0},
                              {-42.0, 57.0, 32.0, 89.0, 89.0, -39.0}},
                             exec);
    auto solver = bicgstab_factory_precision->generate(locmtx);
    auto b = gko::initialize<Mtx>({0.0, -9.0, -2.0, 8.0, -5.0, -6.0}, exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, exec);

    solver->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(
        x,
        l({0.13853406350816114, -0.08147485210505287, -0.0450299311807042,
           -0.0051264177562865719, 0.11609654300797841, 0.1018688746740561}),
        1e-9);
}


TEST_F(Bicgstab, SolvesBigDenseSystemForDivergenceCheck2)
{
    std::shared_ptr<Mtx> locmtx =
        gko::initialize<Mtx>({{-19.0, 47.0, -41.0, 35.0, -21.0, 71.0},
                              {-8.0, -66.0, 29.0, -96.0, -95.0, -14.0},
                              {-93.0, -58.0, -9.0, -87.0, 15.0, 35.0},
                              {60.0, -86.0, 54.0, -40.0, -93.0, 56.0},
                              {53.0, 94.0, -54.0, 86.0, -61.0, 4.0},
                              {-42.0, 57.0, 32.0, 89.0, 89.0, -39.0}},
                             exec);
    auto solver = bicgstab_factory_precision->generate(locmtx);
    auto b = gko::initialize<Mtx>({9.0, -4.0, -6.0, -10.0, 1.0, 10.0}, exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, exec);

    solver->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(
        x,
        l({0.13517641417299162, 0.75117689075221139, 0.47572853185155239,
           -0.50927993095367852, 0.13463333820848167, 0.23126768306576015}),
        1e-9);
}


double infNorm(gko::matrix::Dense<> *mat, size_t col = 0)
{
    using std::abs;
    double norm = 0.0;
    for (size_t i = 0; i < mat->get_size().num_rows; ++i) {
        double absEntry = abs(mat->at(i, col));
        if (norm < absEntry) norm = absEntry;
    }
    return norm;
}


TEST_F(Bicgstab, SolvesMultipleDenseSystemsDivergenceCheck)
{
    std::shared_ptr<Mtx> locmtx =
        gko::initialize<Mtx>({{-19.0, 47.0, -41.0, 35.0, -21.0, 71.0},
                              {-8.0, -66.0, 29.0, -96.0, -95.0, -14.0},
                              {-93.0, -58.0, -9.0, -87.0, 15.0, 35.0},
                              {60.0, -86.0, 54.0, -40.0, -93.0, 56.0},
                              {53.0, 94.0, -54.0, 86.0, -61.0, 4.0},
                              {-42.0, 57.0, 32.0, 89.0, 89.0, -39.0}},
                             exec);
    auto solver = bicgstab_factory_precision->generate(locmtx);
    auto b1 = gko::initialize<Mtx>({0.0, -9.0, -2.0, 8.0, -5.0, -6.0}, exec);
    auto x1 = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, exec);
    auto b2 = gko::initialize<Mtx>({9.0, -4.0, -6.0, -10.0, 1.0, 10.0}, exec);
    auto x2 = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, exec);
    auto bc = gko::initialize<Mtx>(
        {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}}, exec);
    auto xc = gko::initialize<Mtx>(
        {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}}, exec);
    for (size_t i = 0; i < xc->get_size().num_rows; ++i) {
        bc->at(i, 0) = b1->at(i);
        bc->at(i, 1) = b2->at(i);
        xc->at(i, 0) = x1->at(i);
        xc->at(i, 1) = x2->at(i);
    }

    solver->apply(b1.get(), x1.get());
    solver->apply(b2.get(), x2.get());
    solver->apply(bc.get(), xc.get());
    auto testMtx = gko::initialize<Mtx>(
        {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}}, exec);

    for (size_t i = 0; i < testMtx->get_size().num_rows; ++i) {
        testMtx->at(i, 0) = x1->at(i);
        testMtx->at(i, 1) = x2->at(i);
    }

    auto alpha = gko::initialize<Mtx>({1.0}, exec);
    auto beta = gko::initialize<Mtx>({-1.0}, exec);
    auto residual1 = gko::initialize<Mtx>({0.}, exec);
    residual1->copy_from(b1->clone());
    auto residual2 = gko::initialize<Mtx>({0.}, exec);
    residual2->copy_from(b2->clone());
    auto residualC = gko::initialize<Mtx>({0.}, exec);
    residualC->copy_from(bc->clone());

    locmtx->apply(alpha.get(), x1.get(), beta.get(), residual1.get());
    locmtx->apply(alpha.get(), x2.get(), beta.get(), residual2.get());
    locmtx->apply(alpha.get(), xc.get(), beta.get(), residualC.get());

    double normS1 = infNorm(residual1.get());
    double normS2 = infNorm(residual2.get());
    double normC1 = infNorm(residualC.get(), 0);
    double normC2 = infNorm(residualC.get(), 1);
    double normB1 = infNorm(bc.get(), 0);
    double normB2 = infNorm(bc.get(), 1);

    // make sure that all combined solutions are as good or better than the
    // single solutions
    ASSERT_LE(normC1 / normB1, normS1 / normB1 + 1e-12);
    ASSERT_LE(normC2 / normB2, normS2 / normB2 + 1e-12);

    // Not sure if this is necessary, the assertions above should cover what is
    // needed.
    ASSERT_MTX_NEAR(xc, testMtx, 1e-14);
}


}  // namespace
