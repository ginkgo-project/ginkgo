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

#include <core/solver/cgs.hpp>


#include <gtest/gtest.h>


#include <core/base/exception.hpp>
#include <core/base/executor.hpp>
#include <core/matrix/dense.hpp>
#include <core/test/utils/assertions.hpp>


namespace {


class Cgs : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using Solver = gko::solver::Cgs<>;

    Cgs()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{1.0, -3.0, 0.0}, {-4.0, 1.0, -3.0}, {2.0, -1.0, 2.0}}, exec)),
          cgs_factory(Solver::Factory::create()
                          .with_max_iters(40)
                          .with_rel_residual_goal(1e-15)
                          .on_executor(exec)),
          mtx_big(
              gko::initialize<Mtx>({{-99.0, 87.0, -67.0, -62.0, -68.0, -19.0},
                                    {-30.0, -17.0, -1.0, 9.0, 23.0, 77.0},
                                    {80.0, 89.0, 36.0, 94.0, 55.0, 34.0},
                                    {-31.0, 21.0, 96.0, -26.0, 24.0, -57.0},
                                    {60.0, 45.0, -16.0, -4.0, 96.0, 24.0},
                                    {69.0, 32.0, -68.0, 57.0, -30.0, -51.0}},
                                   exec)),
          cgs_factory_big(gko::solver::Cgs<>::Factory::create()
                              .with_max_iters(100)
                              .with_rel_residual_goal(1e-15)
                              .on_executor(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> mtx_big;
    std::unique_ptr<gko::solver::Cgs<>::Factory> cgs_factory;
    std::unique_ptr<gko::solver::Cgs<>::Factory> cgs_factory_big;
};


TEST_F(Cgs, SolvesDenseSystem)
{
    auto solver = cgs_factory->generate(mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, exec);

    solver->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({-4.0, -1.0, 4.0}), 1e-8);
}


TEST_F(Cgs, SolvesMultipleDenseSystem)
{
    auto solver = cgs_factory->generate(mtx);
    auto b =
        gko::initialize<Mtx>({{-1.0, -5.0}, {3.0, 1.0}, {1.0, -2.0}}, exec);
    auto x = gko::initialize<Mtx>({{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}, exec);

    solver->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({{-4.0, 1.0}, {-1.0, 2.0}, {4.0, -1.0}}), 1e-8);
}


TEST_F(Cgs, SolvesDenseSystemUsingAdvancedApply)
{
    auto solver = cgs_factory->generate(mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, exec);
    auto beta = gko::initialize<Mtx>({-1.0}, exec);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    ASSERT_MTX_NEAR(x, l({-8.5, -3.0, 6.0}), 1e-8);
}


TEST_F(Cgs, SolvesMultipleDenseSystemsUsingAdvancedApply)
{
    auto solver = cgs_factory->generate(mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, exec);
    auto beta = gko::initialize<Mtx>({-1.0}, exec);
    auto b =
        gko::initialize<Mtx>({{-1.0, -5.0}, {3.0, 1.0}, {1.0, -2.0}}, exec);
    auto x = gko::initialize<Mtx>({{0.5, 1.0}, {1.0, 2.0}, {2.0, 3.0}}, exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    ASSERT_MTX_NEAR(x, l({{-8.5, 1.0}, {-3.0, 2.0}, {6.0, -5.0}}), 1e-8);
}


TEST_F(Cgs, SolvesBigDenseSystem1)
{
    auto solver = cgs_factory_big->generate(mtx_big);
    auto b = gko::initialize<Mtx>(
        {764.0, -4032.0, -11855.0, 7111.0, -12765.0, -4589}, exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, exec);

    solver->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({-13.0, -49.0, 69.0, -33.0, -82.0, -39.0}), 1e-10);
}


TEST_F(Cgs, SolvesBigDenseSystem2)
{
    auto solver = cgs_factory_big->generate(mtx_big);
    auto b = gko::initialize<Mtx>(
        {17356.0, 5466.0, 748.0, -456.0, 3434.0, -7020.0}, exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, exec);

    solver->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({-58.0, 98.0, -16.0, -58.0, 2.0, 76.0}), 1e-10);
}


double infNorm(gko::matrix::Dense<> *mat, size_t col = 0)
{
    using std::abs;
    double norm = 0.0;
    for (size_t i = 0; i < mat->get_dimensions().num_rows; ++i) {
        double absEntry = abs(mat->at(i, col));
        if (norm < absEntry) norm = absEntry;
    }
    return norm;
}


TEST_F(Cgs, SolvesMultipleDenseSystems)
{
    auto solver = cgs_factory_big->generate(mtx_big);
    auto b1 = gko::initialize<Mtx>(
        {764.0, -4032.0, -11855.0, 7111.0, -12765.0, -4589}, exec);
    auto b2 = gko::initialize<Mtx>(
        {17356.0, 5466.0, 748.0, -456.0, 3434.0, -7020.0}, exec);

    auto x1 = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, exec);
    auto x2 = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, exec);

    auto bc = Mtx::create(exec, mtx_big->get_dimensions().num_rows, 2);
    auto xc = Mtx::create(exec, mtx_big->get_dimensions().num_cols, 2);
    for (size_t i = 0; i < bc->get_dimensions().num_rows; ++i) {
        bc->at(i, 0) = b1->at(i);
        bc->at(i, 1) = b2->at(i);

        xc->at(i, 0) = x1->at(i);
        xc->at(i, 1) = x2->at(i);
    }

    solver->apply(b1.get(), x1.get());
    solver->apply(b2.get(), x2.get());
    solver->apply(bc.get(), xc.get());
    auto mergedRes = Mtx::create(exec, b1->get_dimensions().num_rows, 2);
    for (size_t i = 0; i < mergedRes->get_dimensions().num_rows; ++i) {
        mergedRes->at(i, 0) = x1->at(i);
        mergedRes->at(i, 1) = x2->at(i);
    }

    auto alpha = gko::initialize<Mtx>({1.0}, exec);
    auto beta = gko::initialize<Mtx>({-1.0}, exec);

    auto residual1 = Mtx::create(exec, b1->get_dimensions().num_rows,
                                 b1->get_dimensions().num_cols);
    residual1->copy_from(b1.get());
    auto residual2 = Mtx::create(exec, b2->get_dimensions().num_rows,
                                 b2->get_dimensions().num_cols);
    residual2->copy_from(b2.get());
    auto residualC = Mtx::create(exec, bc->get_dimensions().num_rows,
                                 bc->get_dimensions().num_cols);
    residualC->copy_from(bc.get());

    mtx_big->apply(alpha.get(), x1.get(), beta.get(), residual1.get());
    mtx_big->apply(alpha.get(), x2.get(), beta.get(), residual2.get());
    mtx_big->apply(alpha.get(), xc.get(), beta.get(), residualC.get());

    double normS1 = infNorm(residual1.get());
    double normS2 = infNorm(residual2.get());
    double normC1 = infNorm(residualC.get(), 0);
    double normC2 = infNorm(residualC.get(), 1);
    double normB1 = infNorm(b1.get());
    double normB2 = infNorm(b2.get());

    // make sure that all combined solutions are as good or better than the
    // single solutions
    ASSERT_LE(normC1 / normB1, normS1 / normB1 + 1e-14);
    ASSERT_LE(normC2 / normB2, normS2 / normB2 + 1e-14);

    // Not sure if this is necessary, the assertions above should cover what is
    // needed.
    ASSERT_MTX_NEAR(xc, mergedRes, 1e-14);
}


}  // namespace
