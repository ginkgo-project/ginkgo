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

#include <core/solver/cg.hpp>


#include <gtest/gtest.h>


#include <core/base/exception.hpp>
#include <core/base/executor.hpp>
#include <core/matrix/dense.hpp>
#include <core/stop/combined.hpp>
#include <core/stop/iteration.hpp>
#include <core/stop/residual_norm_reduction.hpp>
#include <core/stop/time.hpp>
#include <core/test/utils/assertions.hpp>


namespace {


class Cg : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    Cg()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          cg_factory(
              gko::solver::Cg<>::Factory::create()
                  .with_criterion(
                      gko::stop::Combined::Factory::create()
                          .with_criteria(
                              gko::stop::Iteration::Factory::create()
                                  .with_max_iters(4u)
                                  .on_executor(exec),
                              gko::stop::Time::Factory::create()
                                  .with_time_limit(std::chrono::seconds(6))
                                  .on_executor(exec),
                              gko::stop::ResidualNormReduction<>::Factory::
                                  create()
                                      .with_reduction_factor(1e-15)
                                      .on_executor(exec))
                          .on_executor(exec))
                  .on_executor(exec)),
          mtx_big(gko::initialize<Mtx>(
              {{8828.0, 2673.0, 4150.0, -3139.5, 3829.5, 5856.0},
               {2673.0, 10765.5, 1805.0, 73.0, 1966.0, 3919.5},
               {4150.0, 1805.0, 6472.5, 2656.0, 2409.5, 3836.5},
               {-3139.5, 73.0, 2656.0, 6048.0, 665.0, -132.0},
               {3829.5, 1966.0, 2409.5, 665.0, 4240.5, 4373.5},
               {5856.0, 3919.5, 3836.5, -132.0, 4373.5, 5678.0}},
              exec)),
          cg_factory_big(
              gko::solver::Cg<>::Factory::create()
                  .with_criterion(
                      gko::stop::Combined::Factory::create()
                          .with_criteria(gko::stop::Iteration::Factory::create()
                                             .with_max_iters(100u)
                                             .on_executor(exec),
                                         gko::stop::ResidualNormReduction<>::
                                             Factory::create()
                                                 .with_reduction_factor(1e-15)
                                                 .on_executor(exec))
                          .on_executor(exec))
                  .on_executor(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> mtx_big;
    std::unique_ptr<gko::solver::Cg<>::Factory> cg_factory;
    std::unique_ptr<gko::solver::Cg<>::Factory> cg_factory_big;
};


TEST_F(Cg, SolvesStencilSystem)
{
    auto solver = cg_factory->generate(mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, exec);

    solver->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), 1e-14);
}


TEST_F(Cg, SolvesMultipleStencilSystems)
{
    auto solver = cg_factory->generate(mtx);
    auto b = gko::initialize<Mtx>({{-1.0, 1.0}, {3.0, 0.0}, {1.0, 1.0}}, exec);
    auto x = gko::initialize<Mtx>({{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}, exec);

    solver->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({{1.0, 1.0}, {3.0, 1.0}, {2.0, 1.0}}), 1e-14);
}


TEST_F(Cg, SolvesStencilSystemUsingAdvancedApply)
{
    auto solver = cg_factory->generate(mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, exec);
    auto beta = gko::initialize<Mtx>({-1.0}, exec);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    ASSERT_MTX_NEAR(x, l({1.5, 5.0, 2.0}), 1e-14);
}


TEST_F(Cg, SolvesMultipleStencilSystemsUsingAdvancedApply)
{
    auto solver = cg_factory->generate(mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, exec);
    auto beta = gko::initialize<Mtx>({-1.0}, exec);
    auto b = gko::initialize<Mtx>({{-1.0, 1.0}, {3.0, 0.0}, {1.0, 1.0}}, exec);
    auto x = gko::initialize<Mtx>({{0.5, 1.0}, {1.0, 2.0}, {2.0, 3.0}}, exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    ASSERT_MTX_NEAR(x, l({{1.5, 1.0}, {5.0, 0.0}, {2.0, -1.0}}), 1e-14);
}


TEST_F(Cg, SolvesBigDenseSystem1)
{
    auto solver = cg_factory_big->generate(mtx_big);
    auto b = gko::initialize<Mtx>(
        {1300083.0, 1018120.5, 906410.0, -42679.5, 846779.5, 1176858.5}, exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, exec);

    solver->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({81.0, 55.0, 45.0, 5.0, 85.0, -10.0}), 1e-10);
}


TEST_F(Cg, SolvesBigDenseSystem2)
{
    auto solver = cg_factory_big->generate(mtx_big);
    auto b = gko::initialize<Mtx>(
        {886630.5, -172578.0, 684522.0, -65310.5, 455487.5, 607436.0}, exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, exec);

    solver->apply(b.get(), x.get());

    ASSERT_MTX_NEAR(x, l({33.0, -56.0, 81.0, -30.0, 21.0, 40.0}), 1e-10);
}


double infNorm(gko::matrix::Dense<> *mat, size_t col = 0)
{
    using std::abs;
    double norm = 0.0;
    for (size_t i = 0; i < mat->get_size()[0]; ++i) {
        double absEntry = abs(mat->at(i, col));
        if (norm < absEntry) norm = absEntry;
    }
    return norm;
}


TEST_F(Cg, SolvesMultipleDenseSystemForDivergenceCheck)
{
    auto solver = cg_factory_big->generate(mtx_big);
    auto b1 = gko::initialize<Mtx>(
        {1300083.0, 1018120.5, 906410.0, -42679.5, 846779.5, 1176858.5}, exec);
    auto b2 = gko::initialize<Mtx>(
        {886630.5, -172578.0, 684522.0, -65310.5, 455487.5, 607436.0}, exec);

    auto x1 = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, exec);
    auto x2 = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, exec);

    auto bc = Mtx::create(exec, gko::dim<2>{mtx_big->get_size()[0], 2});
    auto xc = Mtx::create(exec, gko::dim<2>{mtx_big->get_size()[1], 2});
    for (size_t i = 0; i < bc->get_size()[0]; ++i) {
        bc->at(i, 0) = b1->at(i);
        bc->at(i, 1) = b2->at(i);

        xc->at(i, 0) = x1->at(i);
        xc->at(i, 1) = x2->at(i);
    }

    solver->apply(b1.get(), x1.get());
    solver->apply(b2.get(), x2.get());
    solver->apply(bc.get(), xc.get());
    auto mergedRes = Mtx::create(exec, gko::dim<2>{b1->get_size()[0], 2});
    for (size_t i = 0; i < mergedRes->get_size()[0]; ++i) {
        mergedRes->at(i, 0) = x1->at(i);
        mergedRes->at(i, 1) = x2->at(i);
    }

    auto alpha = gko::initialize<Mtx>({1.0}, exec);
    auto beta = gko::initialize<Mtx>({-1.0}, exec);

    auto residual1 = Mtx::create(exec, b1->get_size());
    residual1->copy_from(b1.get());
    auto residual2 = Mtx::create(exec, b2->get_size());
    residual2->copy_from(b2.get());
    auto residualC = Mtx::create(exec, bc->get_size());
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
