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

#include <ginkgo/core/solver/gmres.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>
#include <ginkgo/core/stop/time.hpp>


#include <core/test/utils.hpp>


namespace {


template <typename T>
class Gmres : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Gmres<value_type>;
    Gmres()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{1.0, 2.0, 3.0}, {3.0, 2.0, -1.0}, {0.0, -1.0, 2}}, exec)),
          gmres_factory(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(4u).on(exec),
                      gko::stop::Time::build()
                          .with_time_limit(std::chrono::seconds(6))
                          .on(exec),
                      gko::stop::ResidualNormReduction<value_type>::build()
                          .with_reduction_factor(r<value_type>::value)
                          .on(exec))
                  .on(exec)),
          mtx_big(gko::initialize<Mtx>(
              {{2295.7, -764.8, 1166.5, 428.9, 291.7, -774.5},
               {2752.6, -1127.7, 1212.8, -299.1, 987.7, 786.8},
               {138.3, 78.2, 485.5, -899.9, 392.9, 1408.9},
               {-1907.1, 2106.6, 1026.0, 634.7, 194.6, -534.1},
               {-365.0, -715.8, 870.7, 67.5, 279.8, 1927.8},
               {-848.1, -280.5, -381.8, -187.1, 51.2, -176.2}},
              exec)),
          gmres_factory_big(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(100u).on(
                          exec),
                      gko::stop::ResidualNormReduction<value_type>::build()
                          .with_reduction_factor(r<value_type>::value)
                          .on(exec))
                  .on(exec)),
          mtx_medium(
              gko::initialize<Mtx>({{-86.40, 153.30, -108.90, 8.60, -61.60},
                                    {7.70, -77.00, 3.30, -149.20, 74.80},
                                    {-121.40, 37.10, 55.30, -74.20, -19.20},
                                    {-111.40, -22.60, 110.10, -106.20, 88.90},
                                    {-0.70, 111.70, 154.40, 235.00, -76.50}},
                                   exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> mtx_medium;
    std::shared_ptr<Mtx> mtx_big;
    std::unique_ptr<typename Solver::Factory> gmres_factory;
    std::unique_ptr<typename Solver::Factory> gmres_factory_big;
};


TYPED_TEST_CASE(Gmres, gko::test::ValueTypes);


TYPED_TEST(Gmres, SolvesStencilSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gmres_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({13.0, 7.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Gmres, SolvesMultipleStencilSystems)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto solver = this->gmres_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {I<T>{13.0, 6.0}, I<T>{7.0, 4.0}, I<T>{1.0, 1.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({{1.0, 1.0}, {3.0, 1.0}, {2.0, 1.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Gmres, SolvesStencilSystemUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gmres_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({13.0, 7.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.5, 5.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Gmres, SolvesMultipleStencilSystemsUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto solver = this->gmres_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>(
        {I<T>{13.0, 6.0}, I<T>{7.0, 4.0}, I<T>{1.0, 1.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.5, 1.0}, I<T>{1.0, 2.0}, I<T>{2.0, 3.0}}, this->exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({{1.5, 1.0}, {5.0, 0.0}, {2.0, -1.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Gmres, SolvesBigDenseSystem1)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gmres_factory_big->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {72748.36, 297469.88, 347229.24, 36290.66, 82958.82, -80192.15},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({52.7, 85.4, 134.2, -250.0, -16.8, 35.3}),
                        r<value_type>::value * 1e3);
}


TYPED_TEST(Gmres, SolvesBigDenseSystem2)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gmres_factory_big->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {175352.10, 313410.50, 131114.10, -134116.30, 179529.30, -43564.90},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({33.0, -56.0, 81.0, -30.0, 21.0, 40.0}),
                        r<value_type>::value * 1e3);
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


TYPED_TEST(Gmres, SolvesMultipleDenseSystemForDivergenceCheck)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gmres_factory_big->generate(this->mtx_big);
    auto b1 = gko::initialize<Mtx>(
        {1300083.0, 1018120.5, 906410.0, -42679.5, 846779.5, 1176858.5},
        this->exec);
    auto b2 = gko::initialize<Mtx>(
        {886630.5, -172578.0, 684522.0, -65310.5, 455487.5, 607436.0},
        this->exec);

    auto x1 = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);
    auto x2 = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    auto bc =
        Mtx::create(this->exec, gko::dim<2>{this->mtx_big->get_size()[0], 2});
    auto xc =
        Mtx::create(this->exec, gko::dim<2>{this->mtx_big->get_size()[1], 2});
    for (size_t i = 0; i < bc->get_size()[0]; ++i) {
        bc->at(i, 0) = b1->at(i);
        bc->at(i, 1) = b2->at(i);

        xc->at(i, 0) = x1->at(i);
        xc->at(i, 1) = x2->at(i);
    }

    solver->apply(b1.get(), x1.get());
    solver->apply(b2.get(), x2.get());
    solver->apply(bc.get(), xc.get());
    auto mergedRes = Mtx::create(this->exec, gko::dim<2>{b1->get_size()[0], 2});
    for (size_t i = 0; i < mergedRes->get_size()[0]; ++i) {
        mergedRes->at(i, 0) = x1->at(i);
        mergedRes->at(i, 1) = x2->at(i);
    }

    auto alpha = gko::initialize<Mtx>({1.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);

    auto residual1 = Mtx::create(this->exec, b1->get_size());
    residual1->copy_from(b1.get());
    auto residual2 = Mtx::create(this->exec, b2->get_size());
    residual2->copy_from(b2.get());
    auto residualC = Mtx::create(this->exec, bc->get_size());
    residualC->copy_from(bc.get());

    this->mtx_big->apply(alpha.get(), x1.get(), beta.get(), residual1.get());
    this->mtx_big->apply(alpha.get(), x2.get(), beta.get(), residual2.get());
    this->mtx_big->apply(alpha.get(), xc.get(), beta.get(), residualC.get());

    auto normS1 = infNorm(residual1.get());
    auto normS2 = infNorm(residual2.get());
    auto normC1 = infNorm(residualC.get(), 0);
    auto normC2 = infNorm(residualC.get(), 1);
    auto normB1 = infNorm(b1.get());
    auto normB2 = infNorm(b2.get());

    // make sure that all combined solutions are as good or better than the
    // single solutions
    ASSERT_LE(normC1 / normB1, normS1 / normB1 + r<value_type>::value);
    ASSERT_LE(normC2 / normB2, normS2 / normB2 + r<value_type>::value);

    // Not sure if this is necessary, the assertions above should cover what is
    // needed.
    GKO_ASSERT_MTX_NEAR(xc, mergedRes, r<value_type>::value);
}


TYPED_TEST(Gmres, SolvesBigDenseSystem1WithRestart)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto gmres_factory_restart =
        Solver::build()
            .with_krylov_dim(4u)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(200u).on(
                    this->exec),
                gko::stop::ResidualNormReduction<value_type>::build()
                    .with_reduction_factor(r<value_type>::value)
                    .on(this->exec))
            .on(this->exec);
    auto solver = gmres_factory_restart->generate(this->mtx_medium);
    auto b = gko::initialize<Mtx>(
        {-13945.16, 11205.66, 16132.96, 24342.18, -10910.98}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({-140.20, -142.20, 48.80, -17.70, -19.60}),
                        half_tol * 1e2);
}


TYPED_TEST(Gmres, SolvesWithPreconditioner)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto gmres_factory_preconditioner =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(100u).on(
                    this->exec),
                gko::stop::ResidualNormReduction<value_type>::build()
                    .with_reduction_factor(r<value_type>::value)
                    .on(this->exec))
            .with_preconditioner(
                gko::preconditioner::Jacobi<value_type>::build()
                    .with_max_block_size(3u)
                    .on(this->exec))
            .on(this->exec);
    auto solver = gmres_factory_preconditioner->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {175352.10, 313410.50, 131114.10, -134116.30, 179529.30, -43564.90},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({33.0, -56.0, 81.0, -30.0, 21.0, 40.0}),
                        r<value_type>::value * 1e3);
}


}  // namespace
