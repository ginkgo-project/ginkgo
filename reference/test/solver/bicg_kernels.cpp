// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/bicg.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/solver/bicg_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename T>
class Bicg : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Bicg<value_type>;
    Bicg()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          stopped{},
          non_stopped{},
          bicg_factory(Solver::build()
                           .with_criteria(
                               gko::stop::Iteration::build().with_max_iters(4u),
                               gko::stop::Time::build().with_time_limit(
                                   std::chrono::seconds(6)),
                               gko::stop::ResidualNorm<value_type>::build()
                                   .with_reduction_factor(r<value_type>::value))
                           .on(exec)),
          mtx_big(gko::initialize<Mtx>(
              {{8828.0, 2673.0, 4150.0, -3139.5, 3829.5, 5856.0},
               {2673.0, 10765.5, 1805.0, 73.0, 1966.0, 3919.5},
               {4150.0, 1805.0, 6472.5, 2656.0, 2409.5, 3836.5},
               {-3139.5, 73.0, 2656.0, 6048.0, 665.0, -132.0},
               {3829.5, 1966.0, 2409.5, 665.0, 4240.5, 4373.5},
               {5856.0, 3919.5, 3836.5, -132.0, 4373.5, 5678.0}},
              exec)),
          bicg_factory_big(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(100u),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value))
                  .on(exec)),
          bicg_factory_big2(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(100u),
                      gko::stop::ImplicitResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value))
                  .on(exec)),
          mtx_non_symmetric(gko::initialize<Mtx>(
              {{1.0, 2.0, 3.0}, {3.0, 2.0, -1.0}, {0.0, -1.0, 2}}, exec))


    {
        auto small_size = gko::dim<2>{2, 2};
        auto small_scalar_size = gko::dim<2>{1, small_size[1]};
        small_b = Mtx::create(exec, small_size, small_size[1] + 1);
        small_x = Mtx::create(exec, small_size, small_size[1] + 2);
        small_one = Mtx::create(exec, small_size);
        small_zero = Mtx::create(exec, small_size);
        small_prev_rho = Mtx::create(exec, small_scalar_size);
        small_rho = Mtx::create(exec, small_scalar_size);
        small_beta = Mtx::create(exec, small_scalar_size);
        small_zero->fill(0);
        small_one->fill(1);
        small_r = small_zero->clone();
        small_z = small_zero->clone();
        small_p = small_zero->clone();
        small_q = small_zero->clone();
        small_r2 = small_zero->clone();
        small_z2 = small_zero->clone();
        small_p2 = small_zero->clone();
        small_q2 = small_zero->clone();
        small_stop = gko::array<gko::stopping_status>(exec, small_size[1]);
        stopped.stop(1);
        non_stopped.reset();
        std::fill_n(small_stop.get_data(), small_stop.get_size(), non_stopped);
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> mtx_big;
    std::shared_ptr<Mtx> mtx_non_symmetric;
    std::unique_ptr<Mtx> small_one;
    std::unique_ptr<Mtx> small_zero;
    std::unique_ptr<Mtx> small_prev_rho;
    std::unique_ptr<Mtx> small_beta;
    std::unique_ptr<Mtx> small_rho;
    std::unique_ptr<Mtx> small_x;
    std::unique_ptr<Mtx> small_b;
    std::unique_ptr<Mtx> small_r;
    std::unique_ptr<Mtx> small_z;
    std::unique_ptr<Mtx> small_p;
    std::unique_ptr<Mtx> small_q;
    std::unique_ptr<Mtx> small_r2;
    std::unique_ptr<Mtx> small_z2;
    std::unique_ptr<Mtx> small_p2;
    std::unique_ptr<Mtx> small_q2;
    gko::array<gko::stopping_status> small_stop;
    gko::stopping_status stopped;
    gko::stopping_status non_stopped;
    std::unique_ptr<typename Solver::Factory> bicg_factory;
    std::unique_ptr<typename Solver::Factory> bicg_factory_big;
    std::unique_ptr<typename Solver::Factory> bicg_factory_big2;
    std::unique_ptr<typename Solver::Factory> bicg_factory_non_symmetric;
};

TYPED_TEST_SUITE(Bicg, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Bicg, KernelInitialize)
{
    this->small_b->fill(2);
    this->small_r->fill(0);
    this->small_z->fill(1);
    this->small_p->fill(1);
    this->small_q->fill(1);
    this->small_r2->fill(0);
    this->small_z2->fill(1);
    this->small_p2->fill(1);
    this->small_q2->fill(1);
    this->small_prev_rho->fill(0);
    this->small_rho->fill(1);
    std::fill_n(this->small_stop.get_data(), this->small_stop.get_size(),
                this->stopped);

    gko::kernels::reference::bicg::initialize(
        this->exec, this->small_b.get(), this->small_r.get(),
        this->small_z.get(), this->small_p.get(), this->small_q.get(),
        this->small_prev_rho.get(), this->small_rho.get(), this->small_r2.get(),
        this->small_z2.get(), this->small_p2.get(), this->small_q2.get(),
        &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_r, this->small_b, 0);
    GKO_ASSERT_MTX_NEAR(this->small_z, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_p, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_q, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_r2, this->small_b, 0);
    GKO_ASSERT_MTX_NEAR(this->small_z2, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_p2, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_q2, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_rho, l({{0.0, 0.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_prev_rho, l({{1.0, 1.0}}), 0);
    ASSERT_EQ(this->small_stop.get_data()[0], this->non_stopped);
    ASSERT_EQ(this->small_stop.get_data()[1], this->non_stopped);
}


TYPED_TEST(Bicg, KernelStep1)
{
    this->small_p->fill(3);
    this->small_z->fill(-2);
    this->small_p2->fill(3);
    this->small_z2->fill(-2);
    this->small_rho->at(0) = 2;
    this->small_rho->at(1) = 3;
    this->small_prev_rho->at(0) = 8;
    this->small_prev_rho->at(1) = 3;
    this->small_stop.get_data()[1] = this->stopped;

    gko::kernels::reference::bicg::step_1(
        this->exec, this->small_p.get(), this->small_z.get(),
        this->small_p2.get(), this->small_z2.get(), this->small_rho.get(),
        this->small_prev_rho.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_p, l({{-1.25, 3.0}, {-1.25, 3.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_p2, l({{-1.25, 3.0}, {-1.25, 3.0}}), 0);
}


TYPED_TEST(Bicg, KernelStep1DivByZero)
{
    this->small_p->fill(3);
    this->small_z->fill(-2);
    this->small_p2->fill(3);
    this->small_z2->fill(-2);
    this->small_rho->fill(1);
    this->small_prev_rho->fill(0);

    gko::kernels::reference::bicg::step_1(
        this->exec, this->small_p.get(), this->small_z.get(),
        this->small_p2.get(), this->small_z2.get(), this->small_rho.get(),
        this->small_prev_rho.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_p, l({{-2.0, -2.0}, {-2.0, -2.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_p2, l({{-2.0, -2.0}, {-2.0, -2.0}}), 0);
}


TYPED_TEST(Bicg, KernelStep2)
{
    this->small_x->fill(-2);
    this->small_p->fill(3);
    this->small_r->fill(4);
    this->small_q->fill(-5);
    this->small_r2->fill(4);
    this->small_q2->fill(-5);
    this->small_rho->at(0) = 2;
    this->small_rho->at(1) = 3;
    this->small_beta->at(0) = 8;
    this->small_beta->at(1) = 3;
    this->small_stop.get_data()[1] = this->stopped;

    gko::kernels::reference::bicg::step_2(
        this->exec, this->small_x.get(), this->small_r.get(),
        this->small_r2.get(), this->small_p.get(), this->small_q.get(),
        this->small_q2.get(), this->small_beta.get(), this->small_rho.get(),
        &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_x, l({{-1.25, -2.0}, {-1.25, -2.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_r, l({{5.25, 4.0}, {5.25, 4.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_r2, l({{5.25, 4.0}, {5.25, 4.0}}), 0);
}


TYPED_TEST(Bicg, KernelStep2DivByZero)
{
    this->small_x->fill(-2);
    this->small_p->fill(3);
    this->small_r->fill(4);
    this->small_q->fill(-5);
    this->small_r2->fill(4);
    this->small_q2->fill(-5);
    this->small_rho->fill(1);
    this->small_beta->fill(0);

    gko::kernels::reference::bicg::step_2(
        this->exec, this->small_x.get(), this->small_r.get(),
        this->small_r2.get(), this->small_p.get(), this->small_q.get(),
        this->small_q2.get(), this->small_beta.get(), this->small_rho.get(),
        &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_x, l({{-2.0, -2.0}, {-2.0, -2.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_r, l({{4.0, 4.0}, {4.0, 4.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_r2, l({{4.0, 4.0}, {4.0, 4.0}}), 0);
}


TYPED_TEST(Bicg, SolvesStencilSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->bicg_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value);
}


TYPED_TEST(Bicg, SolvesStencilSystemMixed)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->bicg_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Bicg, SolvesStencilSystemComplex)
{
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->bicg_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {value_type{-1.0, 2.0}, value_type{3.0, -6.0}, value_type{1.0, -2.0}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.0, 0.0}, value_type{0.0, 0.0}, value_type{0.0, 0.0}},
        this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.0, -2.0}, value_type{3.0, -6.0},
                           value_type{2.0, -4.0}}),
                        r<value_type>::value);
}


TYPED_TEST(Bicg, SolvesStencilSystemMixedComplex)
{
    using value_type =
        gko::to_complex<gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->bicg_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {value_type{-1.0, 2.0}, value_type{3.0, -6.0}, value_type{1.0, -2.0}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.0, 0.0}, value_type{0.0, 0.0}, value_type{0.0, 0.0}},
        this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.0, -2.0}, value_type{3.0, -6.0},
                           value_type{2.0, -4.0}}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Bicg, SolvesMultipleStencilSystems)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto solver = this->bicg_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {I<T>{-1.0, 1.0}, I<T>{3.0, 0.0}, I<T>{1.0, 1.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({{1.0, 1.0}, {3.0, 1.0}, {2.0, 1.0}}),
                        r<value_type>::value);
}


TYPED_TEST(Bicg, SolvesStencilSystemUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->bicg_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.5, 5.0, 2.0}), r<value_type>::value);
}


TYPED_TEST(Bicg, SolvesStencilSystemUsingAdvancedApplyMixed)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->bicg_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.5, 5.0, 2.0}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Bicg, SolvesStencilSystemUsingAdvancedApplyComplex)
{
    using Scalar = typename TestFixture::Mtx;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->bicg_factory->generate(this->mtx);
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
                        l({value_type{1.5, -3.0}, value_type{5.0, -10.0},
                           value_type{2.0, -4.0}}),
                        r<value_type>::value);
}


TYPED_TEST(Bicg, SolvesStencilSystemUsingAdvancedApplyMixedComplex)
{
    using Scalar = gko::matrix::Dense<
        gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->bicg_factory->generate(this->mtx);
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
                        l({value_type{1.5, -3.0}, value_type{5.0, -10.0},
                           value_type{2.0, -4.0}}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Bicg, SolvesMultipleStencilSystemsUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto solver = this->bicg_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>(
        {I<T>{-1.0, 1.0}, I<T>{3.0, 0.0}, I<T>{1.0, 1.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.5, 1.0}, I<T>{1.0, 2.0}, I<T>{2.0, 3.0}}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({{1.5, 1.0}, {5.0, 0.0}, {2.0, -1.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Bicg, SolvesBigDenseSystem1)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->bicg_factory_big->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {1300083.0, 1018120.5, 906410.0, -42679.5, 846779.5, 1176858.5},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({81.0, 55.0, 45.0, 5.0, 85.0, -10.0}),
                        r<value_type>::value * 1e2);
}


TYPED_TEST(Bicg, SolvesBigDenseSystem2)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->bicg_factory_big->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {886630.5, -172578.0, 684522.0, -65310.5, 455487.5, 607436.0},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({33.0, -56.0, 81.0, -30.0, 21.0, 40.0}),
                        r<value_type>::value * 1e2);
}


TYPED_TEST(Bicg, SolvesBigDenseSystemImplicitResNormCrit)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->bicg_factory_big2->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {886630.5, -172578.0, 684522.0, -65310.5, 455487.5, 607436.0},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({33.0, -56.0, 81.0, -30.0, 21.0, 40.0}),
                        r<value_type>::value * 1e2);
}


TYPED_TEST(Bicg, SolvesNonSymmetricStencilSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->bicg_factory->generate(this->mtx_non_symmetric);
    auto b = gko::initialize<Mtx>({13.0, 7.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e2);
}


TYPED_TEST(Bicg, SolvesMultipleDenseSystemForDivergenceCheck)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->bicg_factory_big->generate(this->mtx_big);
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

    solver->apply(b1, x1);
    solver->apply(b2, x2);
    solver->apply(bc, xc);
    auto mergedRes = Mtx::create(this->exec, gko::dim<2>{b1->get_size()[0], 2});
    for (size_t i = 0; i < mergedRes->get_size()[0]; ++i) {
        mergedRes->at(i, 0) = x1->at(i);
        mergedRes->at(i, 1) = x2->at(i);
    }

    auto alpha = gko::initialize<Mtx>({1.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);

    auto residual1 = Mtx::create(this->exec, b1->get_size());
    residual1->copy_from(b1);
    auto residual2 = Mtx::create(this->exec, b2->get_size());
    residual2->copy_from(b2);
    auto residualC = Mtx::create(this->exec, bc->get_size());
    residualC->copy_from(bc);

    this->mtx_big->apply(alpha, x1, beta, residual1);
    this->mtx_big->apply(alpha, x2, beta, residual2);
    this->mtx_big->apply(alpha, xc, beta, residualC);

    auto normS1 = inf_norm(residual1);
    auto normS2 = inf_norm(residual2);
    auto normC1 = inf_norm(residualC, 0);
    auto normC2 = inf_norm(residualC, 1);
    auto normB1 = inf_norm(b1);
    auto normB2 = inf_norm(b2);

    // make sure that all combined solutions are as good or better than the
    // single solutions
    ASSERT_LE(normC1 / normB1, normS1 / normB1 + r<value_type>::value);
    ASSERT_LE(normC2 / normB2, normS2 / normB2 + r<value_type>::value);

    // Not sure if this is necessary, the assertions above should cover what
    // is needed.
    GKO_ASSERT_MTX_NEAR(xc, mergedRes, r<value_type>::value);
}


TYPED_TEST(Bicg, SolvesTransposedNonSymmetricStencilSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver =
        this->bicg_factory->generate(this->mtx_non_symmetric->transpose());
    auto b = gko::initialize<Mtx>({13.0, 7.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e2);
}


TYPED_TEST(Bicg, SolvesConjTransposedNonSymmetricStencilSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver =
        this->bicg_factory->generate(this->mtx_non_symmetric->conj_transpose());
    auto b = gko::initialize<Mtx>({13.0, 7.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->conj_transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e2);
}


}  // namespace
