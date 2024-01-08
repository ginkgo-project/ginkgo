// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/cgs.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/solver/cgs_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename T>
class Cgs : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Cgs<value_type>;

    Cgs()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{1.0, -3.0, 0.0}, {-4.0, 1.0, -3.0}, {2.0, -1.0, 2.0}}, exec)),
          stopped{},
          non_stopped{},
          cgs_factory(Solver::build()
                          .with_criteria(
                              gko::stop::Iteration::build().with_max_iters(40u),
                              gko::stop::ResidualNorm<value_type>::build()
                                  .with_reduction_factor(r<value_type>::value))
                          .on(exec)),
          mtx_big(
              gko::initialize<Mtx>({{-99.0, 87.0, -67.0, -62.0, -68.0, -19.0},
                                    {-30.0, -17.0, -1.0, 9.0, 23.0, 77.0},
                                    {80.0, 89.0, 36.0, 94.0, 55.0, 34.0},
                                    {-31.0, 21.0, 96.0, -26.0, 24.0, -57.0},
                                    {60.0, 45.0, -16.0, -4.0, 96.0, 24.0},
                                    {69.0, 32.0, -68.0, 57.0, -30.0, -51.0}},
                                   exec)),
          cgs_factory_big(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(100u),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value))
                  .on(exec)),
          cgs_factory_big2(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(100u),
                      gko::stop::ImplicitResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value))
                  .on(exec))
    {
        auto small_size = gko::dim<2>{2, 2};
        auto small_scalar_size = gko::dim<2>{1, small_size[1]};
        small_b = Mtx::create(exec, small_size, small_size[1] + 1);
        small_x = Mtx::create(exec, small_size, small_size[1] + 2);
        small_one = Mtx::create(exec, small_size);
        small_zero = Mtx::create(exec, small_size);
        small_prev_rho = Mtx::create(exec, small_scalar_size);
        small_rho = Mtx::create(exec, small_scalar_size);
        small_alpha = Mtx::create(exec, small_scalar_size);
        small_beta = Mtx::create(exec, small_scalar_size);
        small_gamma = Mtx::create(exec, small_scalar_size);
        small_zero->fill(0);
        small_one->fill(1);
        small_r = small_zero->clone();
        small_r = small_zero->clone();
        small_r_tld = small_zero->clone();
        small_p = small_zero->clone();
        small_q = small_zero->clone();
        small_u = small_zero->clone();
        small_u_hat = small_zero->clone();
        small_v = small_zero->clone();
        small_v_hat = small_zero->clone();
        small_t = small_zero->clone();
        small_stop = gko::array<gko::stopping_status>(exec, small_size[1]);
        stopped.stop(1);
        non_stopped.reset();
        std::fill_n(small_stop.get_data(), small_stop.get_size(), non_stopped);
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> mtx_big;
    std::unique_ptr<Mtx> small_one;
    std::unique_ptr<Mtx> small_zero;
    std::unique_ptr<Mtx> small_prev_rho;
    std::unique_ptr<Mtx> small_rho;
    std::unique_ptr<Mtx> small_alpha;
    std::unique_ptr<Mtx> small_beta;
    std::unique_ptr<Mtx> small_gamma;
    std::unique_ptr<Mtx> small_x;
    std::unique_ptr<Mtx> small_b;
    std::unique_ptr<Mtx> small_r;
    std::unique_ptr<Mtx> small_r_tld;
    std::unique_ptr<Mtx> small_p;
    std::unique_ptr<Mtx> small_q;
    std::unique_ptr<Mtx> small_u;
    std::unique_ptr<Mtx> small_u_hat;
    std::unique_ptr<Mtx> small_v;
    std::unique_ptr<Mtx> small_v_hat;
    std::unique_ptr<Mtx> small_t;
    gko::array<gko::stopping_status> small_stop;
    gko::stopping_status stopped;
    gko::stopping_status non_stopped;
    std::unique_ptr<typename Solver::Factory> cgs_factory;
    std::unique_ptr<typename Solver::Factory> cgs_factory_big;
    std::unique_ptr<typename Solver::Factory> cgs_factory_big2;
};

TYPED_TEST_SUITE(Cgs, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Cgs, KernelInitialize)
{
    this->small_b->fill(2);
    this->small_r->fill(0);
    this->small_r_tld->fill(0);
    this->small_p->fill(1);
    this->small_q->fill(1);
    this->small_u->fill(1);
    this->small_u_hat->fill(1);
    this->small_v_hat->fill(1);
    this->small_t->fill(1);
    this->small_prev_rho->fill(0);
    this->small_rho->fill(1);
    std::fill_n(this->small_stop.get_data(), this->small_stop.get_size(),
                this->stopped);

    gko::kernels::reference::cgs::initialize(
        this->exec, this->small_b.get(), this->small_r.get(),
        this->small_r_tld.get(), this->small_p.get(), this->small_q.get(),
        this->small_u.get(), this->small_u_hat.get(), this->small_v_hat.get(),
        this->small_t.get(), this->small_alpha.get(), this->small_beta.get(),
        this->small_gamma.get(), this->small_prev_rho.get(),
        this->small_rho.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_r, this->small_b, 0);
    GKO_ASSERT_MTX_NEAR(this->small_r_tld, this->small_b, 0);
    GKO_ASSERT_MTX_NEAR(this->small_p, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_q, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_u, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_u_hat, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_v_hat, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_t, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_rho, l({{0.0, 0.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_prev_rho, l({{1.0, 1.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_alpha, l({{1.0, 1.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_beta, l({{1.0, 1.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_gamma, l({{1.0, 1.0}}), 0);
    ASSERT_EQ(this->small_stop.get_data()[0], this->non_stopped);
    ASSERT_EQ(this->small_stop.get_data()[1], this->non_stopped);
}


TYPED_TEST(Cgs, KernelStep1)
{
    this->small_r->fill(1);
    this->small_p->fill(-2);
    this->small_q->fill(3);
    this->small_u->fill(-4);
    this->small_beta->fill(2);
    this->small_prev_rho->at(0) = 2;
    this->small_prev_rho->at(1) = 3;
    this->small_rho->at(0) = -4;
    this->small_rho->at(1) = 4;
    this->small_stop.get_data()[1] = this->stopped;

    gko::kernels::reference::cgs::step_1(
        this->exec, this->small_r.get(), this->small_u.get(),
        this->small_p.get(), this->small_q.get(), this->small_beta.get(),
        this->small_rho.get(), this->small_prev_rho.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_u, l({{-5.0, -4.0}, {-5.0, -4.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_p, l({{-19.0, -2.0}, {-19.0, -2.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_beta, l({{-2.0, 2.0}}), 0);
}


TYPED_TEST(Cgs, KernelStep1DivZero)
{
    this->small_r->fill(1);
    this->small_p->fill(-2);
    this->small_q->fill(3);
    this->small_u->fill(-4);
    this->small_beta->fill(2);
    this->small_prev_rho->fill(0);
    this->small_rho->fill(3);

    gko::kernels::reference::cgs::step_1(
        this->exec, this->small_r.get(), this->small_u.get(),
        this->small_p.get(), this->small_q.get(), this->small_beta.get(),
        this->small_rho.get(), this->small_prev_rho.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_u, l({{7.0, 7.0}, {7.0, 7.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_p, l({{5.0, 5.0}, {5.0, 5.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_beta, l({{2.0, 2.0}}), 0);
}


TYPED_TEST(Cgs, KernelStep2)
{
    this->small_q->fill(1);
    this->small_u->fill(-2);
    this->small_v_hat->fill(3);
    this->small_t->fill(-4);
    this->small_alpha->fill(2);
    this->small_gamma->at(0) = 2;
    this->small_gamma->at(1) = 3;
    this->small_rho->at(0) = -4;
    this->small_rho->at(1) = 4;
    this->small_stop.get_data()[1] = this->stopped;

    gko::kernels::reference::cgs::step_2(
        this->exec, this->small_u.get(), this->small_v_hat.get(),
        this->small_q.get(), this->small_t.get(), this->small_alpha.get(),
        this->small_rho.get(), this->small_gamma.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_q, l({{4.0, 1.0}, {4.0, 1.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_t, l({{2.0, -4.0}, {2.0, -4.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_alpha, l({{-2.0, 2.0}}), 0);
}


TYPED_TEST(Cgs, KernelStep2DivZero)
{
    this->small_q->fill(1);
    this->small_u->fill(-2);
    this->small_v_hat->fill(3);
    this->small_t->fill(-4);
    this->small_alpha->fill(2);
    this->small_gamma->fill(0);
    this->small_rho->fill(-3);

    gko::kernels::reference::cgs::step_2(
        this->exec, this->small_u.get(), this->small_v_hat.get(),
        this->small_q.get(), this->small_t.get(), this->small_alpha.get(),
        this->small_rho.get(), this->small_gamma.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_q, l({{-8.0, -8.0}, {-8.0, -8.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_t, l({{-10.0, -10.0}, {-10.0, -10.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_alpha, l({{2.0, 2.0}}), 0);
}


TYPED_TEST(Cgs, KernelStep3)
{
    this->small_r->fill(1);
    this->small_t->fill(-2);
    this->small_x->fill(3);
    this->small_u_hat->fill(-4);
    this->small_beta->fill(2);
    this->small_alpha->at(0) = 2;
    this->small_alpha->at(1) = 3;
    this->small_stop.get_data()[1] = this->stopped;

    gko::kernels::reference::cgs::step_3(
        this->exec, this->small_t.get(), this->small_u_hat.get(),
        this->small_r.get(), this->small_x.get(), this->small_alpha.get(),
        &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_r, l({{5.0, 1.0}, {5.0, 1.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_x, l({{-5.0, 3.0}, {-5.0, 3.0}}), 0);
}


TYPED_TEST(Cgs, SolvesDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->cgs_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-4.0, -1.0, 4.0}), r<value_type>::value);
}


TYPED_TEST(Cgs, SolvesDenseSystemMixed)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->cgs_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-4.0, -1.0, 4.0}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Cgs, SolvesDenseSystemComplex)
{
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->cgs_factory->generate(this->mtx);
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
                        r<value_type>::value * 1e3);
}


TYPED_TEST(Cgs, SolvesDenseSystemMixedComplex)
{
    using value_type =
        gko::to_complex<gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->cgs_factory->generate(this->mtx);
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
                        (r_mixed<value_type, TypeParam>() * 1e2));
}


TYPED_TEST(Cgs, SolvesMultipleDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto solver = this->cgs_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {I<T>{-1.0, -5.0}, I<T>{3.0, 1.0}, I<T>{1.0, -2.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({{-4.0, 1.0}, {-1.0, 2.0}, {4.0, -1.0}}),
                        half_tol);
}


TYPED_TEST(Cgs, SolvesDenseSystemUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->cgs_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({-8.5, -3.0, 6.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Cgs, SolvesDenseSystemUsingAdvancedApplyMixed)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->cgs_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({-8.5, -3.0, 6.0}),
                        (r_mixed<value_type, TypeParam>()) * 1e1);
}


TYPED_TEST(Cgs, SolvesDenseSystemUsingAdvancedApplyComplex)
{
    using Scalar = typename TestFixture::Mtx;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->cgs_factory->generate(this->mtx);
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
                        r<value_type>::value * 1e3);
}


TYPED_TEST(Cgs, SolvesDenseSystemUsingAdvancedApplyMixedComplex)
{
    using Scalar = gko::matrix::Dense<
        gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->cgs_factory->generate(this->mtx);
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
                        (r_mixed<value_type, TypeParam>()) * 1e3);
}


TYPED_TEST(Cgs, SolvesMultipleDenseSystemsUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto solver = this->cgs_factory->generate(this->mtx);
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


TYPED_TEST(Cgs, SolvesBigDenseSystem1)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->cgs_factory_big->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {764.0, -4032.0, -11855.0, 7111.0, -12765.0, -4589}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-13.0, -49.0, 69.0, -33.0, -82.0, -39.0}),
                        r<value_type>::value * 1e3);
}


TYPED_TEST(Cgs, SolvesBigDenseSystemWithImplicitResNormCrit)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->cgs_factory_big2->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {17356.0, 5466.0, 748.0, -456.0, 3434.0, -7020.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-58.0, 98.0, -16.0, -58.0, 2.0, 76.0}),
                        r<value_type>::value * 1e2);
}


TYPED_TEST(Cgs, SolvesBigDenseSystem2)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->cgs_factory_big->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {17356.0, 5466.0, 748.0, -456.0, 3434.0, -7020.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-58.0, 98.0, -16.0, -58.0, 2.0, 76.0}),
                        r<value_type>::value * 1e2);
}


TYPED_TEST(Cgs, SolvesMultipleDenseSystems)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->cgs_factory_big->generate(this->mtx_big);
    auto b1 = gko::initialize<Mtx>(
        {764.0, -4032.0, -11855.0, 7111.0, -12765.0, -4589}, this->exec);
    auto b2 = gko::initialize<Mtx>(
        {17356.0, 5466.0, 748.0, -456.0, 3434.0, -7020.0}, this->exec);

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

    // Not sure if this is necessary, the assertions above should cover what is
    // needed.
    GKO_ASSERT_MTX_NEAR(xc, mergedRes, r<value_type>::value);
}


TYPED_TEST(Cgs, SolvesTransposedBigDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->cgs_factory_big->generate(this->mtx_big->transpose());
    auto b = gko::initialize<Mtx>(
        {764.0, -4032.0, -11855.0, 7111.0, -12765.0, -4589}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-13.0, -49.0, 69.0, -33.0, -82.0, -39.0}),
                        r<value_type>::value * 1e3);
}


TYPED_TEST(Cgs, SolvesConjTransposedBigDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver =
        this->cgs_factory_big->generate(this->mtx_big->conj_transpose());
    auto b = gko::initialize<Mtx>(
        {764.0, -4032.0, -11855.0, 7111.0, -12765.0, -4589}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->conj_transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-13.0, -49.0, 69.0, -33.0, -82.0, -39.0}),
                        r<value_type>::value * 1e3);
}


}  // namespace
