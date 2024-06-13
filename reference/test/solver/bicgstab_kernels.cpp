// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/bicgstab.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/solver/bicgstab_kernels.hpp"
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
          stopped{},
          finalized{},
          non_stopped{},
          bicgstab_factory(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(8u),
                      gko::stop::Time::build().with_time_limit(
                          std::chrono::seconds(6)),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value))
                  .on(exec)),
          bicgstab_factory2(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(8u),
                      gko::stop::Time::build().with_time_limit(
                          std::chrono::seconds(6)),
                      gko::stop::ImplicitResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value))
                  .on(exec)),
          bicgstab_factory_precision(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(50u),
                      gko::stop::Time::build().with_time_limit(
                          std::chrono::seconds(6)),
                      gko::stop::ResidualNorm<value_type>::build()
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
        small_omega = Mtx::create(exec, small_scalar_size);
        small_zero->fill(0);
        small_one->fill(1);
        small_r = small_zero->clone();
        small_rr = small_zero->clone();
        small_v = small_zero->clone();
        small_s = small_zero->clone();
        small_t = small_zero->clone();
        small_z = small_zero->clone();
        small_y = small_zero->clone();
        small_p = small_zero->clone();
        small_stop = gko::array<gko::stopping_status>(exec, small_size[1]);
        stopped.stop(1, false);
        finalized.stop(1, true);
        non_stopped.reset();
        std::fill_n(small_stop.get_data(), small_stop.get_size(), non_stopped);
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> small_one;
    std::unique_ptr<Mtx> small_zero;
    std::unique_ptr<Mtx> small_prev_rho;
    std::unique_ptr<Mtx> small_rho;
    std::unique_ptr<Mtx> small_alpha;
    std::unique_ptr<Mtx> small_beta;
    std::unique_ptr<Mtx> small_gamma;
    std::unique_ptr<Mtx> small_omega;
    std::unique_ptr<Mtx> small_x;
    std::unique_ptr<Mtx> small_b;
    std::unique_ptr<Mtx> small_r;
    std::unique_ptr<Mtx> small_rr;
    std::unique_ptr<Mtx> small_v;
    std::unique_ptr<Mtx> small_s;
    std::unique_ptr<Mtx> small_t;
    std::unique_ptr<Mtx> small_z;
    std::unique_ptr<Mtx> small_y;
    std::unique_ptr<Mtx> small_p;
    gko::array<gko::stopping_status> small_stop;
    gko::stopping_status stopped;
    gko::stopping_status finalized;
    gko::stopping_status non_stopped;
    std::unique_ptr<typename Solver::Factory> bicgstab_factory;
    std::unique_ptr<typename Solver::Factory> bicgstab_factory2;
    std::unique_ptr<typename Solver::Factory> bicgstab_factory_precision;
};

TYPED_TEST_SUITE(Bicgstab, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Bicgstab, KernelInitialize)
{
    this->small_b->fill(2);
    this->small_r->fill(0);
    this->small_rr->fill(1);
    this->small_v->fill(1);
    this->small_s->fill(1);
    this->small_t->fill(1);
    this->small_z->fill(1);
    this->small_y->fill(1);
    this->small_p->fill(1);
    this->small_prev_rho->fill(0);
    this->small_rho->fill(0);
    this->small_alpha->fill(0);
    this->small_beta->fill(0);
    this->small_gamma->fill(0);
    this->small_omega->fill(0);
    std::fill_n(this->small_stop.get_data(), this->small_stop.get_size(),
                this->stopped);

    gko::kernels::reference::bicgstab::initialize(
        this->exec, this->small_b.get(), this->small_r.get(),
        this->small_rr.get(), this->small_y.get(), this->small_s.get(),
        this->small_t.get(), this->small_z.get(), this->small_v.get(),
        this->small_p.get(), this->small_prev_rho.get(), this->small_rho.get(),
        this->small_alpha.get(), this->small_beta.get(),
        this->small_gamma.get(), this->small_omega.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_r, this->small_b, 0);
    GKO_ASSERT_MTX_NEAR(this->small_rr, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_v, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_s, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_t, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_z, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_y, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_p, this->small_zero, 0);
    GKO_ASSERT_MTX_NEAR(this->small_rho, l({{1.0, 1.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_prev_rho, l({{1.0, 1.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_alpha, l({{1.0, 1.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_beta, l({{1.0, 1.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_gamma, l({{1.0, 1.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_omega, l({{1.0, 1.0}}), 0);
    ASSERT_EQ(this->small_stop.get_data()[0], this->non_stopped);
    ASSERT_EQ(this->small_stop.get_data()[1], this->non_stopped);
}


TYPED_TEST(Bicgstab, KernelStep1)
{
    this->small_p->fill(3);
    this->small_r->fill(-2);
    this->small_v->fill(1);
    this->small_rho->at(0) = 2;
    this->small_rho->at(1) = 3;
    this->small_prev_rho->at(0) = 7;
    this->small_prev_rho->at(1) = 6;
    this->small_alpha->at(0) = 7;
    this->small_alpha->at(1) = 5;
    this->small_omega->at(0) = 8;
    this->small_omega->at(1) = 3;
    this->small_stop.get_data()[1] = this->stopped;

    gko::kernels::reference::bicgstab::step_1(
        this->exec, this->small_r.get(), this->small_p.get(),
        this->small_v.get(), this->small_rho.get(), this->small_prev_rho.get(),
        this->small_alpha.get(), this->small_omega.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_p, l({{-3.25, 3.0}, {-3.25, 3.0}}), 0);
}


TYPED_TEST(Bicgstab, KernelStep1DivRhoZero)
{
    this->small_p->fill(3);
    this->small_r->fill(-2);
    this->small_v->fill(1);
    this->small_rho->fill(2);
    this->small_prev_rho->fill(0);
    this->small_alpha->fill(1);
    this->small_omega->fill(1);

    gko::kernels::reference::bicgstab::step_1(
        this->exec, this->small_r.get(), this->small_p.get(),
        this->small_v.get(), this->small_rho.get(), this->small_prev_rho.get(),
        this->small_alpha.get(), this->small_omega.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_p, l({{-2.0, -2.0}, {-2.0, -2.0}}), 0);
}


TYPED_TEST(Bicgstab, KernelStep1DivOmegaZero)
{
    this->small_p->fill(3);
    this->small_r->fill(-2);
    this->small_v->fill(1);
    this->small_rho->fill(2);
    this->small_prev_rho->fill(1);
    this->small_alpha->fill(1);
    this->small_omega->fill(0);

    gko::kernels::reference::bicgstab::step_1(
        this->exec, this->small_r.get(), this->small_p.get(),
        this->small_v.get(), this->small_rho.get(), this->small_prev_rho.get(),
        this->small_alpha.get(), this->small_omega.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_p, l({{-2.0, -2.0}, {-2.0, -2.0}}), 0);
}


TYPED_TEST(Bicgstab, KernelStep1DivBothZero)
{
    this->small_p->fill(3);
    this->small_r->fill(-2);
    this->small_v->fill(1);
    this->small_rho->fill(2);
    this->small_prev_rho->fill(0);
    this->small_alpha->fill(1);
    this->small_omega->fill(0);

    gko::kernels::reference::bicgstab::step_1(
        this->exec, this->small_r.get(), this->small_p.get(),
        this->small_v.get(), this->small_rho.get(), this->small_prev_rho.get(),
        this->small_alpha.get(), this->small_omega.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_p, l({{-2.0, -2.0}, {-2.0, -2.0}}), 0);
}


TYPED_TEST(Bicgstab, KernelStep2)
{
    this->small_s->fill(5);
    this->small_r->fill(-2);
    this->small_v->fill(1);
    this->small_alpha->fill(0);
    this->small_rho->at(0) = 2;
    this->small_rho->at(1) = 3;
    this->small_beta->at(0) = 8;
    this->small_beta->at(1) = 3;
    this->small_stop.get_data()[1] = this->stopped;

    gko::kernels::reference::bicgstab::step_2(
        this->exec, this->small_r.get(), this->small_s.get(),
        this->small_v.get(), this->small_rho.get(), this->small_alpha.get(),
        this->small_beta.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_s, l({{-2.25, 5.0}, {-2.25, 5.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_alpha, l({{0.25, 0.0}}), 0);
}


TYPED_TEST(Bicgstab, KernelStep2DivZero)
{
    this->small_s->fill(5);
    this->small_r->fill(-2);
    this->small_v->fill(1);
    this->small_alpha->fill(4);
    this->small_rho->fill(1);
    this->small_beta->fill(0);

    gko::kernels::reference::bicgstab::step_2(
        this->exec, this->small_r.get(), this->small_s.get(),
        this->small_v.get(), this->small_rho.get(), this->small_alpha.get(),
        this->small_beta.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_s, l({{-2.0, -2.0}, {-2.0, -2.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_alpha, l({{0.0, 0.0}}), 0);
}


TYPED_TEST(Bicgstab, KernelStep3)
{
    this->small_x->fill(5);
    this->small_r->fill(-2);
    this->small_s->fill(1);
    this->small_y->fill(4);
    this->small_z->fill(-6);
    this->small_t->fill(7);
    this->small_omega->fill(10);
    this->small_beta->at(0) = 2;
    this->small_beta->at(1) = 3;
    this->small_gamma->at(0) = 8;
    this->small_gamma->at(1) = 3;
    this->small_alpha->at(0) = 1;
    this->small_alpha->at(1) = -2;
    this->small_stop.get_data()[1] = this->stopped;

    gko::kernels::reference::bicgstab::step_3(
        this->exec, this->small_x.get(), this->small_r.get(),
        this->small_s.get(), this->small_t.get(), this->small_y.get(),
        this->small_z.get(), this->small_alpha.get(), this->small_beta.get(),
        this->small_gamma.get(), this->small_omega.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_x, l({{-15.0, 5.0}, {-15.0, 5.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_r, l({{-27.0, -2.0}, {-27.0, -2.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_omega, l({{4.0, 10.0}}), 0);
}


TYPED_TEST(Bicgstab, KernelStep3DivZero)
{
    this->small_x->fill(5);
    this->small_r->fill(-2);
    this->small_s->fill(1);
    this->small_y->fill(4);
    this->small_z->fill(-6);
    this->small_t->fill(7);
    this->small_omega->fill(10);
    this->small_beta->fill(0);
    this->small_gamma->at(0) = 8;
    this->small_gamma->at(1) = 3;
    this->small_alpha->at(0) = 1;
    this->small_alpha->at(1) = -2;

    gko::kernels::reference::bicgstab::step_3(
        this->exec, this->small_x.get(), this->small_r.get(),
        this->small_s.get(), this->small_t.get(), this->small_y.get(),
        this->small_z.get(), this->small_alpha.get(), this->small_beta.get(),
        this->small_gamma.get(), this->small_omega.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_x, l({{9.0, -3.0}, {9.0, -3.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_omega, l({{0.0, 0.0}}), 0);
}


TYPED_TEST(Bicgstab, KernelFinalize)
{
    this->small_x->fill(5);
    this->small_y->fill(4);
    this->small_alpha->at(0) = 1;
    this->small_alpha->at(1) = -2;
    this->small_stop.get_data()[0] = this->stopped;
    this->small_stop.get_data()[1] = this->finalized;

    gko::kernels::reference::bicgstab::finalize(
        this->exec, this->small_x.get(), this->small_y.get(),
        this->small_alpha.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_x, l({{9.0, 5.0}, {9.0, 5.0}}), 0);
    ASSERT_EQ(this->small_stop.get_data()[0], this->finalized);
    ASSERT_EQ(this->small_stop.get_data()[1], this->finalized);
}


TYPED_TEST(Bicgstab, SolvesDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->bicgstab_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-4.0, -1.0, 4.0}), r<value_type>::value);
}


TYPED_TEST(Bicgstab, SolvesDenseSystemMixed)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->bicgstab_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-4.0, -1.0, 4.0}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Bicgstab, SolvesDenseSystemComplex)
{
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->bicgstab_factory->generate(this->mtx);
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
                        r<value_type>::value);
}


TYPED_TEST(Bicgstab, SolvesDenseSystemMixedComplex)
{
    using value_type =
        gko::to_complex<gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->bicgstab_factory->generate(this->mtx);
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
                        (r_mixed<value_type, TypeParam>()));
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

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({{-4.0, 1.0}, {-1.0, 2.0}, {4.0, -1.0}}),
                        half_tol);
}


TYPED_TEST(Bicgstab, SolvesMultipleDenseSystemsWithImplicitResNormCrit)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto solver = this->bicgstab_factory2->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {I<T>{-1.0, -5.0}, I<T>{3.0, 1.0}, I<T>{1.0, -2.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({{-4.0, 1.0}, {-1.0, 2.0}, {4.0, -1.0}}),
                        half_tol);
}


TYPED_TEST(Bicgstab, SolvesDenseSystemUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->bicgstab_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({-8.5, -3.0, 6.0}), r<value_type>::value);
}


TYPED_TEST(Bicgstab, SolvesDenseSystemUsingAdvancedApplyMixed)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->bicgstab_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({-8.5, -3.0, 6.0}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Bicgstab, SolvesDenseSystemUsingAdvancedApplyComplex)
{
    using Scalar = typename TestFixture::Mtx;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->bicgstab_factory->generate(this->mtx);
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
                        r<value_type>::value);
}


TYPED_TEST(Bicgstab, SolvesDenseSystemUsingAdvancedApplyMixedComplex)
{
    using Scalar = gko::matrix::Dense<
        gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->bicgstab_factory->generate(this->mtx);
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
                        (r_mixed<value_type, TypeParam>()));
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

    solver->apply(alpha, b, beta, x);

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

    solver->apply(b, x);

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

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({0.13517641417299162, 0.75117689075221139, 0.47572853185155239,
           -0.50927993095367852, 0.13463333820848167, 0.23126768306576015}),
        half_tol * 1e-1);
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

    solver->transpose()->apply(b, x);

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

    solver->conj_transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-4.0, -1.0, 4.0}), half_tol);
}


}  // namespace
