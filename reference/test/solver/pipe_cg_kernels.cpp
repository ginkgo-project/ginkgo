// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/pipe_cg_kernels.hpp"

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/pipe_cg.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>

#include "core/test/utils.hpp"

template <typename T>
class PipeCg : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::PipeCg<value_type>;
    PipeCg()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          mtx_big(gko::initialize<Mtx>(
              {{8828.0, 2673.0, 4150.0, -3139.5, 3829.5, 5856.0},
               {2673.0, 10765.5, 1805.0, 73.0, 1966.0, 3919.5},
               {4150.0, 1805.0, 6472.5, 2656.0, 2409.5, 3836.5},
               {-3139.5, 73.0, 2656.0, 6048.0, 665.0, -132.0},
               {3829.5, 1966.0, 2409.5, 665.0, 4240.5, 4373.5},
               {5856.0, 3919.5, 3836.5, -132.0, 4373.5, 5678.0}},
              exec)),
          stopped{},
          non_stopped{},
          pipe_cg_factory(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(400u),
                      gko::stop::Time::build().with_time_limit(
                          std::chrono::seconds(6)),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value))
                  .on(exec)),
          pipe_cg_factory_big(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(100u),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value))
                  .on(exec)),
          pipe_cg_factory_big2(
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
        small_beta = Mtx::create(exec, small_scalar_size);
        small_delta = Mtx::create(exec, small_scalar_size);

        small_zero->fill(0);
        small_one->fill(1);
        small_x = small_zero->clone();
        small_r = small_zero->clone();
        small_z = small_zero->clone();
        small_w = small_zero->clone();
        small_m = small_zero->clone();
        small_n = small_zero->clone();
        small_f = small_zero->clone();
        small_g = small_zero->clone();
        small_p = small_zero->clone();
        small_q = small_zero->clone();
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
    std::unique_ptr<Mtx> small_beta;
    std::unique_ptr<Mtx> small_delta;
    std::unique_ptr<Mtx> small_rho;
    std::unique_ptr<Mtx> small_x;
    std::unique_ptr<Mtx> small_b;
    std::unique_ptr<Mtx> small_r;
    std::unique_ptr<Mtx> small_z;
    std::unique_ptr<Mtx> small_w;
    std::unique_ptr<Mtx> small_m;
    std::unique_ptr<Mtx> small_n;
    std::unique_ptr<Mtx> small_f;
    std::unique_ptr<Mtx> small_g;
    std::unique_ptr<Mtx> small_p;
    std::unique_ptr<Mtx> small_q;
    gko::array<gko::stopping_status> small_stop;
    gko::stopping_status stopped;
    gko::stopping_status non_stopped;
    std::unique_ptr<typename Solver::Factory> pipe_cg_factory;
    std::unique_ptr<typename Solver::Factory> pipe_cg_factory_big;
    std::unique_ptr<typename Solver::Factory> pipe_cg_factory_big2;
};

TYPED_TEST_SUITE(PipeCg, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(PipeCg, KernelInitialize1)
{
    this->small_b->fill(2);
    this->small_r->fill(0);
    this->small_prev_rho->fill(0);
    std::fill_n(this->small_stop.get_data(), this->small_stop.get_size(),
                this->stopped);

    gko::kernels::reference::pipe_cg::initialize_1(
        this->exec, this->small_b.get(), this->small_r.get(),
        this->small_prev_rho.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_r, this->small_b, 0);
    GKO_ASSERT_MTX_NEAR(this->small_prev_rho, l({{1.0, 1.0}}), 0);
    ASSERT_EQ(this->small_stop.get_data()[0], this->non_stopped);
    ASSERT_EQ(this->small_stop.get_data()[1], this->non_stopped);
}


TYPED_TEST(PipeCg, KernelInitialize2)
{
    this->small_z->fill(2);
    this->small_w->fill(8);
    this->small_m->fill(8);
    this->small_n->fill(24);
    this->small_delta->fill(32);

    gko::kernels::reference::pipe_cg::initialize_2(
        this->exec, this->small_p.get(), this->small_q.get(),
        this->small_f.get(), this->small_g.get(), this->small_beta.get(),
        this->small_z.get(), this->small_w.get(), this->small_m.get(),
        this->small_n.get(), this->small_delta.get());

    GKO_ASSERT_MTX_NEAR(this->small_p, this->small_z, 0);
    GKO_ASSERT_MTX_NEAR(this->small_q, this->small_w, 0);
    GKO_ASSERT_MTX_NEAR(this->small_f, this->small_m, 0);
    GKO_ASSERT_MTX_NEAR(this->small_g, this->small_n, 0);
    GKO_ASSERT_MTX_NEAR(this->small_beta, this->small_delta, 0);
}


TYPED_TEST(PipeCg, KernelStep1)
{
    this->small_x->fill(1);
    this->small_r->fill(2);
    this->small_z->fill(3);
    this->small_w->fill(4);
    this->small_p->fill(4);
    this->small_q->fill(3);
    this->small_f->fill(2);
    this->small_g->fill(1);
    this->small_rho->at(0) = 2;
    this->small_rho->at(1) = 3;
    this->small_beta->at(0) = 8;
    this->small_beta->at(1) = 3;
    this->small_stop.get_data()[0].reset();
    this->small_stop.get_data()[1] = this->stopped;

    gko::kernels::reference::pipe_cg::step_1(
        this->exec, this->small_x.get(), this->small_r.get(),
        this->small_z.get(), this->small_w.get(), this->small_p.get(),
        this->small_q.get(), this->small_f.get(), this->small_g.get(),
        this->small_rho.get(), this->small_beta.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_x, l({{2.0, 1.0}, {2.0, 1.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_r, l({{1.25, 2.0}, {1.25, 2.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_z, l({{2.5, 3.0}, {2.5, 3.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_w, l({{3.75, 4.0}, {3.75, 4.0}}), 0);
}


TYPED_TEST(PipeCg, KernelStep1DivByZero)
{
    this->small_x->fill(1);
    this->small_r->fill(2);
    this->small_z->fill(3);
    this->small_w->fill(4);
    this->small_p->fill(4);
    this->small_q->fill(3);
    this->small_f->fill(2);
    this->small_g->fill(1);
    this->small_rho->fill(1);
    this->small_beta->fill(0);

    gko::kernels::reference::pipe_cg::step_1(
        this->exec, this->small_x.get(), this->small_r.get(),
        this->small_z.get(), this->small_w.get(), this->small_p.get(),
        this->small_q.get(), this->small_f.get(), this->small_g.get(),
        this->small_rho.get(), this->small_beta.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_x, l({{1.0, 1.0}, {1.0, 1.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_r, l({{2.0, 2.0}, {2.0, 2.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_z, l({{3.0, 3.0}, {3.0, 3.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_w, l({{4.0, 4.0}, {4.0, 4.0}}), 0);
}


TYPED_TEST(PipeCg, KernelStep2)
{
    this->small_z->fill(1);
    this->small_w->fill(2);
    this->small_m->fill(3);
    this->small_n->fill(4);
    this->small_p->fill(4);
    this->small_q->fill(3);
    this->small_f->fill(2);
    this->small_g->fill(1);
    this->small_rho->at(0) = -2;
    this->small_rho->at(1) = 3;
    this->small_prev_rho->at(0) = 4;
    this->small_prev_rho->at(1) = 3;
    this->small_beta->at(0) = 2;
    this->small_beta->at(1) = 3;
    this->small_delta->at(0) = 5;
    this->small_delta->at(1) = 6;
    this->small_stop.get_data()[0].reset();
    this->small_stop.get_data()[1] = this->stopped;

    gko::kernels::reference::pipe_cg::step_2(
        this->exec, this->small_beta.get(), this->small_p.get(),
        this->small_q.get(), this->small_f.get(), this->small_g.get(),
        this->small_z.get(), this->small_w.get(), this->small_m.get(),
        this->small_n.get(), this->small_prev_rho.get(), this->small_rho.get(),
        this->small_delta.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_beta, l({{4.5, 3.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_p, l({{-1.0, 4.0}, {-1.0, 4.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_q, l({{0.5, 3.0}, {0.5, 3.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_f, l({{2.0, 2.0}, {2.0, 2.0}}), 0);
    GKO_ASSERT_MTX_NEAR(this->small_g, l({{3.5, 1.0}, {3.5, 1.0}}), 0);
}


TYPED_TEST(PipeCg, KernelStep2DivByZero)
{
    this->small_z->fill(1);
    this->small_w->fill(2);
    this->small_m->fill(3);
    this->small_n->fill(4);
    this->small_p->fill(4);
    this->small_q->fill(3);
    this->small_f->fill(2);
    this->small_g->fill(1);
    this->small_rho->at(0) = -2;
    this->small_rho->at(1) = 3;
    this->small_prev_rho->fill(0);
    this->small_beta->at(0) = 2;
    this->small_beta->at(1) = 3;
    this->small_delta->at(0) = 5;
    this->small_delta->at(1) = 6;

    gko::kernels::reference::pipe_cg::step_2(
        this->exec, this->small_beta.get(), this->small_p.get(),
        this->small_q.get(), this->small_f.get(), this->small_g.get(),
        this->small_z.get(), this->small_w.get(), this->small_m.get(),
        this->small_n.get(), this->small_prev_rho.get(), this->small_rho.get(),
        this->small_delta.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_beta, this->small_delta, 0);
    GKO_ASSERT_MTX_NEAR(this->small_p, this->small_z, 0);
    GKO_ASSERT_MTX_NEAR(this->small_q, this->small_w, 0);
    GKO_ASSERT_MTX_NEAR(this->small_f, this->small_m, 0);
    GKO_ASSERT_MTX_NEAR(this->small_g, this->small_n, 0);
}


TYPED_TEST(PipeCg, KernelStep2BetaZero)
{
    using value_type = typename TestFixture::value_type;
    this->small_z->fill(1);
    this->small_w->fill(1);
    this->small_m->fill(1);
    this->small_n->fill(1);
    this->small_p->fill(1);
    this->small_q->fill(1);
    this->small_f->fill(1);
    this->small_g->fill(1);
    this->small_rho->at(0) = 3;
    this->small_rho->at(1) = 3;
    this->small_prev_rho->at(0) = 3;
    this->small_prev_rho->at(1) = 6;
    this->small_beta->at(0) = 2;
    this->small_beta->at(1) = 4;
    this->small_delta->at(0) = 2;
    this->small_delta->at(1) = 1;
    this->small_stop.get_data()[0].reset();
    this->small_stop.get_data()[1].reset();

    gko::kernels::reference::pipe_cg::step_2(
        this->exec, this->small_beta.get(), this->small_p.get(),
        this->small_q.get(), this->small_f.get(), this->small_g.get(),
        this->small_z.get(), this->small_w.get(), this->small_m.get(),
        this->small_n.get(), this->small_prev_rho.get(), this->small_rho.get(),
        this->small_delta.get(), &this->small_stop);

    GKO_ASSERT_MTX_NEAR(this->small_beta, this->small_delta, 0);
    GKO_ASSERT_MTX_NEAR(this->small_p, l({{2.0, 1.5}, {2.0, 1.5}}),
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(this->small_q, l({{2.0, 1.5}, {2.0, 1.5}}),
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(this->small_f, l({{2.0, 1.5}, {2.0, 1.5}}),
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(this->small_g, l({{2.0, 1.5}, {2.0, 1.5}}),
                        r<value_type>::value);
}


TYPED_TEST(PipeCg, SolvesStencilSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->pipe_cg_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value);
}


TYPED_TEST(PipeCg, SolvesStencilSystemMixed)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->pipe_cg_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(PipeCg, SolvesStencilSystemComplex)
{
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->pipe_cg_factory->generate(this->mtx);
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


TYPED_TEST(PipeCg, SolvesStencilSystemMixedComplex)
{
    using value_type =
        gko::to_complex<gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->pipe_cg_factory->generate(this->mtx);
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


TYPED_TEST(PipeCg, SolvesMultipleStencilSystems)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto solver = this->pipe_cg_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {I<T>{-1.0, 1.0}, I<T>{3.0, 0.0}, I<T>{1.0, 1.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({{1.0, 1.0}, {3.0, 1.0}, {2.0, 1.0}}),
                        r<value_type>::value);
}


TYPED_TEST(PipeCg, SolvesStencilSystemUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->pipe_cg_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.5, 5.0, 2.0}), r<value_type>::value);
}


TYPED_TEST(PipeCg, SolvesStencilSystemUsingAdvancedApplyMixed)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->pipe_cg_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.5, 5.0, 2.0}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(PipeCg, SolvesStencilSystemUsingAdvancedApplyComplex)
{
    using Scalar = typename TestFixture::Mtx;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->pipe_cg_factory->generate(this->mtx);
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


TYPED_TEST(PipeCg, SolvesStencilSystemUsingAdvancedApplyMixedComplex)
{
    using Scalar = gko::matrix::Dense<
        gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->pipe_cg_factory->generate(this->mtx);
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


TYPED_TEST(PipeCg, SolvesMultipleStencilSystemsUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto solver = this->pipe_cg_factory->generate(this->mtx);
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


TYPED_TEST(PipeCg, SolvesBigDenseSystem1)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    // the system is already out of half precision range
    SKIP_IF_HALF(value_type);
    auto solver = this->pipe_cg_factory_big->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {1300083.0, 1018120.5, 906410.0, -42679.5, 846779.5, 1176858.5},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({81.0, 55.0, 45.0, 5.0, 85.0, -10.0}),
                        r<value_type>::value * 5 * 1e3);
}


TYPED_TEST(PipeCg, SolvesBigDenseSystem2)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    // the system is already out of half precision range
    SKIP_IF_HALF(value_type);
    auto solver = this->pipe_cg_factory_big->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {886630.5, -172578.0, 684522.0, -65310.5, 455487.5, 607436.0},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    // TODO: the tolerance is too big.
    GKO_ASSERT_MTX_NEAR(x, l({33.0, -56.0, 81.0, -30.0, 21.0, 40.0}),
                        r<value_type>::value * 2 * 1e5);
}


TYPED_TEST(PipeCg, SolvesBigDenseSystem3)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    // the system is already out of half precision range
    SKIP_IF_HALF(value_type);
    auto solver = this->pipe_cg_factory_big2->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {886630.5, -172578.0, 684522.0, -65310.5, 455487.5, 607436.0},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({33.0, -56.0, 81.0, -30.0, 21.0, 40.0}),
                        r<value_type>::value * 2 * 1e5);
}


TYPED_TEST(PipeCg, SolvesMultipleDenseSystemForDivergenceCheck)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    // the system is already out of half precision range
    SKIP_IF_HALF(value_type);
    auto solver = this->pipe_cg_factory_big->generate(this->mtx_big);
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

    // Not sure if this is necessary, the assertions above should cover what is
    // needed.
    GKO_ASSERT_MTX_NEAR(xc, mergedRes, r<value_type>::value);
}


TYPED_TEST(PipeCg, SolvesTransposedBigDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    // the system is already out of half precision range
    SKIP_IF_HALF(value_type);
    auto solver = this->pipe_cg_factory_big->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {1300083.0, 1018120.5, 906410.0, -42679.5, 846779.5, 1176858.5},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({81.0, 55.0, 45.0, 5.0, 85.0, -10.0}),
                        r<value_type>::value * 5 * 1e4);
}


TYPED_TEST(PipeCg, SolvesConjTransposedBigDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    // the system is already out of half precision range
    SKIP_IF_HALF(value_type);
    auto solver = this->pipe_cg_factory_big->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {1300083.0, 1018120.5, 906410.0, -42679.5, 846779.5, 1176858.5},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->conj_transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({81.0, 55.0, 45.0, 5.0, 85.0, -10.0}),
                        r<value_type>::value * 5 * 1e4);
}
