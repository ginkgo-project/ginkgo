// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/minres_kernels.hpp"

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/minres.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>

#include "core/test/utils.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/common_fixture.hpp"
#include "test/utils/executor.hpp"

namespace {

class Minres : public ::testing::Test {
protected:
#if GINKGO_COMMON_SINGLE_MODE
    using value_type = float;
#else
    using value_type = double;
#endif
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Minres<value_type>;

    Minres() : rand_engine(42) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    std::unique_ptr<Mtx> gen_mtx(gko::size_type num_rows,
                                 gko::size_type num_cols, gko::size_type stride,
                                 bool make_hermitian)
    {
        auto tmp_mtx =
            gko::test::generate_random_matrix_data<value_type, gko::int32>(
                num_rows, num_cols,
                std::uniform_int_distribution<>(num_cols, num_cols),
                std::normal_distribution<value_type>(-1.0, 1.0), rand_engine);
        if (make_hermitian) {
            gko::utils::make_unit_diagonal(tmp_mtx);
            gko::utils::make_hermitian(tmp_mtx);
        }
        auto result = Mtx::create(ref, gko::dim<2>{num_rows, num_cols}, stride);
        result->read(tmp_mtx);
        return result;
    }

    void initialize_data()
    {
        gko::size_type m = 597;
        gko::size_type n = 43;
        // all vectors need the same stride as b, except x
        b = gen_mtx(m, n, n + 2, false);
        r = gen_mtx(m, n, n + 2, false);
        z = gen_mtx(m, n, n + 2, false);
        z_tilde = gen_mtx(m, n, n + 2, false);
        p = gen_mtx(m, n, n + 2, false);
        p_prev = gen_mtx(m, n, n + 2, false);
        q = gen_mtx(m, n, n + 2, false);
        q_prev = gen_mtx(m, n, n + 2, false);
        v = gen_mtx(m, n, n + 2, false);
        x = gen_mtx(m, n, n + 3, false);
        alpha = gen_mtx(1, n, n, false);
        beta = gen_mtx(1, n, n, false)->compute_absolute();
        gamma = gen_mtx(1, n, n, false);
        delta = gen_mtx(1, n, n, false);
        cos_prev = gen_mtx(1, n, n, false);
        cos = gen_mtx(1, n, n, false);
        sin_prev = gen_mtx(1, n, n, false);
        sin = gen_mtx(1, n, n, false);
        eta_next = gen_mtx(1, n, n, false);
        eta = gen_mtx(1, n, n, false);
        tau = gen_mtx(1, n, n, false)->compute_absolute();
        // check correct handling for zero values
        beta->at(2) = gko::zero<value_type>();
        stop_status =
            std::make_unique<gko::array<gko::stopping_status>>(ref, n);
        for (gko::size_type i = 0; i < stop_status->get_size(); ++i) {
            stop_status->get_data()[i].reset();
        }
        // check correct handling for stopped columns
        stop_status->get_data()[1].stop(1);

        d_x = gko::clone(exec, x);
        d_b = gko::clone(exec, b);
        d_r = gko::clone(exec, r);
        d_z = gko::clone(exec, z);
        d_p = gko::clone(exec, p);
        d_q = gko::clone(exec, q);
        d_z_tilde = gko::clone(exec, z_tilde);
        d_v = gko::clone(exec, v);
        d_p_prev = gko::clone(exec, p_prev);
        d_q_prev = gko::clone(exec, q_prev);
        d_alpha = gko::clone(exec, alpha);
        d_beta = gko::clone(exec, beta);
        d_gamma = gko::clone(exec, gamma);
        d_delta = gko::clone(exec, delta);
        d_eta_next = gko::clone(exec, eta_next);
        d_eta = gko::clone(exec, eta);
        d_tau = gko::clone(exec, tau);
        d_cos_prev = gko::clone(exec, cos_prev);
        d_cos = gko::clone(exec, cos);
        d_sin_prev = gko::clone(exec, sin_prev);
        d_sin = gko::clone(exec, sin);
        d_stop_status = std::make_unique<gko::array<gko::stopping_status>>(
            exec, *stop_status);
    }


    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> r;
    std::unique_ptr<Mtx> z;
    std::unique_ptr<Mtx> p;
    std::unique_ptr<Mtx> q;
    std::unique_ptr<Mtx> z_tilde;
    std::unique_ptr<Mtx> v;
    std::unique_ptr<Mtx> p_prev;
    std::unique_ptr<Mtx> q_prev;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> gamma;
    std::unique_ptr<Mtx> delta;
    std::unique_ptr<Mtx> eta_next;
    std::unique_ptr<Mtx> eta;
    std::unique_ptr<typename Mtx::absolute_type> tau;
    std::unique_ptr<Mtx> cos_prev;
    std::unique_ptr<Mtx> cos;
    std::unique_ptr<Mtx> sin_prev;
    std::unique_ptr<Mtx> sin;

    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_r;
    std::unique_ptr<Mtx> d_z;
    std::unique_ptr<Mtx> d_p;
    std::unique_ptr<Mtx> d_q;
    std::unique_ptr<Mtx> d_z_tilde;
    std::unique_ptr<Mtx> d_v;
    std::unique_ptr<Mtx> d_p_prev;
    std::unique_ptr<Mtx> d_q_prev;
    std::unique_ptr<Mtx> d_alpha;
    std::unique_ptr<Mtx> d_beta;
    std::unique_ptr<Mtx> d_gamma;
    std::unique_ptr<Mtx> d_delta;
    std::unique_ptr<Mtx> d_eta_next;
    std::unique_ptr<Mtx> d_eta;
    std::unique_ptr<typename Mtx::absolute_type> d_tau;
    std::unique_ptr<Mtx> d_cos_prev;
    std::unique_ptr<Mtx> d_cos;
    std::unique_ptr<Mtx> d_sin_prev;
    std::unique_ptr<Mtx> d_sin;

    std::unique_ptr<gko::array<gko::stopping_status>> stop_status;
    std::unique_ptr<gko::array<gko::stopping_status>> d_stop_status;
};

TEST_F(Minres, MinresInitializeIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::minres::initialize(
        ref, r.get(), z.get(), p.get(), p_prev.get(), q.get(), q_prev.get(),
        v.get(), beta.get(), gamma.get(), delta.get(), cos_prev.get(),
        cos.get(), sin_prev.get(), sin.get(), eta_next.get(), eta.get(),
        stop_status.get());
    gko::kernels::GKO_DEVICE_NAMESPACE::minres::initialize(
        exec, d_r.get(), d_z.get(), d_p.get(), d_p_prev.get(), d_q.get(),
        d_q_prev.get(), d_v.get(), d_beta.get(), d_gamma.get(), d_delta.get(),
        d_cos_prev.get(), d_cos.get(), d_sin_prev.get(), d_sin.get(),
        d_eta_next.get(), d_eta.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_r, r, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_z, z, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_p, p, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_q, q, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_p_prev, p_prev, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_q_prev, q_prev, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_v, v, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_alpha, alpha, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_beta, beta, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_gamma, gamma, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_delta, delta, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_eta_next, eta_next, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_eta, eta, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_cos_prev, cos_prev, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_cos, cos, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_sin_prev, sin_prev, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_sin, sin, ::r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(*d_stop_status, *stop_status);
}


TEST_F(Minres, MinresStep1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::minres::step_1(
        ref, alpha.get(), beta.get(), gamma.get(), delta.get(), cos_prev.get(),
        cos.get(), sin_prev.get(), sin.get(), eta.get(), eta_next.get(),
        tau.get(), stop_status.get());
    gko::kernels::GKO_DEVICE_NAMESPACE::minres::step_1(
        exec, d_alpha.get(), d_beta.get(), d_gamma.get(), d_delta.get(),
        d_cos_prev.get(), d_cos.get(), d_sin_prev.get(), d_sin.get(),
        d_eta.get(), d_eta_next.get(), d_tau.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_alpha, alpha, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_beta, beta, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_gamma, gamma, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_delta, delta, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_eta_next, eta_next, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_eta, eta, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_tau, tau, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_cos_prev, cos_prev, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_cos, cos, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_sin_prev, sin_prev, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_sin, sin, ::r<value_type>::value);
}


TEST_F(Minres, MinresStep2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::minres::step_2(
        ref, x.get(), p.get(), p_prev.get(), z.get(), z_tilde.get(), q.get(),
        q_prev.get(), v.get(), alpha.get(), beta.get(), gamma.get(),
        delta.get(), cos.get(), eta.get(), stop_status.get());
    gko::kernels::GKO_DEVICE_NAMESPACE::minres::step_2(
        exec, d_x.get(), d_p.get(), d_p_prev.get(), d_z.get(), d_z_tilde.get(),
        d_q.get(), d_q_prev.get(), d_v.get(), d_alpha.get(), d_beta.get(),
        d_gamma.get(), d_delta.get(), d_cos.get(), d_eta.get(),
        d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_z, z, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_p, p, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_q, q, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_v, v, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_p_prev, p_prev, ::r<value_type>::value);
}


TEST_F(Minres, ApplyIsEquivalentToRef)
{
    auto mtx = gen_mtx(50, 50, 53, true);
    auto x = gen_mtx(50, 1, 5, false);
    auto b = gen_mtx(50, 1, 4, false);
    auto d_mtx = gko::clone(exec, mtx);
    auto d_x = gko::clone(exec, x);
    auto d_b = gko::clone(exec, b);
    auto minres_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(400u).on(ref),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(::r<value_type>::value)
                    .on(ref))
            .on(ref);
    auto d_minres_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(400u).on(exec),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(::r<value_type>::value)
                    .on(exec))
            .on(exec);
    auto solver = minres_factory->generate(std::move(mtx));
    auto d_solver = d_minres_factory->generate(std::move(d_mtx));

    solver->apply(b.get(), x.get());
    d_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value * 100);
}


TEST_F(Minres, PreconditionedApplyIsEquivalentToRef)
{
    auto mtx = gen_mtx(50, 50, 53, true);
    auto x = gen_mtx(50, 1, 5, false);
    auto b = gen_mtx(50, 1, 4, false);
    auto d_mtx = gko::clone(exec, mtx);
    auto d_x = gko::clone(exec, x);
    auto d_b = gko::clone(exec, b);
    auto minres_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(400u).on(ref),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(::r<value_type>::value)
                    .on(ref))
            .with_preconditioner(
                gko::preconditioner::Jacobi<value_type>::build()
                    .with_max_block_size(1u)
                    .on(ref))
            .on(ref);
    auto d_minres_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(400u).on(exec),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(::r<value_type>::value)
                    .on(exec))
            .with_preconditioner(
                gko::preconditioner::Jacobi<value_type>::build()
                    .with_max_block_size(1u)
                    .on(exec))
            .on(exec);
    auto solver = minres_factory->generate(std::move(mtx));
    auto d_solver = d_minres_factory->generate(std::move(d_mtx));

    solver->apply(b.get(), x.get());
    d_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value * 100);
}


}  // namespace
