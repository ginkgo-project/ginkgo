// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/pipe_cg_kernels.hpp"

#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/pipe_cg.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/test/utils.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/common_fixture.hpp"


class PipeCg : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Dense<value_type>;

    PipeCg() : rand_engine(30) {}

    std::unique_ptr<Mtx> gen_mtx(gko::size_type num_rows,
                                 gko::size_type num_cols, gko::size_type stride)
    {
        auto tmp_mtx = gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
        auto result = Mtx::create(ref, gko::dim<2>{num_rows, num_cols}, stride);
        result->copy_from(tmp_mtx);
        return result;
    }

    void initialize_data()
    {
        gko::size_type size_m = 597;
        gko::size_type size_n = 43;
        // all vectors need the same stride as b, except x
        b = gen_mtx(size_m, size_n, size_n + 2);
        r = gen_mtx(size_m, size_n, size_n + 2);
        z = gen_mtx(size_m, size_n, size_n + 2);
        w = gen_mtx(size_m, size_n, size_n + 2);
        m = gen_mtx(size_m, size_n, size_n + 2);
        n = gen_mtx(size_m, size_n, size_n + 2);
        f = gen_mtx(size_m, size_n, size_n + 2);
        g = gen_mtx(size_m, size_n, size_n + 2);
        p = gen_mtx(size_m, size_n, size_n + 2);
        q = gen_mtx(size_m, size_n, size_n + 2);
        x = gen_mtx(size_m, size_n, size_n + 3);
        beta = gen_mtx(1, size_n, size_n);
        delta = gen_mtx(1, size_n, size_n);
        prev_rho = gen_mtx(1, size_n, size_n);
        rho = gen_mtx(1, size_n, size_n);
        // check correct handling for zero values
        beta->at(2) = 0.0;
        delta->at(2) = 0.0;
        prev_rho->at(2) = 0.0;
        stop_status =
            std::make_unique<gko::array<gko::stopping_status>>(ref, size_n);
        for (size_t i = 0; i < stop_status->get_size(); ++i) {
            stop_status->get_data()[i].reset();
        }
        // check correct handling for stopped columns
        stop_status->get_data()[1].stop(1);

        d_b = gko::clone(exec, b);
        d_r = gko::clone(exec, r);
        d_z = gko::clone(exec, z);
        d_w = gko::clone(exec, w);
        d_m = gko::clone(exec, m);
        d_n = gko::clone(exec, n);
        d_f = gko::clone(exec, f);
        d_g = gko::clone(exec, g);
        d_p = gko::clone(exec, p);
        d_q = gko::clone(exec, q);
        d_x = gko::clone(exec, x);
        d_beta = gko::clone(exec, beta);
        d_delta = gko::clone(exec, delta);
        d_prev_rho = gko::clone(exec, prev_rho);
        d_rho = gko::clone(exec, rho);
        d_stop_status = std::make_unique<gko::array<gko::stopping_status>>(
            exec, *stop_status);
    }

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> prev_rho;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> delta;
    std::unique_ptr<Mtx> rho;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> r;
    std::unique_ptr<Mtx> z;
    std::unique_ptr<Mtx> w;
    std::unique_ptr<Mtx> m;
    std::unique_ptr<Mtx> n;
    std::unique_ptr<Mtx> f;
    std::unique_ptr<Mtx> g;
    std::unique_ptr<Mtx> p;
    std::unique_ptr<Mtx> q;
    std::unique_ptr<gko::array<gko::stopping_status>> stop_status;

    std::unique_ptr<Mtx> d_prev_rho;
    std::unique_ptr<Mtx> d_beta;
    std::unique_ptr<Mtx> d_delta;
    std::unique_ptr<Mtx> d_rho;
    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_r;
    std::unique_ptr<Mtx> d_z;
    std::unique_ptr<Mtx> d_w;
    std::unique_ptr<Mtx> d_m;
    std::unique_ptr<Mtx> d_n;
    std::unique_ptr<Mtx> d_f;
    std::unique_ptr<Mtx> d_g;
    std::unique_ptr<Mtx> d_p;
    std::unique_ptr<Mtx> d_q;
    std::unique_ptr<gko::array<gko::stopping_status>> d_stop_status;
};


TEST_F(PipeCg, PipeCgInitialize1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::pipe_cg::initialize_1(
        ref, b.get(), r.get(), prev_rho.get(), stop_status.get());
    gko::kernels::GKO_DEVICE_NAMESPACE::pipe_cg::initialize_1(
        exec, d_b.get(), d_r.get(), d_prev_rho.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_r, r, 0);
    GKO_ASSERT_MTX_NEAR(d_prev_rho, prev_rho, 0);
    GKO_ASSERT_MTX_NEAR(d_rho, rho, 0);
    GKO_ASSERT_ARRAY_EQ(*d_stop_status, *stop_status);
}


TEST_F(PipeCg, PipeCgInitialize2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::pipe_cg::initialize_2(
        ref, this->p.get(), this->q.get(), this->f.get(), this->g.get(),
        this->beta.get(), this->z.get(), this->w.get(), this->m.get(),
        this->n.get(), this->delta.get());
    gko::kernels::GKO_DEVICE_NAMESPACE::pipe_cg::initialize_2(
        this->exec, this->d_p.get(), this->d_q.get(), this->d_f.get(),
        this->d_g.get(), this->d_beta.get(), this->d_z.get(), this->d_w.get(),
        this->d_m.get(), this->d_n.get(), this->d_delta.get());

    GKO_ASSERT_MTX_NEAR(d_p, p, 0);
    GKO_ASSERT_MTX_NEAR(d_q, q, 0);
    GKO_ASSERT_MTX_NEAR(d_f, f, 0);
    GKO_ASSERT_MTX_NEAR(d_g, g, 0);
    GKO_ASSERT_MTX_NEAR(d_beta, beta, 0);
}


TEST_F(PipeCg, CgStep1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::pipe_cg::step_1(
        ref, this->x.get(), this->r.get(), this->z.get(), this->w.get(),
        this->p.get(), this->q.get(), this->f.get(), this->g.get(),
        this->rho.get(), this->beta.get(), this->stop_status.get());
    gko::kernels::GKO_DEVICE_NAMESPACE::pipe_cg::step_1(
        this->exec, this->d_x.get(), this->d_r.get(), this->d_z.get(),
        this->d_w.get(), this->d_p.get(), this->d_q.get(), this->d_f.get(),
        this->d_g.get(), this->d_rho.get(), this->d_beta.get(),
        this->d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_r, r, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_z, z, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_w, w, ::r<value_type>::value);
}


TEST_F(PipeCg, CgStep2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::pipe_cg::step_2(
        ref, this->beta.get(), this->p.get(), this->q.get(), this->f.get(),
        this->g.get(), this->z.get(), this->w.get(), this->m.get(),
        this->n.get(), this->prev_rho.get(), this->rho.get(), this->delta.get(),
        this->stop_status.get());
    gko::kernels::GKO_DEVICE_NAMESPACE::pipe_cg::step_2(
        this->exec, this->d_beta.get(), this->d_p.get(), this->d_q.get(),
        this->d_f.get(), this->d_g.get(), this->d_z.get(), this->d_w.get(),
        this->d_m.get(), this->d_n.get(), this->d_prev_rho.get(),
        this->d_rho.get(), this->d_delta.get(), this->d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_beta, beta, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_p, p, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_q, q, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_f, f, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_g, g, ::r<value_type>::value);
}


TEST_F(PipeCg, ApplyIsEquivalentToRef)
{
    auto data = gko::matrix_data<value_type, index_type>(
        gko::dim<2>{50, 50}, std::normal_distribution<value_type>(-1.0, 1.0),
        rand_engine);
    gko::utils::make_hpd(data);
    auto mtx = Mtx::create(ref, data.size, 53);
    mtx->read(data);
    auto x = gen_mtx(50, 3, 5);
    auto b = gen_mtx(50, 3, 4);
    auto d_mtx = gko::clone(exec, mtx);
    auto d_x = gko::clone(exec, x);
    auto d_b = gko::clone(exec, b);
    auto cg_factory =
        gko::solver::PipeCg<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(50u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(::r<value_type>::value))
            .on(ref);
    auto d_cg_factory =
        gko::solver::PipeCg<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(50u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(::r<value_type>::value))
            .on(exec);
    auto solver = cg_factory->generate(std::move(mtx));
    auto d_solver = d_cg_factory->generate(std::move(d_mtx));

    solver->apply(b, x);
    d_solver->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value * 5 * 1e4);
}
