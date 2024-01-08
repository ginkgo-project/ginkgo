// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/fcg_kernels.hpp"


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/executor.hpp"


class Fcg : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Fcg<value_type>;

    Fcg() : rand_engine(30) {}

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
        gko::size_type m = 597;
        gko::size_type n = 43;
        b = gen_mtx(m, n, n + 2);
        r = gen_mtx(m, n, n + 2);
        t = gen_mtx(m, n, n + 2);
        z = gen_mtx(m, n, n + 2);
        p = gen_mtx(m, n, n + 2);
        q = gen_mtx(m, n, n + 2);
        x = gen_mtx(m, n, n + 3);
        beta = gen_mtx(1, n, n);
        prev_rho = gen_mtx(1, n, n);
        rho = gen_mtx(1, n, n);
        rho_t = gen_mtx(1, n, n);
        // check correct handling for zero values
        beta->at(2) = 0.0;
        prev_rho->at(2) = 0.0;
        stop_status =
            std::make_unique<gko::array<gko::stopping_status>>(ref, n);
        for (size_t i = 0; i < stop_status->get_size(); ++i) {
            stop_status->get_data()[i].reset();
        }
        // check correct handling for stopped columns
        stop_status->get_data()[1].stop(1);

        d_b = gko::clone(exec, b);
        d_r = gko::clone(exec, r);
        d_t = gko::clone(exec, t);
        d_z = gko::clone(exec, z);
        d_p = gko::clone(exec, p);
        d_q = gko::clone(exec, q);
        d_x = gko::clone(exec, x);
        d_beta = gko::clone(exec, beta);
        d_prev_rho = gko::clone(exec, prev_rho);
        d_rho_t = gko::clone(exec, rho_t);
        d_rho = gko::clone(exec, rho);
        d_stop_status = std::make_unique<gko::array<gko::stopping_status>>(
            exec, *stop_status);
    }

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> r;
    std::unique_ptr<Mtx> t;
    std::unique_ptr<Mtx> z;
    std::unique_ptr<Mtx> p;
    std::unique_ptr<Mtx> q;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> prev_rho;
    std::unique_ptr<Mtx> rho;
    std::unique_ptr<Mtx> rho_t;
    std::unique_ptr<gko::array<gko::stopping_status>> stop_status;

    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_r;
    std::unique_ptr<Mtx> d_t;
    std::unique_ptr<Mtx> d_z;
    std::unique_ptr<Mtx> d_p;
    std::unique_ptr<Mtx> d_q;
    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_beta;
    std::unique_ptr<Mtx> d_prev_rho;
    std::unique_ptr<Mtx> d_rho;
    std::unique_ptr<Mtx> d_rho_t;
    std::unique_ptr<gko::array<gko::stopping_status>> d_stop_status;
};


TEST_F(Fcg, FcgInitializeIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::fcg::initialize(
        ref, b.get(), r.get(), z.get(), p.get(), q.get(), t.get(),
        prev_rho.get(), rho.get(), rho_t.get(), stop_status.get());
    gko::kernels::EXEC_NAMESPACE::fcg::initialize(
        exec, d_b.get(), d_r.get(), d_z.get(), d_p.get(), d_q.get(), d_t.get(),
        d_prev_rho.get(), d_rho.get(), d_rho_t.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_r, r, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_t, t, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_z, z, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_p, p, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_q, q, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_prev_rho, prev_rho, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_rho, rho, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_rho_t, rho_t, ::r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(*d_stop_status, *stop_status);
}


TEST_F(Fcg, FcgStep1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::fcg::step_1(ref, p.get(), z.get(), rho_t.get(),
                                         prev_rho.get(), stop_status.get());
    gko::kernels::EXEC_NAMESPACE::fcg::step_1(exec, d_p.get(), d_z.get(),
                                              d_rho_t.get(), d_prev_rho.get(),
                                              d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_p, p, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_z, z, ::r<value_type>::value);
}


TEST_F(Fcg, FcgStep2IsEquivalentToRef)
{
    initialize_data();
    gko::kernels::reference::fcg::step_2(ref, x.get(), r.get(), t.get(),
                                         p.get(), q.get(), beta.get(),
                                         rho.get(), stop_status.get());
    gko::kernels::EXEC_NAMESPACE::fcg::step_2(
        exec, d_x.get(), d_r.get(), d_t.get(), d_p.get(), d_q.get(),
        d_beta.get(), d_rho.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_r, r, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_t, t, ::r<value_type>::value);
}


TEST_F(Fcg, ApplyIsEquivalentToRef)
{
    auto data = gko::matrix_data<value_type, index_type>(
        gko::dim<2>{50, 50}, std::normal_distribution<value_type>(-1.0, 1.0),
        rand_engine);
    gko::utils::make_hpd(data, 1.5);
    auto mtx = Mtx::create(ref, data.size, 53);
    mtx->read(data);
    auto x = gen_mtx(50, 3, 4);
    auto b = gen_mtx(50, 3, 5);
    auto d_mtx = gko::clone(exec, mtx);
    auto d_x = gko::clone(exec, x);
    auto d_b = gko::clone(exec, b);
    auto fcg_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(50u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(::r<value_type>::value))
            .on(ref);
    auto d_fcg_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(50u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(::r<value_type>::value))
            .on(exec);
    auto solver = fcg_factory->generate(std::move(mtx));
    auto d_solver = d_fcg_factory->generate(std::move(d_mtx));

    solver->apply(b, x);
    d_solver->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value * 1000);
}
