// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/bicg_kernels.hpp"


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/bicg.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"
#include "core/utils/matrix_utils.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


class Bicg : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Dense<value_type>;

    Bicg() : rand_engine(30)
    {
        std::string file_name(gko::matrices::location_ani1_mtx);
        auto input_file = std::ifstream(file_name, std::ios::in);
        mtx_ani = gko::read<Mtx>(input_file, ref);
        d_mtx_ani = gko::clone(exec, mtx_ani.get());
    }

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
        // all vectors need the same stride as b, except x
        b = gen_mtx(m, n, n + 2);
        r = gen_mtx(m, n, n + 2);
        z = gen_mtx(m, n, n + 2);
        p = gen_mtx(m, n, n + 2);
        q = gen_mtx(m, n, n + 2);
        r2 = gen_mtx(m, n, n + 2);
        z2 = gen_mtx(m, n, n + 2);
        p2 = gen_mtx(m, n, n + 2);
        q2 = gen_mtx(m, n, n + 2);
        x = gen_mtx(m, n, n + 3);
        beta = gen_mtx(1, n, n);
        prev_rho = gen_mtx(1, n, n);
        rho = gen_mtx(1, n, n);
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
        d_z = gko::clone(exec, z);
        d_p = gko::clone(exec, p);
        d_q = gko::clone(exec, q);
        d_r2 = gko::clone(exec, r2);
        d_z2 = gko::clone(exec, z2);
        d_p2 = gko::clone(exec, p2);
        d_q2 = gko::clone(exec, q2);
        d_x = gko::clone(exec, x);
        d_beta = gko::clone(exec, beta);
        d_prev_rho = gko::clone(exec, prev_rho);
        d_rho = gko::clone(exec, rho);
        d_stop_status = std::make_unique<gko::array<gko::stopping_status>>(
            exec, *stop_status);
    }

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> r;
    std::unique_ptr<Mtx> z;
    std::unique_ptr<Mtx> p;
    std::unique_ptr<Mtx> q;
    std::unique_ptr<Mtx> r2;
    std::unique_ptr<Mtx> z2;
    std::unique_ptr<Mtx> p2;
    std::unique_ptr<Mtx> q2;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> prev_rho;
    std::unique_ptr<Mtx> rho;
    std::shared_ptr<Mtx> mtx_ani;
    std::unique_ptr<gko::array<gko::stopping_status>> stop_status;

    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_r;
    std::unique_ptr<Mtx> d_z;
    std::unique_ptr<Mtx> d_p;
    std::unique_ptr<Mtx> d_q;
    std::unique_ptr<Mtx> d_r2;
    std::unique_ptr<Mtx> d_z2;
    std::unique_ptr<Mtx> d_p2;
    std::unique_ptr<Mtx> d_q2;
    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_beta;
    std::unique_ptr<Mtx> d_prev_rho;
    std::unique_ptr<Mtx> d_rho;
    std::shared_ptr<Mtx> d_mtx_ani;
    std::unique_ptr<gko::array<gko::stopping_status>> d_stop_status;
};


TEST_F(Bicg, BicgInitializeIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicg::initialize(
        ref, b.get(), r.get(), z.get(), p.get(), q.get(), prev_rho.get(),
        rho.get(), r2.get(), z2.get(), p2.get(), q2.get(), stop_status.get());
    gko::kernels::EXEC_NAMESPACE::bicg::initialize(
        exec, d_b.get(), d_r.get(), d_z.get(), d_p.get(), d_q.get(),
        d_prev_rho.get(), d_rho.get(), d_r2.get(), d_z2.get(), d_p2.get(),
        d_q2.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_r, r, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_z, z, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_p, p, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_q, q, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_r2, r2, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_z2, z2, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_p2, p2, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_q2, q2, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_prev_rho, prev_rho, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_rho, rho, ::r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(*d_stop_status, *stop_status);
}


TEST_F(Bicg, BicgStep1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicg::step_1(ref, p.get(), z.get(), p2.get(),
                                          z2.get(), rho.get(), prev_rho.get(),
                                          stop_status.get());
    gko::kernels::EXEC_NAMESPACE::bicg::step_1(
        exec, d_p.get(), d_z.get(), d_p2.get(), d_z2.get(), d_rho.get(),
        d_prev_rho.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_p, p, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_z, z, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_p2, p2, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_z2, z2, ::r<value_type>::value);
}


TEST_F(Bicg, BicgStep2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicg::step_2(
        ref, x.get(), r.get(), r2.get(), p.get(), q.get(), q2.get(), beta.get(),
        rho.get(), stop_status.get());
    gko::kernels::EXEC_NAMESPACE::bicg::step_2(
        exec, d_x.get(), d_r.get(), d_r2.get(), d_p.get(), d_q.get(),
        d_q2.get(), d_beta.get(), d_rho.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_r, r, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_r2, r2, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_p, p, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_q, q, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_q2, q2, ::r<value_type>::value);
}


TEST_F(Bicg, ApplyWithSpdMatrixIsEquivalentToRef)
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
    auto bicg_factory =
        gko::solver::Bicg<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(50u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(::r<value_type>::value))
            .on(ref);
    auto d_bicg_factory =
        gko::solver::Bicg<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(50u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(::r<value_type>::value))
            .on(exec);
    auto solver = bicg_factory->generate(std::move(mtx));
    auto d_solver = d_bicg_factory->generate(std::move(d_mtx));

    solver->apply(b, x);
    d_solver->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value * 1000);
}


TEST_F(Bicg, ApplyWithSuiteSparseMatrixIsEquivalentToRef)
{
    auto x = gen_mtx(36, 1, 2);
    auto b = gen_mtx(36, 1, 3);
    auto d_x = gko::clone(exec, x);
    auto d_b = gko::clone(exec, b);
    auto bicg_factory =
        gko::solver::Bicg<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(50u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(::r<value_type>::value))
            .on(ref);
    auto d_bicg_factory =
        gko::solver::Bicg<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(50u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(::r<value_type>::value))
            .on(exec);
    auto solver = bicg_factory->generate(std::move(mtx_ani));
    auto d_solver = d_bicg_factory->generate(std::move(d_mtx_ani));

    solver->apply(b, x);
    d_solver->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value * 100);
}
