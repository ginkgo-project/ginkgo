// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/bicgstab_kernels.hpp"


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/executor.hpp"


class Bicgstab : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Bicgstab<value_type>;

    Bicgstab() : rand_engine(30)
    {
        auto data = gko::matrix_data<value_type, index_type>(
            gko::dim<2>{123, 123},
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine);
        gko::utils::make_diag_dominant(data);
        mtx = Mtx::create(ref, data.size, 125);
        mtx->read(data);
        d_mtx = gko::clone(exec, mtx);
        exec_bicgstab_factory =
            Solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(246u),
                    gko::stop::ResidualNorm<value_type>::build()
                        .with_reduction_factor(::r<value_type>::value))
                .on(exec);

        ref_bicgstab_factory =
            Solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(246u),
                    gko::stop::ResidualNorm<value_type>::build()
                        .with_reduction_factor(::r<value_type>::value))
                .on(ref);
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
        gko::size_type n = 17;
        x = gen_mtx(m, n, n + 3);
        b = gen_mtx(m, n, n + 2);
        r = gen_mtx(m, n, n + 2);
        z = gen_mtx(m, n, n + 2);
        p = gen_mtx(m, n, n + 2);
        rr = gen_mtx(m, n, n + 2);
        s = gen_mtx(m, n, n + 2);
        t = gen_mtx(m, n, n + 2);
        y = gen_mtx(m, n, n + 2);
        v = gen_mtx(m, n, n + 2);
        prev_rho = gen_mtx(1, n, n);
        rho = gen_mtx(1, n, n);
        alpha = gen_mtx(1, n, n);
        beta = gen_mtx(1, n, n);
        gamma = gen_mtx(1, n, n);
        omega = gen_mtx(1, n, n);
        // check correct handling for zero values
        prev_rho->at(2) = 0.0;
        prev_rho->at(4) = 0.0;
        beta->at(2) = 0.0;
        omega->at(2) = 0.0;
        omega->at(3) = 0.0;
        stop_status =
            std::make_unique<gko::array<gko::stopping_status>>(ref, n);
        for (size_t i = 0; i < n; ++i) {
            stop_status->get_data()[i].reset();
        }
        // check correct handling for stopped columns
        stop_status->get_data()[1].stop(1);

        d_x = gko::clone(exec, x);
        d_b = gko::clone(exec, b);
        d_r = gko::clone(exec, r);
        d_z = gko::clone(exec, z);
        d_p = gko::clone(exec, p);
        d_t = gko::clone(exec, t);
        d_s = gko::clone(exec, s);
        d_y = gko::clone(exec, y);
        d_v = gko::clone(exec, v);
        d_rr = gko::clone(exec, rr);
        d_prev_rho = gko::clone(exec, prev_rho);
        d_rho = gko::clone(exec, rho);
        d_alpha = gko::clone(exec, alpha);
        d_beta = gko::clone(exec, beta);
        d_gamma = gko::clone(exec, gamma);
        d_omega = gko::clone(exec, omega);
        d_stop_status = std::make_unique<gko::array<gko::stopping_status>>(
            exec, *stop_status);
    }

    std::default_random_engine rand_engine;

    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> d_mtx;
    std::unique_ptr<Solver::Factory> exec_bicgstab_factory;
    std::unique_ptr<Solver::Factory> ref_bicgstab_factory;

    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> r;
    std::unique_ptr<Mtx> z;
    std::unique_ptr<Mtx> p;
    std::unique_ptr<Mtx> rr;
    std::unique_ptr<Mtx> s;
    std::unique_ptr<Mtx> t;
    std::unique_ptr<Mtx> y;
    std::unique_ptr<Mtx> v;
    std::unique_ptr<Mtx> prev_rho;
    std::unique_ptr<Mtx> rho;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> gamma;
    std::unique_ptr<Mtx> omega;
    std::unique_ptr<gko::array<gko::stopping_status>> stop_status;

    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_r;
    std::unique_ptr<Mtx> d_z;
    std::unique_ptr<Mtx> d_p;
    std::unique_ptr<Mtx> d_t;
    std::unique_ptr<Mtx> d_s;
    std::unique_ptr<Mtx> d_y;
    std::unique_ptr<Mtx> d_v;
    std::unique_ptr<Mtx> d_rr;
    std::unique_ptr<Mtx> d_prev_rho;
    std::unique_ptr<Mtx> d_rho;
    std::unique_ptr<Mtx> d_alpha;
    std::unique_ptr<Mtx> d_beta;
    std::unique_ptr<Mtx> d_gamma;
    std::unique_ptr<Mtx> d_omega;
    std::unique_ptr<gko::array<gko::stopping_status>> d_stop_status;
};


TEST_F(Bicgstab, BicgstabInitializeIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicgstab::initialize(
        ref, b.get(), r.get(), rr.get(), y.get(), s.get(), t.get(), z.get(),
        v.get(), p.get(), prev_rho.get(), rho.get(), alpha.get(), beta.get(),
        gamma.get(), omega.get(), stop_status.get());
    gko::kernels::EXEC_NAMESPACE::bicgstab::initialize(
        exec, d_b.get(), d_r.get(), d_rr.get(), d_y.get(), d_s.get(), d_t.get(),
        d_z.get(), d_v.get(), d_p.get(), d_prev_rho.get(), d_rho.get(),
        d_alpha.get(), d_beta.get(), d_gamma.get(), d_omega.get(),
        d_stop_status.get());

    GKO_EXPECT_MTX_NEAR(d_r, r, ::r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(d_z, z, ::r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(d_p, p, ::r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(d_y, y, ::r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(d_t, t, ::r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(d_s, s, ::r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(d_rr, rr, ::r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(d_v, v, ::r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(d_prev_rho, prev_rho, ::r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(d_rho, rho, ::r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(d_alpha, alpha, ::r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(d_beta, beta, ::r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(d_gamma, gamma, ::r<value_type>::value);
    GKO_EXPECT_MTX_NEAR(d_omega, omega, ::r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(*d_stop_status, *stop_status);
}


TEST_F(Bicgstab, BicgstabStep1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicgstab::step_1(
        ref, r.get(), p.get(), v.get(), rho.get(), prev_rho.get(), alpha.get(),
        omega.get(), stop_status.get());
    gko::kernels::EXEC_NAMESPACE::bicgstab::step_1(
        exec, d_r.get(), d_p.get(), d_v.get(), d_rho.get(), d_prev_rho.get(),
        d_alpha.get(), d_omega.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_p, p, ::r<value_type>::value);
}


TEST_F(Bicgstab, BicgstabStep2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicgstab::step_2(ref, r.get(), s.get(), v.get(),
                                              rho.get(), alpha.get(),
                                              beta.get(), stop_status.get());
    gko::kernels::EXEC_NAMESPACE::bicgstab::step_2(
        exec, d_r.get(), d_s.get(), d_v.get(), d_rho.get(), d_alpha.get(),
        d_beta.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_alpha, alpha, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_s, s, ::r<value_type>::value);
}


TEST_F(Bicgstab, BicgstabStep3IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicgstab::step_3(
        ref, x.get(), r.get(), s.get(), t.get(), y.get(), z.get(), alpha.get(),
        beta.get(), gamma.get(), omega.get(), stop_status.get());
    gko::kernels::EXEC_NAMESPACE::bicgstab::step_3(
        exec, d_x.get(), d_r.get(), d_s.get(), d_t.get(), d_y.get(), d_z.get(),
        d_alpha.get(), d_beta.get(), d_gamma.get(), d_omega.get(),
        d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_omega, omega, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_r, r, ::r<value_type>::value);
}


TEST_F(Bicgstab, BicgstabApplyOneRHSIsEquivalentToRef)
{
    int m = 123;
    int n = 1;
    auto ref_solver = ref_bicgstab_factory->generate(mtx);
    auto exec_solver = exec_bicgstab_factory->generate(d_mtx);
    auto b = gen_mtx(m, n, n + 2);
    auto x = gen_mtx(m, n, n + 3);
    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);

    ref_solver->apply(b, x);
    exec_solver->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value * 1000);
}


TEST_F(Bicgstab, BicgstabApplyMultipleRHSIsEquivalentToRef)
{
    int m = 123;
    int n = 16;
    auto exec_solver = exec_bicgstab_factory->generate(d_mtx);
    auto ref_solver = ref_bicgstab_factory->generate(mtx);
    auto b = gen_mtx(m, n, n + 4);
    auto x = gen_mtx(m, n, n + 3);
    auto d_b = gko::clone(exec, b);
    auto d_x = gko::clone(exec, x);

    ref_solver->apply(b, x);
    exec_solver->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, ::r<value_type>::value * 2000);
}
