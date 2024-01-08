// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/multigrid_kernels.hpp"


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


class Multigrid : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Dense<>;
    Multigrid() : rand_engine(30) {}

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void initialize_data()
    {
        int m = 597;
        int n = 43;
        v = gen_mtx(m, n);
        d = gen_mtx(m, n);
        g = gen_mtx(m, n);
        e = gen_mtx(m, n);
        alpha = gen_mtx(1, n);
        rho = gen_mtx(1, n);
        beta = gen_mtx(1, n);
        gamma = gen_mtx(1, n);
        zeta = gen_mtx(1, n);
        old_norm = gen_mtx(1, n);
        new_norm = Mtx::create(ref, gko::dim<2>{1, n});
        this->modify_norm(old_norm, new_norm);
        this->modify_scalar(alpha, rho, beta, gamma, zeta);

        d_v = gko::clone(exec, v);
        d_d = gko::clone(exec, d);
        d_g = gko::clone(exec, g);
        d_e = gko::clone(exec, e);
        d_alpha = gko::clone(exec, alpha);
        d_rho = gko::clone(exec, rho);
        d_beta = gko::clone(exec, beta);
        d_gamma = gko::clone(exec, gamma);
        d_zeta = gko::clone(exec, zeta);
        d_old_norm = gko::clone(exec, old_norm);
        d_new_norm = gko::clone(exec, new_norm);
    }

    void modify_norm(std::unique_ptr<Mtx>& old_norm,
                     std::unique_ptr<Mtx>& new_norm)
    {
        double ratio = 0.7;
        for (gko::size_type i = 0; i < old_norm->get_size()[1]; i++) {
            old_norm->at(0, i) = gko::abs(old_norm->at(0, i));
            new_norm->at(0, i) = ratio * old_norm->at(0, i);
        }
    }

    void modify_scalar(std::unique_ptr<Mtx>& alpha, std::unique_ptr<Mtx>& rho,
                       std::unique_ptr<Mtx>& beta, std::unique_ptr<Mtx>& gamma,
                       std::unique_ptr<Mtx>& zeta)
    {
        // modify the first three element such that the isfinite condition can
        // be reached, which are checked in the last three group in reference
        // test.
        // scalar_d = zeta/(beta - gamma * gamma / rho)
        // scalar_e = one<ValueType>() - gamma / alpha * scalar_d
        // temp = alpha/rho

        // scalar_d, scalar_e are not finite
        alpha->at(0, 0) = 3.0;
        rho->at(0, 0) = 2.0;
        beta->at(0, 0) = 2.0;
        gamma->at(0, 0) = 2.0;
        zeta->at(0, 0) = -1.0;

        // temp, scalar_d, scalar_e are not finite
        alpha->at(0, 1) = 0.0;
        rho->at(0, 1) = 0.0;
        beta->at(0, 1) = -1.0;
        gamma->at(0, 1) = 0.0;
        zeta->at(0, 1) = 3.0;

        // scalar_e is not finite
        alpha->at(0, 2) = 0.0;
        rho->at(0, 2) = 1.0;
        beta->at(0, 2) = 2.0;
        gamma->at(0, 2) = 1.0;
        zeta->at(0, 2) = 2.0;
    }

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> v;
    std::unique_ptr<Mtx> d;
    std::unique_ptr<Mtx> g;
    std::unique_ptr<Mtx> e;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> rho;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> gamma;
    std::unique_ptr<Mtx> zeta;
    std::unique_ptr<Mtx> old_norm;
    std::unique_ptr<Mtx> new_norm;

    std::unique_ptr<Mtx> d_v;
    std::unique_ptr<Mtx> d_d;
    std::unique_ptr<Mtx> d_g;
    std::unique_ptr<Mtx> d_e;
    std::unique_ptr<Mtx> d_alpha;
    std::unique_ptr<Mtx> d_rho;
    std::unique_ptr<Mtx> d_beta;
    std::unique_ptr<Mtx> d_gamma;
    std::unique_ptr<Mtx> d_zeta;
    std::unique_ptr<Mtx> d_old_norm;
    std::unique_ptr<Mtx> d_new_norm;
};


TEST_F(Multigrid, MultigridKCycleStep1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::multigrid::kcycle_step_1(
        ref, alpha.get(), rho.get(), v.get(), g.get(), d.get(), e.get());
    gko::kernels::EXEC_NAMESPACE::multigrid::kcycle_step_1(
        exec, d_alpha.get(), d_rho.get(), d_v.get(), d_g.get(), d_d.get(),
        d_e.get());

    GKO_ASSERT_MTX_NEAR(d_g, g, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_d, d, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_e, e, 1e-14);
}


TEST_F(Multigrid, MultigridKCycleStep2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::multigrid::kcycle_step_2(
        ref, alpha.get(), rho.get(), gamma.get(), beta.get(), zeta.get(),
        d.get(), e.get());
    gko::kernels::EXEC_NAMESPACE::multigrid::kcycle_step_2(
        exec, d_alpha.get(), d_rho.get(), d_gamma.get(), d_beta.get(),
        d_zeta.get(), d_d.get(), d_e.get());

    GKO_ASSERT_MTX_NEAR(d_e, e, 1e-14);
}


TEST_F(Multigrid, MultigridKCycleCheckStopIsEquivalentToRef)
{
    initialize_data();
    bool is_stop_10;
    bool d_is_stop_10;
    bool is_stop_5;
    bool d_is_stop_5;

    gko::kernels::reference::multigrid::kcycle_check_stop(
        ref, old_norm.get(), new_norm.get(), 1.0, is_stop_10);
    gko::kernels::EXEC_NAMESPACE::multigrid::kcycle_check_stop(
        exec, d_old_norm.get(), d_new_norm.get(), 1.0, d_is_stop_10);
    gko::kernels::reference::multigrid::kcycle_check_stop(
        ref, old_norm.get(), new_norm.get(), 0.5, is_stop_5);
    gko::kernels::EXEC_NAMESPACE::multigrid::kcycle_check_stop(
        exec, d_old_norm.get(), d_new_norm.get(), 0.5, d_is_stop_5);

    GKO_ASSERT_EQ(d_is_stop_10, is_stop_10);
    GKO_ASSERT_EQ(d_is_stop_10, true);
    GKO_ASSERT_EQ(d_is_stop_5, is_stop_5);
    GKO_ASSERT_EQ(d_is_stop_5, false);
}
