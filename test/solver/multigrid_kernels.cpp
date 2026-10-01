// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/multigrid_kernels.hpp"

#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/multigrid.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/common_fixture.hpp"


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
        ref, alpha->get_const_device_view(), rho->get_const_device_view(),
        v->get_const_device_view(), g->get_device_view(), d->get_device_view(),
        e->get_device_view());
    gko::kernels::GKO_DEVICE_NAMESPACE::multigrid::kcycle_step_1(
        exec, d_alpha->get_const_device_view(), d_rho->get_const_device_view(),
        d_v->get_const_device_view(), d_g->get_device_view(),
        d_d->get_device_view(), d_e->get_device_view());

    GKO_ASSERT_MTX_NEAR(d_g, g, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_d, d, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_e, e, 1e-14);
}


TEST_F(Multigrid, MultigridKCycleStep2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::multigrid::kcycle_step_2(
        ref, alpha->get_const_device_view(), rho->get_const_device_view(),
        gamma->get_const_device_view(), beta->get_const_device_view(),
        zeta->get_const_device_view(), d->get_const_device_view(),
        e->get_device_view());
    gko::kernels::GKO_DEVICE_NAMESPACE::multigrid::kcycle_step_2(
        exec, d_alpha->get_const_device_view(), d_rho->get_const_device_view(),
        d_gamma->get_const_device_view(), d_beta->get_const_device_view(),
        d_zeta->get_const_device_view(), d_d->get_const_device_view(),
        d_e->get_device_view());

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
        ref, old_norm->get_const_device_view(),
        new_norm->get_const_device_view(), 1.0, is_stop_10);
    gko::kernels::GKO_DEVICE_NAMESPACE::multigrid::kcycle_check_stop(
        exec, d_old_norm->get_const_device_view(),
        d_new_norm->get_const_device_view(), 1.0, d_is_stop_10);
    gko::kernels::reference::multigrid::kcycle_check_stop(
        ref, old_norm->get_const_device_view(),
        new_norm->get_const_device_view(), 0.5, is_stop_5);
    gko::kernels::GKO_DEVICE_NAMESPACE::multigrid::kcycle_check_stop(
        exec, d_old_norm->get_const_device_view(),
        d_new_norm->get_const_device_view(), 0.5, d_is_stop_5);

    GKO_ASSERT_EQ(d_is_stop_10, is_stop_10);
    GKO_ASSERT_EQ(d_is_stop_10, true);
    GKO_ASSERT_EQ(d_is_stop_5, is_stop_5);
    GKO_ASSERT_EQ(d_is_stop_5, false);
}


// Multigrid::update_matrix_value runs the Pgm mapping and regenerates the
// smoothers on whichever executor the solver lives on, so the whole update
// has to give the same solver as on the reference executor. test/multigrid
// only covers the Pgm level itself.
class MultigridUpdate : public CommonTestFixture {
protected:
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using Coarse = gko::multigrid::Pgm<value_type, index_type>;
    using Smoother = gko::solver::Ir<value_type>;
    using InnerSolver = gko::preconditioner::Jacobi<value_type>;

    MultigridUpdate() : rand_engine(42)
    {
#ifdef GINKGO_FAST_TESTS
        const int m = 129;
#else
        const int m = 597;
#endif
        auto data =
            gko::test::generate_random_matrix_data<value_type, index_type>(
                m, m, std::uniform_int_distribution<>(m / 20, m / 10),
                std::normal_distribution<value_type>(-1.0, 1.0), rand_engine);
        gko::utils::make_hpd(data);
        mtx = gko::share(Csr::create(ref));
        mtx->read(data);
        // scaling by a constant keeps the sparsity pattern and the
        // aggregates, which is exactly what the update relies on
        scaled = gko::share(gko::clone(mtx));
        scaled->scale(gko::initialize<Vec>({value_type{2}}, ref));
        d_mtx = gko::share(gko::clone(exec, mtx));
        d_scaled = gko::share(gko::clone(exec, scaled));
        b = gko::test::generate_random_matrix<Vec>(
            m, 1, std::uniform_int_distribution<>(1, 1),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
        d_b = gko::clone(exec, b);
    }

    std::unique_ptr<gko::solver::Multigrid::Factory> gen_factory(
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::solver::Multigrid::build()
            .with_max_levels(2u)
            .with_min_coarse_rows(8u)
            .with_post_uses_pre(true)
            .with_mg_level(
                Coarse::build().with_deterministic(true).with_updatable_values(
                    true))
            .with_pre_smoother(
                Smoother::build()
                    .with_solver(InnerSolver::build().with_max_block_size(1u))
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u)))
            .with_criteria(gko::stop::Iteration::build().with_max_iters(2u))
            .on(exec);
    }

    std::default_random_engine rand_engine;
    std::shared_ptr<Csr> mtx;
    std::shared_ptr<Csr> scaled;
    std::shared_ptr<Csr> d_mtx;
    std::shared_ptr<Csr> d_scaled;
    std::unique_ptr<Vec> b;
    std::unique_ptr<Vec> d_b;
};


TEST_F(MultigridUpdate, UpdateMatrixValueIsEquivalentToRef)
{
    auto solver = gen_factory(ref)->generate(mtx);
    auto d_solver = gen_factory(exec)->generate(d_mtx);
    auto x = Vec::create(ref, gko::dim<2>{b->get_size()[0], 1});
    x->fill(gko::zero<value_type>());
    auto d_x = gko::clone(exec, x);

    solver->update_matrix_value(scaled);
    d_solver->update_matrix_value(d_scaled);
    solver->apply(b, x);
    d_solver->apply(d_b, d_x);

    auto mg_level = solver->get_mg_level_list();
    auto d_mg_level = d_solver->get_mg_level_list();
    ASSERT_GT(mg_level.size(), 0);
    ASSERT_EQ(mg_level.size(), d_mg_level.size());
    for (gko::size_type i = 0; i < mg_level.size(); i++) {
        GKO_ASSERT_MTX_NEAR(gko::as<Csr>(d_mg_level.at(i)->get_coarse_op()),
                            gko::as<Csr>(mg_level.at(i)->get_coarse_op()),
                            r<value_type>::value);
    }
    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value * 1e3);
}


// An updated solver has to behave like one generated on the new matrix from
// scratch, on the device as well.
TEST_F(MultigridUpdate, UpdatedSolverAppliesLikeRegeneratedSolver)
{
    auto d_solver = gen_factory(exec)->generate(d_mtx);
    auto d_expected = gen_factory(exec)->generate(d_scaled);
    auto d_x = Vec::create(exec, gko::dim<2>{b->get_size()[0], 1});
    d_x->fill(gko::zero<value_type>());
    auto d_expected_x = gko::clone(d_x);

    d_solver->update_matrix_value(d_scaled);
    d_solver->apply(d_b, d_x);
    d_expected->apply(d_b, d_expected_x);

    GKO_ASSERT_MTX_NEAR(d_x, d_expected_x, r<value_type>::value * 1e3);
}
