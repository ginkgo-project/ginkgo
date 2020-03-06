/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/solver/bicgstab.hpp>


#include <gtest/gtest.h>


#include <random>


#include <core/solver/bicgstab_kernels.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>


#include "hip/test/utils.hip.hpp"


namespace {


class Bicgstab : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using Solver = gko::solver::Bicgstab<>;

    Bicgstab() : rand_engine(30) {}

    void SetUp()
    {
        ASSERT_GT(gko::HipExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        hip = gko::HipExecutor::create(0, ref);

        mtx = gen_mtx(123, 123);
        make_diag_dominant(mtx.get());
        d_mtx = Mtx::create(hip);
        d_mtx->copy_from(mtx.get());

        hip_bicgstab_factory =
            Solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(246u).on(hip),
                    gko::stop::ResidualNormReduction<>::build()
                        .with_reduction_factor(1e-15)
                        .on(hip))
                .on(hip);
        ref_bicgstab_factory =
            Solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(246u).on(ref),
                    gko::stop::ResidualNormReduction<>::build()
                        .with_reduction_factor(1e-15)
                        .on(ref))
                .on(ref);
    }

    void TearDown()
    {
        if (hip != nullptr) {
            ASSERT_NO_THROW(hip->synchronize());
        }
    }

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(0.0, 1.0), rand_engine, ref);
    }

    void initialize_data()
    {
        int m = 597;
        int n = 17;
        x = gen_mtx(m, n);
        b = gen_mtx(m, n);
        r = gen_mtx(m, n);
        z = gen_mtx(m, n);
        p = gen_mtx(m, n);
        rr = gen_mtx(m, n);
        s = gen_mtx(m, n);
        t = gen_mtx(m, n);
        y = gen_mtx(m, n);
        v = gen_mtx(m, n);
        prev_rho = gen_mtx(1, n);
        rho = gen_mtx(1, n);
        alpha = gen_mtx(1, n);
        beta = gen_mtx(1, n);
        gamma = gen_mtx(1, n);
        omega = gen_mtx(1, n);
        stop_status = std::unique_ptr<gko::Array<gko::stopping_status>>(
            new gko::Array<gko::stopping_status>(ref, n));
        for (size_t i = 0; i < n; ++i) {
            stop_status->get_data()[i].reset();
        }

        d_x = Mtx::create(hip);
        d_b = Mtx::create(hip);
        d_r = Mtx::create(hip);
        d_z = Mtx::create(hip);
        d_p = Mtx::create(hip);
        d_t = Mtx::create(hip);
        d_s = Mtx::create(hip);
        d_y = Mtx::create(hip);
        d_v = Mtx::create(hip);
        d_rr = Mtx::create(hip);
        d_prev_rho = Mtx::create(hip);
        d_rho = Mtx::create(hip);
        d_alpha = Mtx::create(hip);
        d_beta = Mtx::create(hip);
        d_gamma = Mtx::create(hip);
        d_omega = Mtx::create(hip);
        d_stop_status = std::unique_ptr<gko::Array<gko::stopping_status>>(
            new gko::Array<gko::stopping_status>(hip));
        d_stop_status = std::unique_ptr<gko::Array<gko::stopping_status>>(
            new gko::Array<gko::stopping_status>(hip));

        d_x->copy_from(x.get());
        d_b->copy_from(b.get());
        d_r->copy_from(r.get());
        d_z->copy_from(z.get());
        d_p->copy_from(p.get());
        d_v->copy_from(v.get());
        d_y->copy_from(y.get());
        d_t->copy_from(t.get());
        d_s->copy_from(s.get());
        d_rr->copy_from(rr.get());
        d_prev_rho->copy_from(prev_rho.get());
        d_rho->copy_from(rho.get());
        d_alpha->copy_from(alpha.get());
        d_beta->copy_from(beta.get());
        d_gamma->copy_from(gamma.get());
        d_omega->copy_from(omega.get());
        *d_stop_status =
            *stop_status;  // copy_from is not a public member function of Array
    }

    void make_diag_dominant(Mtx *mtx)
    {
        using std::abs;
        for (int i = 0; i < mtx->get_size()[0]; ++i) {
            auto sum = gko::zero<Mtx::value_type>();
            for (int j = 0; j < mtx->get_size()[1]; ++j) {
                sum += abs(mtx->at(i, j));
            }
            mtx->at(i, i) = sum;
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::HipExecutor> hip;

    std::ranlux48 rand_engine;

    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> d_mtx;
    std::unique_ptr<Solver::Factory> hip_bicgstab_factory;
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
    std::unique_ptr<gko::Array<gko::stopping_status>> stop_status;

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
    std::unique_ptr<gko::Array<gko::stopping_status>> d_stop_status;
};


TEST_F(Bicgstab, HipBicgstabInitializeIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicgstab::initialize(
        ref, b.get(), r.get(), rr.get(), y.get(), s.get(), t.get(), z.get(),
        v.get(), p.get(), prev_rho.get(), rho.get(), alpha.get(), beta.get(),
        gamma.get(), omega.get(), stop_status.get());
    gko::kernels::hip::bicgstab::initialize(
        hip, d_b.get(), d_r.get(), d_rr.get(), d_y.get(), d_s.get(), d_t.get(),
        d_z.get(), d_v.get(), d_p.get(), d_prev_rho.get(), d_rho.get(),
        d_alpha.get(), d_beta.get(), d_gamma.get(), d_omega.get(),
        d_stop_status.get());

    GKO_EXPECT_MTX_NEAR(d_r, r, 1e-14);
    GKO_EXPECT_MTX_NEAR(d_z, z, 1e-14);
    GKO_EXPECT_MTX_NEAR(d_p, p, 1e-14);
    GKO_EXPECT_MTX_NEAR(d_y, y, 1e-14);
    GKO_EXPECT_MTX_NEAR(d_t, t, 1e-14);
    GKO_EXPECT_MTX_NEAR(d_s, s, 1e-14);
    GKO_EXPECT_MTX_NEAR(d_rr, rr, 1e-14);
    GKO_EXPECT_MTX_NEAR(d_v, v, 1e-14);
    GKO_EXPECT_MTX_NEAR(d_prev_rho, prev_rho, 1e-14);
    GKO_EXPECT_MTX_NEAR(d_rho, rho, 1e-14);
    GKO_EXPECT_MTX_NEAR(d_alpha, alpha, 1e-14);
    GKO_EXPECT_MTX_NEAR(d_beta, beta, 1e-14);
    GKO_EXPECT_MTX_NEAR(d_gamma, gamma, 1e-14);
    GKO_EXPECT_MTX_NEAR(d_omega, omega, 1e-14);
    GKO_ASSERT_ARRAY_EQ(d_stop_status, stop_status);
}


TEST_F(Bicgstab, HipBicgstabStep1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicgstab::step_1(
        ref, r.get(), p.get(), v.get(), rho.get(), prev_rho.get(), alpha.get(),
        omega.get(), stop_status.get());
    gko::kernels::hip::bicgstab::step_1(
        hip, d_r.get(), d_p.get(), d_v.get(), d_rho.get(), d_prev_rho.get(),
        d_alpha.get(), d_omega.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_p, p, 1e-14);
}


TEST_F(Bicgstab, HipBicgstabStep2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicgstab::step_2(ref, r.get(), s.get(), v.get(),
                                              rho.get(), alpha.get(),
                                              beta.get(), stop_status.get());
    gko::kernels::hip::bicgstab::step_2(hip, d_r.get(), d_s.get(), d_v.get(),
                                        d_rho.get(), d_alpha.get(),
                                        d_beta.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_alpha, alpha, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_s, s, 1e-14);
}


TEST_F(Bicgstab, HipBicgstabStep3IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicgstab::step_3(
        ref, x.get(), r.get(), s.get(), t.get(), y.get(), z.get(), alpha.get(),
        beta.get(), gamma.get(), omega.get(), stop_status.get());
    gko::kernels::hip::bicgstab::step_3(
        hip, d_x.get(), d_r.get(), d_s.get(), d_t.get(), d_y.get(), d_z.get(),
        d_alpha.get(), d_beta.get(), d_gamma.get(), d_omega.get(),
        d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_omega, omega, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_r, r, 1e-14);
}


TEST_F(Bicgstab, HipBicgstabApplyOneRHSIsEquivalentToRef)
{
    int m = 123;
    int n = 1;
    auto ref_solver = ref_bicgstab_factory->generate(mtx);
    auto hip_solver = hip_bicgstab_factory->generate(d_mtx);
    auto b = gen_mtx(m, n);
    auto x = gen_mtx(m, n);
    auto d_b = Mtx::create(hip);
    auto d_x = Mtx::create(hip);
    d_b->copy_from(b.get());
    d_x->copy_from(x.get());

    ref_solver->apply(b.get(), x.get());
    hip_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_b, b, 1e-13);
    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-13);
}


TEST_F(Bicgstab, HipBicgstabApplyMultipleRHSIsEquivalentToRef)
{
    int m = 123;
    int n = 16;
    auto hip_solver = hip_bicgstab_factory->generate(d_mtx);
    auto ref_solver = ref_bicgstab_factory->generate(mtx);
    auto b = gen_mtx(m, n);
    auto x = gen_mtx(m, n);
    auto d_b = Mtx::create(hip);
    auto d_x = Mtx::create(hip);
    d_b->copy_from(b.get());
    d_x->copy_from(x.get());

    ref_solver->apply(b.get(), x.get());
    hip_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_b, b, 1e-13);
    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-13);
}


}  // namespace
