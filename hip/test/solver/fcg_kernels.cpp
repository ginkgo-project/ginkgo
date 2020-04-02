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

#include <ginkgo/core/solver/fcg.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>


#include "core/solver/fcg_kernels.hpp"
#include "hip/test/utils.hip.hpp"


namespace {


class Fcg : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using Solver = gko::solver::Fcg<>;

    Fcg() : rand_engine(30) {}

    void SetUp()
    {
        ASSERT_GT(gko::HipExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        hip = gko::HipExecutor::create(0, ref);
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
        int n = 43;
        b = gen_mtx(m, n);
        r = gen_mtx(m, n);
        t = gen_mtx(m, n);
        z = gen_mtx(m, n);
        p = gen_mtx(m, n);
        q = gen_mtx(m, n);
        x = gen_mtx(m, n);
        beta = gen_mtx(1, n);
        prev_rho = gen_mtx(1, n);
        rho = gen_mtx(1, n);
        rho_t = gen_mtx(1, n);
        stop_status = std::unique_ptr<gko::Array<gko::stopping_status>>(
            new gko::Array<gko::stopping_status>(ref, n));
        for (size_t i = 0; i < stop_status->get_num_elems(); ++i) {
            stop_status->get_data()[i].reset();
        }

        d_b = Mtx::create(hip);
        d_b->copy_from(b.get());
        d_r = Mtx::create(hip);
        d_r->copy_from(r.get());
        d_t = Mtx::create(hip);
        d_t->copy_from(t.get());
        d_z = Mtx::create(hip);
        d_z->copy_from(z.get());
        d_p = Mtx::create(hip);
        d_p->copy_from(p.get());
        d_q = Mtx::create(hip);
        d_q->copy_from(q.get());
        d_x = Mtx::create(hip);
        d_x->copy_from(x.get());
        d_beta = Mtx::create(hip);
        d_beta->copy_from(beta.get());
        d_prev_rho = Mtx::create(hip);
        d_prev_rho->copy_from(prev_rho.get());
        d_rho_t = Mtx::create(hip);
        d_rho_t->copy_from(rho_t.get());
        d_rho = Mtx::create(hip);
        d_rho->copy_from(rho.get());
        d_stop_status = std::unique_ptr<gko::Array<gko::stopping_status>>(
            new gko::Array<gko::stopping_status>(hip, n));
        *d_stop_status = *stop_status;
    }

    void make_symetric(Mtx *mtx)
    {
        for (int i = 0; i < mtx->get_size()[0]; ++i) {
            for (int j = i + 1; j < mtx->get_size()[1]; ++j) {
                mtx->at(i, j) = mtx->at(j, i);
            }
        }
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

    void make_spd(Mtx *mtx)
    {
        make_symetric(mtx);
        make_diag_dominant(mtx);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::HipExecutor> hip;

    std::ranlux48 rand_engine;

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
    std::unique_ptr<gko::Array<gko::stopping_status>> stop_status;

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
    std::unique_ptr<gko::Array<gko::stopping_status>> d_stop_status;
};


TEST_F(Fcg, HipFcgInitializeIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::fcg::initialize(
        ref, b.get(), r.get(), z.get(), p.get(), q.get(), t.get(),
        prev_rho.get(), rho.get(), rho_t.get(), stop_status.get());
    gko::kernels::hip::fcg::initialize(
        hip, d_b.get(), d_r.get(), d_z.get(), d_p.get(), d_q.get(), d_t.get(),
        d_prev_rho.get(), d_rho.get(), d_rho_t.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_r, r, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_t, t, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_z, z, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_p, p, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_q, q, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_prev_rho, prev_rho, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_rho, rho, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_rho_t, rho_t, 1e-14);
    GKO_ASSERT_ARRAY_EQ(d_stop_status, stop_status);
}


TEST_F(Fcg, HipFcgStep1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::fcg::step_1(ref, p.get(), z.get(), rho_t.get(),
                                         prev_rho.get(), stop_status.get());
    gko::kernels::hip::fcg::step_1(hip, d_p.get(), d_z.get(), d_rho_t.get(),
                                   d_prev_rho.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_p, p, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_z, z, 1e-14);
}


TEST_F(Fcg, HipFcgStep2IsEquivalentToRef)
{
    initialize_data();
    gko::kernels::reference::fcg::step_2(ref, x.get(), r.get(), t.get(),
                                         p.get(), q.get(), beta.get(),
                                         rho.get(), stop_status.get());
    gko::kernels::hip::fcg::step_2(hip, d_x.get(), d_r.get(), d_t.get(),
                                   d_p.get(), d_q.get(), d_beta.get(),
                                   d_rho.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_r, r, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_t, t, 1e-14);
}


TEST_F(Fcg, ApplyIsEquivalentToRef)
{
    auto mtx = gen_mtx(50, 50);
    make_spd(mtx.get());
    auto x = gen_mtx(50, 3);
    auto b = gen_mtx(50, 3);
    auto d_mtx = Mtx::create(hip);
    d_mtx->copy_from(mtx.get());
    auto d_x = Mtx::create(hip);
    d_x->copy_from(x.get());
    auto d_b = Mtx::create(hip);
    d_b->copy_from(b.get());
    auto fcg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(50u).on(ref),
                gko::stop::ResidualNormReduction<>::build()
                    .with_reduction_factor(1e-14)
                    .on(ref))
            .on(ref);
    auto d_fcg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(50u).on(hip),
                gko::stop::ResidualNormReduction<>::build()
                    .with_reduction_factor(1e-14)
                    .on(hip))
            .on(hip);
    auto solver = fcg_factory->generate(std::move(mtx));
    auto d_solver = d_fcg_factory->generate(std::move(d_mtx));

    solver->apply(b.get(), x.get());
    d_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


}  // namespace
