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

#include <ginkgo/core/solver/cg.hpp>


#include <gtest/gtest.h>


#include <random>


#include <core/solver/cg_kernels.hpp>
#include <core/test/utils.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>

namespace {


class Cg : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    Cg() : rand_engine(30) {}

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        cuda = gko::CudaExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

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
        b = gen_mtx(m, n);
        r = gen_mtx(m, n);
        z = gen_mtx(m, n);
        p = gen_mtx(m, n);
        q = gen_mtx(m, n);
        x = gen_mtx(m, n);
        beta = gen_mtx(1, n);
        prev_rho = gen_mtx(1, n);
        rho = gen_mtx(1, n);
        stop_status = std::unique_ptr<gko::Array<gko::stopping_status>>(
            new gko::Array<gko::stopping_status>(ref, n));
        for (size_t i = 0; i < stop_status->get_num_elems(); ++i) {
            stop_status->get_data()[i].reset();
        }

        d_b = Mtx::create(cuda);
        d_b->copy_from(b.get());
        d_r = Mtx::create(cuda);
        d_r->copy_from(r.get());
        d_z = Mtx::create(cuda);
        d_z->copy_from(z.get());
        d_p = Mtx::create(cuda);
        d_p->copy_from(p.get());
        d_q = Mtx::create(cuda);
        d_q->copy_from(q.get());
        d_x = Mtx::create(cuda);
        d_x->copy_from(x.get());
        d_beta = Mtx::create(cuda);
        d_beta->copy_from(beta.get());
        d_prev_rho = Mtx::create(cuda);
        d_prev_rho->copy_from(prev_rho.get());
        d_rho = Mtx::create(cuda);
        d_rho->copy_from(rho.get());
        d_stop_status = std::unique_ptr<gko::Array<gko::stopping_status>>(
            new gko::Array<gko::stopping_status>(cuda, n));
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
    std::shared_ptr<const gko::CudaExecutor> cuda;

    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> r;
    std::unique_ptr<Mtx> z;
    std::unique_ptr<Mtx> p;
    std::unique_ptr<Mtx> q;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> prev_rho;
    std::unique_ptr<Mtx> rho;
    std::unique_ptr<gko::Array<gko::stopping_status>> stop_status;

    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_r;
    std::unique_ptr<Mtx> d_z;
    std::unique_ptr<Mtx> d_p;
    std::unique_ptr<Mtx> d_q;
    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_beta;
    std::unique_ptr<Mtx> d_prev_rho;
    std::unique_ptr<Mtx> d_rho;
    std::unique_ptr<gko::Array<gko::stopping_status>> d_stop_status;
};


TEST_F(Cg, CudaCgInitializeIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::cg::initialize(ref, b.get(), r.get(), z.get(),
                                            p.get(), q.get(), prev_rho.get(),
                                            rho.get(), stop_status.get());
    gko::kernels::cuda::cg::initialize(cuda, d_b.get(), d_r.get(), d_z.get(),
                                       d_p.get(), d_q.get(), d_prev_rho.get(),
                                       d_rho.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_r, r, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_z, z, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_p, p, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_q, q, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_prev_rho, prev_rho, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_rho, rho, 1e-14);
    GKO_ASSERT_ARRAY_EQ(d_stop_status, stop_status);
}


TEST_F(Cg, CudaCgStep1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::cg::step_1(ref, p.get(), z.get(), rho.get(),
                                        prev_rho.get(), stop_status.get());
    gko::kernels::cuda::cg::step_1(cuda, d_p.get(), d_z.get(), d_rho.get(),
                                   d_prev_rho.get(), d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_p, p, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_z, z, 1e-14);
}


TEST_F(Cg, CudaCgStep2IsEquivalentToRef)
{
    initialize_data();
    gko::kernels::reference::cg::step_2(ref, x.get(), r.get(), p.get(), q.get(),
                                        beta.get(), rho.get(),
                                        stop_status.get());
    gko::kernels::cuda::cg::step_2(cuda, d_x.get(), d_r.get(), d_p.get(),
                                   d_q.get(), d_beta.get(), d_rho.get(),
                                   d_stop_status.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_r, r, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_p, p, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_q, q, 1e-14);
}


TEST_F(Cg, ApplyIsEquivalentToRef)
{
    auto mtx = gen_mtx(50, 50);
    make_spd(mtx.get());
    auto x = gen_mtx(50, 3);
    auto b = gen_mtx(50, 3);
    auto d_mtx = Mtx::create(cuda);
    d_mtx->copy_from(mtx.get());
    auto d_x = Mtx::create(cuda);
    d_x->copy_from(x.get());
    auto d_b = Mtx::create(cuda);
    d_b->copy_from(b.get());
    auto cg_factory =
        gko::solver::Cg<>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(50u).on(ref),
                gko::stop::ResidualNormReduction<>::build()
                    .with_reduction_factor(1e-14)
                    .on(ref))
            .on(ref);
    auto d_cg_factory =
        gko::solver::Cg<>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(50u).on(cuda),
                gko::stop::ResidualNormReduction<>::build()
                    .with_reduction_factor(1e-14)
                    .on(cuda))
            .on(cuda);
    auto solver = cg_factory->generate(std::move(mtx));
    auto d_solver = d_cg_factory->generate(std::move(d_mtx));

    solver->apply(b.get(), x.get());
    d_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


}  // namespace
