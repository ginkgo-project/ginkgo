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

#include <ginkgo/core/solver/multigrid.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>


#include "core/solver/multigrid_kernels.hpp"
#include "cuda/test/utils.hpp"


namespace {


class Multigrid : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    Multigrid() : rand_engine(30) {}

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
        this->modify_scaler(alpha, rho, beta, gamma, zeta);

        d_v = Mtx::create(cuda);
        d_v->copy_from(v.get());
        d_d = Mtx::create(cuda);
        d_d->copy_from(d.get());
        d_g = Mtx::create(cuda);
        d_g->copy_from(g.get());
        d_e = Mtx::create(cuda);
        d_e->copy_from(e.get());
        d_alpha = Mtx::create(cuda);
        d_alpha->copy_from(alpha.get());
        d_rho = Mtx::create(cuda);
        d_rho->copy_from(rho.get());
        d_beta = Mtx::create(cuda);
        d_beta->copy_from(beta.get());
        d_gamma = Mtx::create(cuda);
        d_gamma->copy_from(gamma.get());
        d_zeta = Mtx::create(cuda);
        d_zeta->copy_from(zeta.get());
        d_old_norm = Mtx::create(cuda);
        d_old_norm->copy_from(old_norm.get());
        d_new_norm = Mtx::create(cuda);
        d_new_norm->copy_from(new_norm.get());
    }

    void modify_norm(std::unique_ptr<Mtx> &old_norm,
                     std::unique_ptr<Mtx> &new_norm)
    {
        double ratio = 0.7;
        for (gko::size_type i = 0; i < old_norm->get_size()[1]; i++) {
            old_norm->at(0, i) = gko::abs(old_norm->at(0, i));
            new_norm->at(0, i) = ratio * old_norm->at(0, i);
        }
    }

    void modify_scaler(std::unique_ptr<Mtx> &alpha, std::unique_ptr<Mtx> &rho,
                       std::unique_ptr<Mtx> &beta, std::unique_ptr<Mtx> &gamma,
                       std::unique_ptr<Mtx> &zeta)
    {
        // modify the first three element such that the isfinite condition can
        // be reached, which are checked in the last three group in reference
        // test.
        // scaler_d = zeta/(beta - gamma * gamma / rho)
        // scaler_e = one<ValueType>() - gamma / alpha * scaler_d
        // temp = alpha/rho

        // scaler_d, scaler_e are not finite
        alpha->at(0, 0) = 3.0;
        rho->at(0, 0) = 2.0;
        beta->at(0, 0) = 2.0;
        gamma->at(0, 0) = 2.0;
        zeta->at(0, 0) = -1.0;

        // temp, scaler_d, scaler_e are not finite
        alpha->at(0, 1) = 0.0;
        rho->at(0, 1) = 0.0;
        beta->at(0, 1) = -1.0;
        gamma->at(0, 1) = 0.0;
        zeta->at(0, 1) = 3.0;

        // scaler_e is not finite
        alpha->at(0, 2) = 0.0;
        rho->at(0, 2) = 1.0;
        beta->at(0, 2) = 2.0;
        gamma->at(0, 2) = 1.0;
        zeta->at(0, 2) = 2.0;
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> cuda;

    std::ranlux48 rand_engine;

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


TEST_F(Multigrid, CudaMultigridKCycleStep1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::multigrid::kcycle_step_1(
        ref, gko::lend(alpha), gko::lend(rho), gko::lend(v), gko::lend(g),
        gko::lend(d), gko::lend(e));
    gko::kernels::cuda::multigrid::kcycle_step_1(
        cuda, gko::lend(d_alpha), gko::lend(d_rho), gko::lend(d_v),
        gko::lend(d_g), gko::lend(d_d), gko::lend(d_e));

    GKO_ASSERT_MTX_NEAR(d_g, g, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_d, d, 1e-14);
    GKO_ASSERT_MTX_NEAR(d_e, e, 1e-14);
}


TEST_F(Multigrid, CudaMultigridKCycleStep2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::multigrid::kcycle_step_2(
        ref, gko::lend(alpha), gko::lend(rho), gko::lend(gamma),
        gko::lend(beta), gko::lend(zeta), gko::lend(d), gko::lend(e));
    gko::kernels::cuda::multigrid::kcycle_step_2(
        cuda, gko::lend(d_alpha), gko::lend(d_rho), gko::lend(d_gamma),
        gko::lend(d_beta), gko::lend(d_zeta), gko::lend(d_d), gko::lend(d_e));

    GKO_ASSERT_MTX_NEAR(d_e, e, 1e-14);
}


TEST_F(Multigrid, CudaMultigridKCycleCheckStopIsEquivalentToRef)
{
    initialize_data();
    bool is_stop_10;
    bool d_is_stop_10;
    bool is_stop_5;
    bool d_is_stop_5;

    gko::kernels::reference::multigrid::kcycle_check_stop(
        ref, gko::lend(old_norm), gko::lend(new_norm), 1.0, is_stop_10);
    gko::kernels::cuda::multigrid::kcycle_check_stop(
        cuda, gko::lend(d_old_norm), gko::lend(d_new_norm), 1.0, d_is_stop_10);
    gko::kernels::reference::multigrid::kcycle_check_stop(
        ref, gko::lend(old_norm), gko::lend(new_norm), 0.5, is_stop_5);
    gko::kernels::cuda::multigrid::kcycle_check_stop(
        cuda, gko::lend(d_old_norm), gko::lend(d_new_norm), 0.5, d_is_stop_5);

    GKO_ASSERT_EQ(d_is_stop_10, is_stop_10);
    GKO_ASSERT_EQ(d_is_stop_10, true);
    GKO_ASSERT_EQ(d_is_stop_5, is_stop_5);
    GKO_ASSERT_EQ(d_is_stop_5, false);
}


}  // namespace
