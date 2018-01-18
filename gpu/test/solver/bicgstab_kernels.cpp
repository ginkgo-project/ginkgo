/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/solver/bicgstab.hpp>


#include <gtest/gtest.h>


#include <random>


#include <core/base/exception.hpp>
#include <core/base/executor.hpp>
#include <core/matrix/dense.hpp>
#include <core/solver/bicgstab_kernels.hpp>
#include <core/test/utils.hpp>

namespace {


class Bicgstab : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    Bicgstab() : rand_engine(30) {}

    void SetUp()
    {
        ASSERT_GT(gko::GpuExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        gpu = gko::GpuExecutor::create(0, ref);

        mtx = gen_mtx(48, 48);
        make_diag_dominant(mtx.get());
        d_mtx = Mtx::create(gpu);
        d_mtx->copy_from(mtx.get());
        gpu_bicgstab_factory =
            gko::solver::BicgstabFactory<>::create(gpu, 48, 1e-15);
        ref_bicgstab_factory =
            gko::solver::BicgstabFactory<>::create(ref, 48, 1e-15);
    }

    void TearDown()
    {
        if (gpu != nullptr) {
            ASSERT_NO_THROW(gpu->synchronize());
        }
    }

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            ref, num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(0.0, 1.0), rand_engine);
    }

    void make_symetric(Mtx *mtx)
    {
        for (int i = 0; i < mtx->get_num_rows(); ++i) {
            for (int j = i + 1; j < mtx->get_num_cols(); ++j) {
                mtx->at(i, j) = mtx->at(j, i);
            }
        }
    }

    void make_diag_dominant(Mtx *mtx)
    {
        using std::abs;
        for (int i = 0; i < mtx->get_num_rows(); ++i) {
            auto sum = gko::zero<Mtx::value_type>();
            for (int j = 0; j < mtx->get_num_cols(); ++j) {
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
    std::shared_ptr<const gko::GpuExecutor> gpu;

    std::ranlux48 rand_engine;


    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> d_mtx;
    std::unique_ptr<gko::solver::BicgstabFactory<>> gpu_bicgstab_factory;
    std::unique_ptr<gko::solver::BicgstabFactory<>> ref_bicgstab_factory;
};


TEST_F(Bicgstab, GpuBicgstabInitializeIsEquivalentToRef)
{
    int m = 24;
    int n = 7;

    auto b = gen_mtx(m, n);
    auto r = gen_mtx(m, n);
    auto z = gen_mtx(m, n);
    auto p = gen_mtx(m, n);
    auto rr = gen_mtx(m, n);
    auto s = gen_mtx(m, n);
    auto t = gen_mtx(m, n);
    auto y = gen_mtx(m, n);
    auto v = gen_mtx(m, n);
    auto prev_rho = Mtx::create(ref, n, 1);
    auto rho = Mtx::create(ref, n, 1);
    auto alpha = Mtx::create(ref, n, 1);
    auto beta = Mtx::create(ref, n, 1);
    auto omega = Mtx::create(ref, n, 1);

    auto d_b = Mtx::create(gpu);
    auto d_r = Mtx::create(gpu);
    auto d_z = Mtx::create(gpu);
    auto d_p = Mtx::create(gpu);
    auto d_t = Mtx::create(gpu);
    auto d_s = Mtx::create(gpu);
    auto d_y = Mtx::create(gpu);
    auto d_v = Mtx::create(gpu);
    auto d_rr = Mtx::create(gpu);
    auto d_prev_rho = Mtx::create(gpu, n, 1);
    auto d_rho = Mtx::create(gpu, n, 1);
    auto d_alpha = Mtx::create(gpu, n, 1);
    auto d_beta = Mtx::create(gpu, n, 1);
    auto d_omega = Mtx::create(gpu, n, 1);
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
    d_omega->copy_from(omega.get());

    gko::kernels::reference::bicgstab::initialize(
        b.get(), r.get(), rr.get(), y.get(), s.get(), t.get(), z.get(), v.get(),
        p.get(), prev_rho.get(), rho.get(), alpha.get(), beta.get(),
        omega.get());
    gko::kernels::gpu::bicgstab::initialize(
        d_b.get(), d_r.get(), d_rr.get(), d_y.get(), d_s.get(), d_t.get(),
        d_z.get(), d_v.get(), d_p.get(), d_prev_rho.get(), d_rho.get(),
        d_alpha.get(), d_beta.get(), d_omega.get());

    auto b_result = Mtx::create(ref);
    auto r_result = Mtx::create(ref);
    auto z_result = Mtx::create(ref);
    auto p_result = Mtx::create(ref);
    auto rr_result = Mtx::create(ref);
    auto t_result = Mtx::create(ref);
    auto s_result = Mtx::create(ref);
    auto y_result = Mtx::create(ref);
    auto v_result = Mtx::create(ref);
    auto prev_rho_result = Mtx::create(ref);
    auto rho_result = Mtx::create(ref);
    auto alpha_result = Mtx::create(ref);
    auto beta_result = Mtx::create(ref);
    auto omega_result = Mtx::create(ref);
    b_result->copy_from(d_b.get());
    r_result->copy_from(d_r.get());
    z_result->copy_from(d_z.get());
    p_result->copy_from(d_p.get());
    y_result->copy_from(d_y.get());
    rr_result->copy_from(d_rr.get());
    t_result->copy_from(d_t.get());
    s_result->copy_from(d_s.get());
    v_result->copy_from(d_v.get());
    prev_rho_result->copy_from(d_prev_rho.get());
    rho_result->copy_from(d_rho.get());
    alpha_result->copy_from(d_alpha.get());
    beta_result->copy_from(d_beta.get());
    omega_result->copy_from(d_omega.get());

    ASSERT_MTX_NEAR(b_result, b, 1e-14);
    ASSERT_MTX_NEAR(r_result, r, 1e-14);
    ASSERT_MTX_NEAR(z_result, z, 1e-14);
    ASSERT_MTX_NEAR(p_result, p, 1e-14);
    ASSERT_MTX_NEAR(y_result, y, 1e-14);
    ASSERT_MTX_NEAR(t_result, t, 1e-14);
    ASSERT_MTX_NEAR(s_result, s, 1e-14);
    ASSERT_MTX_NEAR(rr_result, rr, 1e-14);
    ASSERT_MTX_NEAR(v_result, v, 1e-14);
    ASSERT_MTX_NEAR(prev_rho_result, prev_rho, 1e-14);
    ASSERT_MTX_NEAR(rho_result, rho, 1e-14);
    ASSERT_MTX_NEAR(alpha_result, alpha, 1e-14);
    ASSERT_MTX_NEAR(beta_result, beta, 1e-14);
    ASSERT_MTX_NEAR(omega_result, omega, 1e-14);
}


TEST_F(Bicgstab, GpuBicgstabStep1IsEquivalentToRef)
{
    int m = 24;
    int n = 7;

    auto p = gen_mtx(m, n);
    auto r = gen_mtx(m, n);
    auto v = gen_mtx(m, n);
    auto rho = gen_mtx(n, 1);
    auto alpha = gen_mtx(n, 1);
    auto omega = gen_mtx(n, 1);
    auto prev_rho = gen_mtx(n, 1);

    auto d_p = Mtx::create(gpu);
    auto d_r = Mtx::create(gpu);
    auto d_v = Mtx::create(gpu);
    auto d_prev_rho = Mtx::create(gpu, n, 1);
    auto d_rho = Mtx::create(gpu, n, 1);
    auto d_alpha = Mtx::create(gpu, n, 1);
    auto d_omega = Mtx::create(gpu, n, 1);
    d_p->copy_from(p.get());
    d_r->copy_from(r.get());
    d_v->copy_from(v.get());
    d_rho->copy_from(rho.get());
    d_alpha->copy_from(alpha.get());
    d_omega->copy_from(omega.get());
    d_prev_rho->copy_from(prev_rho.get());

    gko::kernels::reference::bicgstab::step_1(r.get(), p.get(), v.get(),
                                              rho.get(), prev_rho.get(),
                                              alpha.get(), omega.get());
    gko::kernels::gpu::bicgstab::step_1(d_r.get(), d_p.get(), d_v.get(),
                                        d_rho.get(), d_prev_rho.get(),
                                        d_alpha.get(), d_omega.get());

    auto p_result = Mtx::create(ref);
    auto r_result = Mtx::create(ref);
    auto v_result = Mtx::create(ref);
    p_result->copy_from(d_p.get());
    r_result->copy_from(d_r.get());
    v_result->copy_from(d_v.get());

    ASSERT_MTX_NEAR(p_result, p, 1e-14);
    ASSERT_MTX_NEAR(r_result, r, 1e-14);
    ASSERT_MTX_NEAR(v_result, v, 1e-14);
}


TEST_F(Bicgstab, GpuBicgstabStep2IsEquivalentToRef)
{
    int m = 24;
    int n = 7;

    auto s = gen_mtx(m, n);
    auto r = gen_mtx(m, n);
    auto v = gen_mtx(m, n);
    auto beta = gen_mtx(n, 1);
    auto alpha = gen_mtx(n, 1);
    auto rho = gen_mtx(n, 1);

    auto d_s = Mtx::create(gpu);
    auto d_r = Mtx::create(gpu);
    auto d_v = Mtx::create(gpu);
    auto d_beta = Mtx::create(gpu, n, 1);
    auto d_alpha = Mtx::create(gpu, n, 1);
    auto d_rho = Mtx::create(gpu, n, 1);
    d_s->copy_from(s.get());
    d_r->copy_from(r.get());
    d_v->copy_from(v.get());
    d_alpha->copy_from(alpha.get());
    d_beta->copy_from(beta.get());
    d_rho->copy_from(rho.get());

    gko::kernels::reference::bicgstab::step_2(
        r.get(), s.get(), v.get(), rho.get(), alpha.get(), beta.get());
    gko::kernels::gpu::bicgstab::step_2(d_r.get(), d_s.get(), d_v.get(),
                                        d_rho.get(), d_alpha.get(),
                                        d_beta.get());

    auto s_result = Mtx::create(ref);
    auto r_result = Mtx::create(ref);
    auto v_result = Mtx::create(ref);
    auto alpha_result = Mtx::create(ref);
    s_result->copy_from(d_s.get());
    r_result->copy_from(d_r.get());
    v_result->copy_from(d_v.get());
    alpha_result->copy_from(d_alpha.get());

    ASSERT_MTX_NEAR(s_result, s, 1e-14);
    ASSERT_MTX_NEAR(r_result, r, 1e-14);
    ASSERT_MTX_NEAR(v_result, v, 1e-14);
    ASSERT_MTX_NEAR(alpha_result, alpha, 1e-14);
}


TEST_F(Bicgstab, GpuBicgstabStep3IsEquivalentToRef)
{
    int m = 24;
    int n = 7;

    auto x = gen_mtx(m, n);
    auto r = gen_mtx(m, n);
    auto y = gen_mtx(m, n);
    auto z = gen_mtx(m, n);
    auto s = gen_mtx(m, n);
    auto t = gen_mtx(m, n);
    auto beta = gen_mtx(n, 1);
    auto alpha = gen_mtx(n, 1);
    auto omega = gen_mtx(n, 1);

    auto d_x = Mtx::create(gpu);
    auto d_r = Mtx::create(gpu);
    auto d_y = Mtx::create(gpu);
    auto d_z = Mtx::create(gpu);
    auto d_s = Mtx::create(gpu);
    auto d_t = Mtx::create(gpu);
    auto d_beta = Mtx::create(gpu, n, 1);
    auto d_alpha = Mtx::create(gpu, n, 1);
    auto d_omega = Mtx::create(gpu, n, 1);
    d_x->copy_from(x.get());
    d_r->copy_from(r.get());
    d_y->copy_from(y.get());
    d_z->copy_from(z.get());
    d_s->copy_from(s.get());
    d_t->copy_from(t.get());
    d_beta->copy_from(beta.get());
    d_alpha->copy_from(alpha.get());
    d_omega->copy_from(omega.get());

    gko::kernels::reference::bicgstab::step_3(
        x.get(), r.get(), s.get(), t.get(), y.get(), z.get(), alpha.get(),
        beta.get(), omega.get());
    gko::kernels::gpu::bicgstab::step_3(
        d_x.get(), d_r.get(), d_s.get(), d_t.get(), d_y.get(), d_z.get(),
        d_alpha.get(), d_beta.get(), d_omega.get());

    auto x_result = Mtx::create(ref);
    auto r_result = Mtx::create(ref);
    auto s_result = Mtx::create(ref);
    auto t_result = Mtx::create(ref);
    auto y_result = Mtx::create(ref);
    auto z_result = Mtx::create(ref);
    auto omega_result = Mtx::create(ref);
    x_result->copy_from(d_x.get());
    r_result->copy_from(d_r.get());
    s_result->copy_from(d_s.get());
    t_result->copy_from(d_t.get());
    y_result->copy_from(d_y.get());
    z_result->copy_from(d_z.get());
    omega_result->copy_from(d_omega.get());

    ASSERT_MTX_NEAR(x_result, x, 1e-14);
    ASSERT_MTX_NEAR(r_result, r, 1e-14);
    ASSERT_MTX_NEAR(s_result, s, 1e-14);
    ASSERT_MTX_NEAR(t_result, t, 1e-14);
    ASSERT_MTX_NEAR(y_result, y, 1e-14);
    ASSERT_MTX_NEAR(z_result, z, 1e-14);
    ASSERT_MTX_NEAR(omega_result, omega, 1e-14);
}

TEST_F(Bicgstab, GpuBicgstabApplyOneRHSIsEquivalentToRef)
{
    int m = 48;
    int n = 1;

    auto gpu_solver = gpu_bicgstab_factory->generate(d_mtx);
    auto ref_solver = ref_bicgstab_factory->generate(mtx);

    auto b = gen_mtx(m, n);
    auto x = gen_mtx(m, n);
    auto d_b = Mtx::create(gpu);
    auto d_x = Mtx::create(gpu);
    d_b->copy_from(b.get());
    d_x->copy_from(x.get());

    gpu_solver->apply(d_b.get(), d_x.get());
    ref_solver->apply(b.get(), x.get());


    auto b_result = Mtx::create(ref);
    auto x_result = Mtx::create(ref);
    b_result->copy_from(d_b.get());
    x_result->copy_from(d_x.get());
    ASSERT_MTX_NEAR(b_result, b, 1e-13);
    ASSERT_MTX_NEAR(x_result, x, 1e-13);
}

TEST_F(Bicgstab, GpuBicgstabApplyMultipleRHSIsEquivalentToRef)
{
    int m = 48;
    int n = 16;

    auto gpu_solver = gpu_bicgstab_factory->generate(d_mtx);
    auto ref_solver = ref_bicgstab_factory->generate(mtx);

    auto b = gen_mtx(m, n);
    auto x = gen_mtx(m, n);
    auto d_b = Mtx::create(gpu);
    auto d_x = Mtx::create(gpu);
    d_b->copy_from(b.get());
    d_x->copy_from(x.get());

    gpu_solver->apply(d_b.get(), d_x.get());
    ref_solver->apply(b.get(), x.get());


    auto b_result = Mtx::create(ref);
    auto x_result = Mtx::create(ref);
    b_result->copy_from(d_b.get());
    x_result->copy_from(d_x.get());
    ASSERT_MTX_NEAR(b_result, b, 1e-13);
    ASSERT_MTX_NEAR(x_result, x, 1e-13);
}

}  // namespace
