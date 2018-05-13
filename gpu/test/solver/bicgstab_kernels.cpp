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

        mtx = gen_mtx(123, 123);
        make_diag_dominant(mtx.get());
        d_mtx = Mtx::create(gpu);
        d_mtx->copy_from(mtx.get());
        gpu_bicgstab_factory =
            gko::solver::BicgstabFactory<>::create(gpu, 246, 1e-15);
        ref_bicgstab_factory =
            gko::solver::BicgstabFactory<>::create(ref, 246, 1e-15);
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

        d_x = Mtx::create(gpu);
        d_b = Mtx::create(gpu);
        d_r = Mtx::create(gpu);
        d_z = Mtx::create(gpu);
        d_p = Mtx::create(gpu);
        d_t = Mtx::create(gpu);
        d_s = Mtx::create(gpu);
        d_y = Mtx::create(gpu);
        d_v = Mtx::create(gpu);
        d_rr = Mtx::create(gpu);
        d_prev_rho = Mtx::create(gpu);
        d_rho = Mtx::create(gpu);
        d_alpha = Mtx::create(gpu);
        d_beta = Mtx::create(gpu);
        d_gamma = Mtx::create(gpu);
        d_omega = Mtx::create(gpu);
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

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::GpuExecutor> gpu;

    std::ranlux48 rand_engine;

    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> d_mtx;
    std::unique_ptr<gko::solver::BicgstabFactory<>> gpu_bicgstab_factory;
    std::unique_ptr<gko::solver::BicgstabFactory<>> ref_bicgstab_factory;

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
};


TEST_F(Bicgstab, GpuBicgstabInitializeIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicgstab::initialize(
        ref, b.get(), r.get(), rr.get(), y.get(), s.get(), t.get(), z.get(),
        v.get(), p.get(), prev_rho.get(), rho.get(), alpha.get(), beta.get(),
        gamma.get(), omega.get());
    gko::kernels::gpu::bicgstab::initialize(
        gpu, d_b.get(), d_r.get(), d_rr.get(), d_y.get(), d_s.get(), d_t.get(),
        d_z.get(), d_v.get(), d_p.get(), d_prev_rho.get(), d_rho.get(),
        d_alpha.get(), d_beta.get(), d_gamma.get(), d_omega.get());

    EXPECT_MTX_NEAR(d_r, r, 1e-14);
    EXPECT_MTX_NEAR(d_z, z, 1e-14);
    EXPECT_MTX_NEAR(d_p, p, 1e-14);
    EXPECT_MTX_NEAR(d_y, y, 1e-14);
    EXPECT_MTX_NEAR(d_t, t, 1e-14);
    EXPECT_MTX_NEAR(d_s, s, 1e-14);
    EXPECT_MTX_NEAR(d_rr, rr, 1e-14);
    EXPECT_MTX_NEAR(d_v, v, 1e-14);
    EXPECT_MTX_NEAR(d_prev_rho, prev_rho, 1e-14);
    EXPECT_MTX_NEAR(d_rho, rho, 1e-14);
    EXPECT_MTX_NEAR(d_alpha, alpha, 1e-14);
    EXPECT_MTX_NEAR(d_beta, beta, 1e-14);
    EXPECT_MTX_NEAR(d_gamma, gamma, 1e-14);
    EXPECT_MTX_NEAR(d_omega, omega, 1e-14);
}


TEST_F(Bicgstab, GpuBicgstabStep1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicgstab::step_1(ref, r.get(), p.get(), v.get(),
                                              rho.get(), prev_rho.get(),
                                              alpha.get(), omega.get());
    gko::kernels::gpu::bicgstab::step_1(gpu, d_r.get(), d_p.get(), d_v.get(),
                                        d_rho.get(), d_prev_rho.get(),
                                        d_alpha.get(), d_omega.get());

    ASSERT_MTX_NEAR(d_p, p, 1e-14);
}


TEST_F(Bicgstab, GpuBicgstabStep2IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicgstab::step_2(
        ref, r.get(), s.get(), v.get(), rho.get(), alpha.get(), beta.get());
    gko::kernels::gpu::bicgstab::step_2(gpu, d_r.get(), d_s.get(), d_v.get(),
                                        d_rho.get(), d_alpha.get(),
                                        d_beta.get());

    ASSERT_MTX_NEAR(d_alpha, alpha, 1e-14);
    ASSERT_MTX_NEAR(d_s, s, 1e-14);
}


TEST_F(Bicgstab, GpuBicgstabStep3IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::bicgstab::step_3(
        ref, x.get(), r.get(), s.get(), t.get(), y.get(), z.get(), alpha.get(),
        beta.get(), gamma.get(), omega.get());
    gko::kernels::gpu::bicgstab::step_3(
        gpu, d_x.get(), d_r.get(), d_s.get(), d_t.get(), d_y.get(), d_z.get(),
        d_alpha.get(), d_beta.get(), d_gamma.get(), d_omega.get());

    ASSERT_MTX_NEAR(d_omega, omega, 1e-14);
    ASSERT_MTX_NEAR(d_x, x, 1e-14);
    ASSERT_MTX_NEAR(d_r, r, 1e-14);
}


TEST_F(Bicgstab, GpuBicgstabApplyOneRHSIsEquivalentToRef)
{
    int m = 123;
    int n = 1;
    auto ref_solver = ref_bicgstab_factory->generate(mtx);
    auto gpu_solver = gpu_bicgstab_factory->generate(d_mtx);
    auto b = gen_mtx(m, n);
    auto x = gen_mtx(m, n);
    auto d_b = Mtx::create(gpu);
    auto d_x = Mtx::create(gpu);
    d_b->copy_from(b.get());
    d_x->copy_from(x.get());

    ref_solver->apply(b.get(), x.get());
    gpu_solver->apply(d_b.get(), d_x.get());

    ASSERT_MTX_NEAR(d_b, b, 1e-13);
    ASSERT_MTX_NEAR(d_x, x, 1e-13);
}


TEST_F(Bicgstab, GpuBicgstabApplyMultipleRHSIsEquivalentToRef)
{
    int m = 123;
    int n = 16;
    auto gpu_solver = gpu_bicgstab_factory->generate(d_mtx);
    auto ref_solver = ref_bicgstab_factory->generate(mtx);
    auto b = gen_mtx(m, n);
    auto x = gen_mtx(m, n);
    auto d_b = Mtx::create(gpu);
    auto d_x = Mtx::create(gpu);
    d_b->copy_from(b.get());
    d_x->copy_from(x.get());

    ref_solver->apply(b.get(), x.get());
    gpu_solver->apply(d_b.get(), d_x.get());

    ASSERT_MTX_NEAR(d_b, b, 1e-13);
    ASSERT_MTX_NEAR(d_x, x, 1e-13);
}


}  // namespace
