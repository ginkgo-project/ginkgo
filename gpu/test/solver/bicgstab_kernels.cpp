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
    void SetUp()
    {
        ASSERT_GT(gko::GpuExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        gpu = gko::GpuExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (gpu != nullptr) {
            ASSERT_NO_THROW(gpu->synchronize());
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::GpuExecutor> gpu;
};


TEST_F(Bicgstab, GpuBicgstabInitializeIsEquivalentToRef)
{
    std::ranlux48 rand_engine(30);


    auto gen_mtx = [&](int m, int n) {
        return gko::test::generate_random_matrix<Mtx>(
            ref, m, n, std::uniform_int_distribution<>(1, 1),
            std::normal_distribution<>(0.0, 1.0), rand_engine);
    };

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
    std::ranlux48 rand_engine(30);


    auto gen_mtx = [&](int m, int n) {
        return gko::test::generate_random_matrix<Mtx>(
            ref, m, n, std::uniform_int_distribution<>(1, 1),
            std::normal_distribution<>(0.0, 1.0), rand_engine);
    };

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

    //     printf("GPU: (padding %d) \n", p_result->get_padding());
    // for (int row=0; row<m; row++) {
    //     for (int col=0; col<n; col++) {
    //         printf("%.2f  ",   p_result->at(row, col));
    //     }
    //     printf("\n");
    // }

    ASSERT_MTX_NEAR(p_result, p, 1e-14);
    ASSERT_MTX_NEAR(r_result, r, 1e-14);
    ASSERT_MTX_NEAR(v_result, v, 1e-14);
}


TEST_F(Bicgstab, GpuBicgstabStep2IsEquivalentToRef)
{
    std::ranlux48 rand_engine(30);


    auto gen_mtx = [&](int m, int n) {
        return gko::test::generate_random_matrix<Mtx>(
            ref, m, n, std::uniform_int_distribution<>(1, 1),
            std::normal_distribution<>(0.0, 1.0), rand_engine);
    };

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
    std::ranlux48 rand_engine(30);


    auto gen_mtx = [&](int m, int n) {
        return gko::test::generate_random_matrix<Mtx>(
            ref, m, n, std::uniform_int_distribution<>(1, 1),
            std::normal_distribution<>(0.0, 1.0), rand_engine);
    };

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


}  // namespace
