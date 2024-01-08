// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/gcr.hpp>


#include <algorithm>
#include <limits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/solver/gcr_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename T>
class Gcr : public ::testing::Test {
protected:
    using value_type = T;
    using rc_value_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    using rc_Mtx = gko::matrix::Dense<rc_value_type>;
    using Solver = gko::solver::Gcr<value_type>;
    Gcr()
        : exec(gko::ReferenceExecutor::create()),
          stopped{},
          non_stopped{},
          mtx(gko::initialize<Mtx>(
              {{1.0, 2.0, 3.0}, {3.0, 2.0, -1.0}, {0.0, -1.0, 2}}, exec)),
          gcr_factory(Solver::build()
                          .with_criteria(
                              gko::stop::Iteration::build().with_max_iters(4u),
                              gko::stop::Time::build().with_time_limit(
                                  std::chrono::seconds(6)),
                              gko::stop::ResidualNorm<value_type>::build()
                                  .with_reduction_factor(r<value_type>::value))
                          .with_krylov_dim(3u)
                          .on(exec)),
          mtx_big(gko::initialize<Mtx>(
              {{2295.7, -764.8, 1166.5, 428.9, 291.7, -774.5},
               {2752.6, -1127.7, 1212.8, -299.1, 987.7, 786.8},
               {138.3, 78.2, 485.5, -899.9, 392.9, 1408.9},
               {-1907.1, 2106.6, 1026.0, 634.7, 194.6, -534.1},
               {-365.0, -715.8, 870.7, 67.5, 279.8, 1927.8},
               {-848.1, -280.5, -381.8, -187.1, 51.2, -176.2}},
              exec)),
          gcr_factory_big(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(100u),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value))
                  .on(exec)),
          gcr_factory_big2(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(100u),
                      gko::stop::ImplicitResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value))
                  .on(exec)),
          mtx_medium(
              gko::initialize<Mtx>({{-86.40, 153.30, -108.90, 8.60, -61.60},
                                    {7.70, -77.00, 3.30, -149.20, 74.80},
                                    {-121.40, 37.10, 55.30, -74.20, -19.20},
                                    {-111.40, -22.60, 110.10, -106.20, 88.90},
                                    {-0.70, 111.70, 154.40, 235.00, -76.50}},
                                   exec))
    {
        auto small_size = gko::dim<2>{3, 2};
        small_b = gko::initialize<Mtx>(
            {I<T>{1., 2.}, I<T>{3., 4.}, I<T>{5., 6.}}, exec);
        small_x = Mtx::create(exec, small_size);
        small_residual = Mtx::create(exec, small_size);
        small_A_residual = Mtx::create(exec, small_size);
        small_krylov_bases_p = Mtx::create(exec, small_size);
        small_mapped_krylov_bases_Ap = Mtx::create(exec, small_size);
        small_Ap_norm = rc_Mtx::create(exec, gko::dim<2>{1, small_size[1]});
        small_tmp_rAp = Mtx::create(exec, gko::dim<2>{1, small_size[1]});

        stopped.converge(1, true);
        non_stopped.reset();
        small_stop = gko::array<gko::stopping_status>(exec, small_size[1]);
        std::fill_n(small_stop.get_data(), small_stop.get_size(), non_stopped);
        small_final_iter_nums = gko::array<gko::size_type>(exec, small_size[1]);
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<Mtx> small_x;
    std::unique_ptr<Mtx> small_b;
    std::unique_ptr<Mtx> small_residual;
    std::unique_ptr<Mtx> small_A_residual;
    std::unique_ptr<Mtx> small_krylov_bases_p;
    std::unique_ptr<Mtx> small_mapped_krylov_bases_Ap;
    std::unique_ptr<rc_Mtx> small_Ap_norm;
    std::unique_ptr<Mtx> small_tmp_rAp;
    gko::array<gko::size_type> small_final_iter_nums;
    gko::array<gko::stopping_status> small_stop;

    gko::stopping_status stopped;
    gko::stopping_status non_stopped;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> mtx_medium;
    std::shared_ptr<Mtx> mtx_big;
    std::unique_ptr<typename Solver::Factory> gcr_factory;
    std::unique_ptr<typename Solver::Factory> gcr_factory_big;
    std::unique_ptr<typename Solver::Factory> gcr_factory_big2;
};

TYPED_TEST_SUITE(Gcr, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Gcr, KernelInitialize)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    const T nan = std::numeric_limits<gko::remove_complex<T>>::quiet_NaN();
    this->small_residual->fill(nan);
    std::fill_n(this->small_stop.get_data(), this->small_stop.get_size(),
                this->stopped);

    gko::kernels::reference::gcr::initialize(this->exec, this->small_b.get(),
                                             this->small_residual.get(),
                                             this->small_stop.get_data());

    GKO_ASSERT_MTX_NEAR(this->small_residual, this->small_b, 0);
    for (int i = 0; i < this->small_stop.get_size(); ++i) {
        ASSERT_EQ(this->small_stop.get_data()[i], this->non_stopped);
    }
}


TYPED_TEST(Gcr, KernelRestart)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    const value_type nan =
        std::numeric_limits<gko::remove_complex<value_type>>::quiet_NaN();
    this->small_residual->copy_from(this->small_b.get());
    this->mtx->apply(this->small_residual.get(), this->small_A_residual.get());
    this->small_krylov_bases_p->fill(nan);
    this->small_mapped_krylov_bases_Ap->fill(nan);
    std::fill_n(this->small_final_iter_nums.get_data(),
                this->small_final_iter_nums.get_size(), 999);
    auto expected_p_bases = gko::clone(this->exec, this->small_krylov_bases_p);
    auto expected_Ap_bases =
        gko::clone(this->exec, this->small_mapped_krylov_bases_Ap);
    const auto small_size = this->small_residual->get_size();
    for (int i = 0; i < small_size[0]; ++i) {
        for (int j = 0; j < small_size[1]; ++j) {
            expected_p_bases->get_values()[i * small_size[1] + j] =
                this->small_residual->get_const_values()[i * small_size[1] + j];
            expected_Ap_bases->get_values()[i * small_size[1] + j] =
                this->small_A_residual
                    ->get_const_values()[i * small_size[1] + j];
        }
    }

    gko::kernels::reference::gcr::restart(
        this->exec, this->small_residual.get(), this->small_A_residual.get(),
        this->small_krylov_bases_p.get(),
        this->small_mapped_krylov_bases_Ap.get(),
        this->small_final_iter_nums.get_data());

    ASSERT_EQ(this->small_final_iter_nums.get_size(),
              this->small_residual->get_size()[1]);
    for (int i = 0; i < this->small_final_iter_nums.get_size(); ++i) {
        ASSERT_EQ(this->small_final_iter_nums.get_const_data()[i], 0);
    }
    GKO_ASSERT_MTX_NEAR(this->small_krylov_bases_p, this->small_residual,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(this->small_mapped_krylov_bases_Ap,
                        this->small_A_residual, r<value_type>::value);
}


TYPED_TEST(Gcr, KernelStep1)
{
    using T = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    const auto nan = std::numeric_limits<gko::remove_complex<T>>::quiet_NaN();
    this->small_x->fill(nan);
    this->small_residual->fill(nan);
    this->small_krylov_bases_p = gko::initialize<Mtx>(
        {I<T>{0.5, -0.75}, I<T>{1.25, 1.5}, I<T>{-0.5, 1}}, this->exec);
    this->mtx->apply(this->small_krylov_bases_p.get(),
                     this->small_mapped_krylov_bases_Ap.get());
    this->small_mapped_krylov_bases_Ap->compute_norm2(
        this->small_Ap_norm.get());
    this->small_tmp_rAp = gko::initialize<Mtx>({13.0, 7.0, 1.0}, this->exec);

    gko::kernels::reference::gcr::step_1(
        this->exec, this->small_x.get(), this->small_residual.get(),
        this->small_krylov_bases_p.get(),
        this->small_mapped_krylov_bases_Ap.get(), this->small_Ap_norm.get(),
        this->small_tmp_rAp.get(), this->small_stop.get_data());
}


TYPED_TEST(Gcr, SolvesStencilSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gcr_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({13.0, 7.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Gcr, SolvesStencilSystemMixed)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->gcr_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({13.0, 7.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Gcr, SolvesStencilSystemComplex)
{
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->gcr_factory->generate(this->mtx);
    auto b =
        gko::initialize<Mtx>({value_type{13.0, -26.0}, value_type{7.0, -14.0},
                              value_type{1.0, -2.0}},
                             this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.0, 0.0}, value_type{0.0, 0.0}, value_type{0.0, 0.0}},
        this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.0, -2.0}, value_type{3.0, -6.0},
                           value_type{2.0, -4.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Gcr, SolvesStencilSystemMixedComplex)
{
    using value_type =
        gko::to_complex<gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->gcr_factory->generate(this->mtx);
    auto b =
        gko::initialize<Mtx>({value_type{13.0, -26.0}, value_type{7.0, -14.0},
                              value_type{1.0, -2.0}},
                             this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.0, 0.0}, value_type{0.0, 0.0}, value_type{0.0, 0.0}},
        this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.0, -2.0}, value_type{3.0, -6.0},
                           value_type{2.0, -4.0}}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Gcr, SolvesMultipleStencilSystems)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto solver = this->gcr_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {I<T>{13.0, 6.0}, I<T>{7.0, 4.0}, I<T>{1.0, 1.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({{1.0, 1.0}, {3.0, 1.0}, {2.0, 1.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Gcr, SolvesStencilSystemUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gcr_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({13.0, 7.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.5, 5.0, 2.0}), r<value_type>::value * 1e2);
}


TYPED_TEST(Gcr, SolvesStencilSystemUsingAdvancedApplyMixed)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->gcr_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({13.0, 7.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.5, 5.0, 2.0}),
                        (r_mixed<value_type, TypeParam>()) * 1e1);
}


TYPED_TEST(Gcr, SolvesStencilSystemUsingAdvancedApplyComplex)
{
    using Scalar = typename TestFixture::Mtx;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->gcr_factory->generate(this->mtx);
    auto alpha = gko::initialize<Scalar>({2.0}, this->exec);
    auto beta = gko::initialize<Scalar>({-1.0}, this->exec);
    auto b =
        gko::initialize<Mtx>({value_type{13.0, -26.0}, value_type{7.0, -14.0},
                              value_type{1.0, -2.0}},
                             this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.5, -1.0}, value_type{1.0, -2.0}, value_type{2.0, -4.0}},
        this->exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.5, -3.0}, value_type{5.0, -10.0},
                           value_type{2.0, -4.0}}),
                        r<value_type>::value * 1e2);
}


TYPED_TEST(Gcr, SolvesStencilSystemUsingAdvancedApplyMixedComplex)
{
    using Scalar = gko::matrix::Dense<
        gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->gcr_factory->generate(this->mtx);
    auto alpha = gko::initialize<Scalar>({2.0}, this->exec);
    auto beta = gko::initialize<Scalar>({-1.0}, this->exec);
    auto b =
        gko::initialize<Mtx>({value_type{13.0, -26.0}, value_type{7.0, -14.0},
                              value_type{1.0, -2.0}},
                             this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.5, -1.0}, value_type{1.0, -2.0}, value_type{2.0, -4.0}},
        this->exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.5, -3.0}, value_type{5.0, -10.0},
                           value_type{2.0, -4.0}}),
                        (r_mixed<value_type, TypeParam>()) * 1e2);
}


TYPED_TEST(Gcr, SolvesMultipleStencilSystemsUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto solver = this->gcr_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>(
        {I<T>{13.0, 6.0}, I<T>{7.0, 4.0}, I<T>{1.0, 1.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.5, 1.0}, I<T>{1.0, 2.0}, I<T>{2.0, 3.0}}, this->exec);

    solver->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({{1.5, 1.0}, {5.0, 0.0}, {2.0, -1.0}}),
                        r<value_type>::value * 1e2);
}


TYPED_TEST(Gcr, SolvesBigDenseSystem1)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gcr_factory_big->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {72748.36, 297469.88, 347229.24, 36290.66, 82958.82, -80192.15},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({52.7, 85.4, 134.2, -250.0, -16.8, 35.3}),
                        r<value_type>::value * 1e3);
}


TYPED_TEST(Gcr, SolvesBigDenseSystem2)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gcr_factory_big->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {175352.10, 313410.50, 131114.10, -134116.30, 179529.30, -43564.90},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({33.0, -56.0, 81.0, -30.0, 21.0, 40.0}),
                        r<value_type>::value * 1e3);
}


TYPED_TEST(Gcr, SolveWithImplicitResNormCritIsDisabled)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gcr_factory_big2->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {175352.10, 313410.50, 131114.10, -134116.30, 179529.30, -43564.90},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    ASSERT_THROW(solver->apply(b.get(), x.get()), gko::NotSupported);
}


template <typename T>
gko::remove_complex<T> infNorm(gko::matrix::Dense<T>* mat, size_t col = 0)
{
    using std::abs;
    using no_cpx_t = gko::remove_complex<T>;
    no_cpx_t norm = 0.0;
    for (size_t i = 0; i < mat->get_size()[0]; ++i) {
        no_cpx_t absEntry = abs(mat->at(i, col));
        if (norm < absEntry) norm = absEntry;
    }
    return norm;
}


TYPED_TEST(Gcr, SolvesMultipleDenseSystemForDivergenceCheck)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gcr_factory_big->generate(this->mtx_big);
    auto b1 = gko::initialize<Mtx>(
        {1300083.0, 1018120.5, 906410.0, -42679.5, 846779.5, 1176858.5},
        this->exec);
    auto b2 = gko::initialize<Mtx>(
        {886630.5, -172578.0, 684522.0, -65310.5, 455487.5, 607436.0},
        this->exec);

    auto x1 = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);
    auto x2 = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    auto bc =
        Mtx::create(this->exec, gko::dim<2>{this->mtx_big->get_size()[0], 2});
    auto xc =
        Mtx::create(this->exec, gko::dim<2>{this->mtx_big->get_size()[1], 2});
    for (size_t i = 0; i < bc->get_size()[0]; ++i) {
        bc->at(i, 0) = b1->at(i);
        bc->at(i, 1) = b2->at(i);

        xc->at(i, 0) = x1->at(i);
        xc->at(i, 1) = x2->at(i);
    }

    solver->apply(b1.get(), x1.get());
    solver->apply(b2.get(), x2.get());
    solver->apply(bc.get(), xc.get());
    auto mergedRes = Mtx::create(this->exec, gko::dim<2>{b1->get_size()[0], 2});
    for (size_t i = 0; i < mergedRes->get_size()[0]; ++i) {
        mergedRes->at(i, 0) = x1->at(i);
        mergedRes->at(i, 1) = x2->at(i);
    }

    auto alpha = gko::initialize<Mtx>({1.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);

    auto residual1 = gko::clone(this->exec, b1);
    auto residual2 = gko::clone(this->exec, b2);
    auto residualC = gko::clone(this->exec, bc);

    this->mtx_big->apply(alpha.get(), x1.get(), beta.get(), residual1.get());
    this->mtx_big->apply(alpha.get(), x2.get(), beta.get(), residual2.get());
    this->mtx_big->apply(alpha.get(), xc.get(), beta.get(), residualC.get());

    auto normS1 = infNorm(residual1.get());
    auto normS2 = infNorm(residual2.get());
    auto normC1 = infNorm(residualC.get(), 0);
    auto normC2 = infNorm(residualC.get(), 1);
    auto normB1 = infNorm(b1.get());
    auto normB2 = infNorm(b2.get());

    // make sure that all combined solutions are as good or better than the
    // single solutions
    ASSERT_LE(normC1 / normB1, normS1 / normB1 + r<value_type>::value);
    ASSERT_LE(normC2 / normB2, normS2 / normB2 + r<value_type>::value);

    // Not sure if this is necessary, the assertions above should cover what
    // is needed.
    GKO_ASSERT_MTX_NEAR(xc, mergedRes, r<value_type>::value);
}


TYPED_TEST(Gcr, SolvesBigDenseSystem1WithRestart)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto gcr_factory_restart =
        Solver::build()
            .with_krylov_dim(4u)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(200u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(r<value_type>::value))
            .on(this->exec);
    auto solver = gcr_factory_restart->generate(this->mtx_medium);
    auto b = gko::initialize<Mtx>(
        {-13945.16, 11205.66, 16132.96, 24342.18, -10910.98}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({-140.20, -142.20, 48.80, -17.70, -19.60}),
                        half_tol * 1e2);
}


TYPED_TEST(Gcr, SolvesWithPreconditioner)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto gcr_factory_preconditioner =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(100u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(r<value_type>::value))
            .with_preconditioner(
                gko::preconditioner::Jacobi<value_type>::build()
                    .with_max_block_size(3u))
            .on(this->exec);
    auto solver = gcr_factory_preconditioner->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {175352.10, 313410.50, 131114.10, -134116.30, 179529.30, -43564.90},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({33.0, -56.0, 81.0, -30.0, 21.0, 40.0}),
                        r<value_type>::value * 1e3);
}


TYPED_TEST(Gcr, SolvesTransposedBigDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gcr_factory_big->generate(this->mtx_big->transpose());
    auto b = gko::initialize<Mtx>(
        {72748.36, 297469.88, 347229.24, 36290.66, 82958.82, -80192.15},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->transpose()->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({52.7, 85.4, 134.2, -250.0, -16.8, 35.3}),
                        r<value_type>::value * 1e3);
}


TYPED_TEST(Gcr, SolvesConjTransposedBigDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver =
        this->gcr_factory_big->generate(this->mtx_big->conj_transpose());
    auto b = gko::initialize<Mtx>(
        {72748.36, 297469.88, 347229.24, 36290.66, 82958.82, -80192.15},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->conj_transpose()->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({52.7, 85.4, 134.2, -250.0, -16.8, 35.3}),
                        r<value_type>::value * 1e3);
}


}  // namespace
