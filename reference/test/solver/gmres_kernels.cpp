// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/gmres.hpp>


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


#include "core/solver/common_gmres_kernels.hpp"
#include "core/solver/gmres_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename T>
class Gmres : public ::testing::Test {
protected:
    using value_type = T;
    using rc_value_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    using rc_Mtx = gko::matrix::Dense<rc_value_type>;
    using Solver = gko::solver::Gmres<value_type>;
    Gmres()
        : exec(gko::ReferenceExecutor::create()),
          stopped{},
          non_stopped{},
          mtx(gko::initialize<Mtx>(
              {{1.0, 2.0, 3.0}, {3.0, 2.0, -1.0}, {0.0, -1.0, 2}}, exec)),
          gmres_factory(
              Solver::build()
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
          gmres_factory_big(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(100u),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value))
                  .on(exec)),
          gmres_factory_big2(
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
        constexpr gko::size_type small_restart{2};
        small_b = gko::initialize<Mtx>(
            {I<T>{1., 2.}, I<T>{3., 4.}, I<T>{5., 6.}}, exec);
        small_x = Mtx::create(exec, small_size);
        small_residual = Mtx::create(exec, small_size);
        small_residual_norm =
            rc_Mtx::create(exec, gko::dim<2>{1, small_size[1]});
        small_residual_norm_collection =
            Mtx::create(exec, gko::dim<2>{small_restart + 1, small_size[1]});
        small_krylov_bases = Mtx::create(
            exec,
            gko::dim<2>{small_size[0] * (small_restart + 1), small_size[1]});

        small_givens_sin =
            Mtx::create(exec, gko::dim<2>{small_restart, small_size[1]});
        small_givens_cos =
            Mtx::create(exec, gko::dim<2>{small_restart, small_size[1]});
        small_y = Mtx::create(exec, gko::dim<2>{small_restart, small_size[1]});
        small_hessenberg = Mtx::create(
            exec,
            gko::dim<2>{small_restart + 1, small_restart * small_size[1]});
        small_hessenberg->fill(gko::zero<value_type>());

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
    std::unique_ptr<rc_Mtx> small_residual_norm;
    std::unique_ptr<Mtx> small_residual_norm_collection;
    std::unique_ptr<Mtx> small_krylov_bases;
    std::unique_ptr<Mtx> small_givens_sin;
    std::unique_ptr<Mtx> small_givens_cos;
    std::unique_ptr<Mtx> small_y;
    std::unique_ptr<Mtx> small_hessenberg;
    gko::array<gko::size_type> small_final_iter_nums;
    gko::array<gko::stopping_status> small_stop;

    gko::stopping_status stopped;
    gko::stopping_status non_stopped;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> mtx_medium;
    std::shared_ptr<Mtx> mtx_big;
    std::unique_ptr<typename Solver::Factory> gmres_factory;
    std::unique_ptr<typename Solver::Factory> gmres_factory_big;
    std::unique_ptr<typename Solver::Factory> gmres_factory_big2;
};

TYPED_TEST_SUITE(Gmres, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Gmres, KernelInitialize)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    const T nan = std::numeric_limits<gko::remove_complex<T>>::quiet_NaN();
    this->small_residual->fill(nan);
    this->small_givens_sin->fill(nan);
    this->small_givens_cos->fill(nan);
    std::fill_n(this->small_stop.get_data(), this->small_stop.get_size(),
                this->stopped);
    auto expected_sin_cos =
        Mtx::create(this->exec, this->small_givens_sin->get_size());
    expected_sin_cos->fill(gko::zero<T>());

    gko::kernels::reference::common_gmres::initialize(
        this->exec, this->small_b.get(), this->small_residual.get(),
        this->small_givens_sin.get(), this->small_givens_cos.get(),
        this->small_stop.get_data());

    GKO_ASSERT_MTX_NEAR(this->small_residual, this->small_b, 0);
    GKO_ASSERT_MTX_NEAR(this->small_givens_sin, expected_sin_cos, 0);
    GKO_ASSERT_MTX_NEAR(this->small_givens_cos, expected_sin_cos, 0);
    for (int i = 0; i < this->small_stop.get_size(); ++i) {
        ASSERT_EQ(this->small_stop.get_data()[i], this->non_stopped);
    }
}


TYPED_TEST(Gmres, KernelRestart)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    const value_type nan =
        std::numeric_limits<gko::remove_complex<value_type>>::quiet_NaN();
    this->small_residual->copy_from(this->small_b);
    this->small_residual->compute_norm2(this->small_residual_norm);
    this->small_residual_norm_collection->fill(nan);
    this->small_krylov_bases->fill(9999);
    std::fill_n(this->small_final_iter_nums.get_data(),
                this->small_final_iter_nums.get_size(), 999);
    auto expected_krylov = gko::clone(this->exec, this->small_krylov_bases);
    const auto small_size = this->small_residual->get_size();
    for (int i = 0; i < small_size[0]; ++i) {
        for (int j = 0; j < small_size[1]; ++j) {
            expected_krylov
                ->get_values()[(0 * small_size[0] + i) * small_size[1] + j] =
                this->small_residual
                    ->get_const_values()[i * small_size[1] + j] /
                this->small_residual_norm->get_const_values()[j];
        }
    }

    gko::kernels::reference::gmres::restart(
        this->exec, this->small_residual.get(), this->small_residual_norm.get(),
        this->small_residual_norm_collection.get(),
        this->small_krylov_bases.get(), this->small_final_iter_nums.get_data());

    ASSERT_EQ(this->small_final_iter_nums.get_size(),
              this->small_residual_norm_collection->get_size()[1]);
    for (int i = 0; i < this->small_final_iter_nums.get_size(); ++i) {
        ASSERT_EQ(this->small_final_iter_nums.get_const_data()[i], 0);
        ASSERT_EQ(this->small_residual_norm_collection->get_const_values()[i],
                  this->small_residual_norm->get_const_values()[i]);
    }
    GKO_ASSERT_MTX_NEAR(this->small_krylov_bases, expected_krylov,
                        r<value_type>::value);
}


TYPED_TEST(Gmres, KernelHessenbergQrIter0)
{
    using T = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using std::sqrt;
    const auto nan = std::numeric_limits<gko::remove_complex<T>>::quiet_NaN();
    const gko::size_type iteration{0};
    this->small_givens_cos =
        gko::initialize<Mtx>({I<T>{-0.5, 1.}, I<T>{70., -71}}, this->exec);
    this->small_givens_sin =
        gko::initialize<Mtx>({I<T>{1., 0.}, I<T>{-72., 73.}}, this->exec);
    this->small_residual_norm->fill(nan);
    this->small_residual_norm_collection = gko::initialize<Mtx>(
        {I<T>{1.25, 1.5}, I<T>{nan, nan}, I<T>{95., 94.}}, this->exec);
    this->small_hessenberg = gko::initialize<Mtx>(
        {I<T>{0.5, -0.75}, I<T>{-0.5, 1}, I<T>{97., 96.}}, this->exec);
    this->small_final_iter_nums.get_data()[0] = 0;
    this->small_final_iter_nums.get_data()[1] = 0;

    gko::kernels::reference::common_gmres::hessenberg_qr(
        this->exec, this->small_givens_sin.get(), this->small_givens_cos.get(),
        this->small_residual_norm.get(),
        this->small_residual_norm_collection.get(),
        this->small_hessenberg.get(), iteration,
        this->small_final_iter_nums.get_data(),
        this->small_stop.get_const_data());

    ASSERT_EQ(this->small_final_iter_nums.get_data()[0], 1);
    ASSERT_EQ(this->small_final_iter_nums.get_data()[1], 1);
    GKO_EXPECT_MTX_NEAR(this->small_givens_cos,
                        l({{0.5 * sqrt(2.), -0.6}, {70., -71.}}), r<T>::value);
    GKO_EXPECT_MTX_NEAR(this->small_givens_sin,
                        l({{-0.5 * sqrt(2.), 0.8}, {-72., 73.}}), r<T>::value);
    GKO_EXPECT_MTX_NEAR(this->small_hessenberg,
                        l({{0.5 * sqrt(2.), 1.25}, {0., 0.}, {97., 96.}}),
                        r<T>::value);
    GKO_EXPECT_MTX_NEAR(
        this->small_residual_norm_collection,
        l({{0.625 * sqrt(2.), -0.9}, {0.625 * sqrt(2.), -1.2}, {95., 94.}}),
        r<T>::value);
    GKO_EXPECT_MTX_NEAR(this->small_residual_norm, l({{0.625 * sqrt(2.), 1.2}}),
                        r<T>::value);
}


TYPED_TEST(Gmres, KernelHessenbergQrIter1)
{
    using T = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using std::sqrt;
    const auto nan = std::numeric_limits<gko::remove_complex<T>>::quiet_NaN();
    const gko::size_type iteration{1};
    this->small_givens_cos =
        gko::initialize<Mtx>({I<T>{1., 0.5}, I<T>{-0.5, 1.}}, this->exec);
    this->small_givens_sin =
        gko::initialize<Mtx>({I<T>{0.5, 0.25}, I<T>{1., 0.}}, this->exec);
    this->small_residual_norm->fill(nan);
    this->small_residual_norm_collection = gko::initialize<Mtx>(
        {I<T>{95., 94.}, I<T>{1.25, 1.5}, I<T>{nan, nan}}, this->exec);
    this->small_hessenberg = gko::initialize<Mtx>(
        {I<T>{-0.5, 4}, I<T>{0.25, 0.5}, I<T>{-0.5, 1}}, this->exec);
    this->small_final_iter_nums.get_data()[0] = 1;
    this->small_final_iter_nums.get_data()[1] = 1;

    gko::kernels::reference::common_gmres::hessenberg_qr(
        this->exec, this->small_givens_sin.get(), this->small_givens_cos.get(),
        this->small_residual_norm.get(),
        this->small_residual_norm_collection.get(),
        this->small_hessenberg.get(), iteration,
        this->small_final_iter_nums.get_data(),
        this->small_stop.get_const_data());

    ASSERT_EQ(this->small_final_iter_nums.get_data()[0], 2);
    ASSERT_EQ(this->small_final_iter_nums.get_data()[1], 2);
    GKO_EXPECT_MTX_NEAR(this->small_givens_cos,
                        l({{1., 0.5}, {0.5 * sqrt(2.), -0.6}}), r<T>::value);
    GKO_EXPECT_MTX_NEAR(this->small_givens_sin,
                        l({{0.5, 0.25}, {-0.5 * sqrt(2.), 0.8}}), r<T>::value);
    GKO_EXPECT_MTX_NEAR(this->small_hessenberg,
                        l({{-0.375, 2.125}, {0.5 * sqrt(2.), 1.25}, {0., 0.}}),
                        r<T>::value);
    GKO_EXPECT_MTX_NEAR(
        this->small_residual_norm_collection,
        l({{95., 94.}, {0.625 * sqrt(2.), -0.9}, {0.625 * sqrt(2.), -1.2}}),
        r<T>::value);
    GKO_EXPECT_MTX_NEAR(this->small_residual_norm, l({{0.625 * sqrt(2.), 1.2}}),
                        r<T>::value);
}


TYPED_TEST(Gmres, KernelSolveKrylov)
{
    using T = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    const T nan = std::numeric_limits<gko::remove_complex<T>>::quiet_NaN();
    const auto restart = this->small_givens_sin->get_size()[0];
    this->small_y->fill(nan);
    this->small_final_iter_nums.get_data()[0] = restart;
    this->small_final_iter_nums.get_data()[1] = restart;
    this->small_hessenberg = gko::initialize<Mtx>(
        // clang-format off
        {{-1, 3, 2, -4},
         {0, 0, 1, 5},
         {nan, nan, nan, nan}},
        // clang-format on
        this->exec);
    this->small_residual_norm_collection =
        gko::initialize<Mtx>({I<T>{12, 3}, I<T>{-3, 15}}, this->exec);

    gko::kernels::reference::common_gmres::solve_krylov(
        this->exec, this->small_residual_norm_collection.get(),
        this->small_hessenberg.get(), this->small_y.get(),
        this->small_final_iter_nums.get_const_data(),
        this->small_stop.get_const_data());

    GKO_ASSERT_MTX_NEAR(this->small_y, l({{-18., 5.}, {-3., 3.}}), r<T>::value);
}


TYPED_TEST(Gmres, KernelMultiAxpy)
{
    using T = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    const T nan = std::numeric_limits<gko::remove_complex<T>>::quiet_NaN();
    const auto restart = this->small_givens_sin->get_size()[0];
    this->small_x->fill(nan);
    this->small_y =
        gko::initialize<Mtx>({I<T>{1., 2.}, I<T>{3., -1.}}, this->exec);
    this->small_final_iter_nums.get_data()[0] = restart;
    this->small_final_iter_nums.get_data()[1] = restart;
    this->small_krylov_bases = gko::initialize<Mtx>(  // restart+1 x rows x #rhs
        {
            I<T>{1, 10},     // 0, 0, x
            I<T>{2, 11},     // 0, 1, x
            I<T>{3, 12},     // 0, 2, x
            I<T>{4, 13},     // 1, 0, x
            I<T>{5, 14},     // 1, 1, x
            I<T>{6, 15},     // 1, 2, x
            I<T>{nan, nan},  // 2, 0, x
            I<T>{nan, nan},  // 2, 1, x
            I<T>{nan, nan},  // 2, 2, x
        },
        this->exec);
    this->small_stop.get_data()[0].stop(7, false);
    gko::stopping_status expected_stop{};
    expected_stop.stop(7, true);

    gko::kernels::reference::gmres::multi_axpy(
        this->exec, this->small_krylov_bases.get(), this->small_y.get(),
        this->small_x.get(), this->small_final_iter_nums.get_const_data(),
        this->small_stop.get_data());

    ASSERT_EQ(this->small_stop.get_const_data()[0], expected_stop);
    ASSERT_EQ(this->small_stop.get_const_data()[1], this->non_stopped);
    GKO_ASSERT_MTX_NEAR(this->small_x, l({{13., 7.}, {17., 8.}, {21., 9.}}),
                        r<T>::value);
}


TYPED_TEST(Gmres, SolvesStencilSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gmres_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({13.0, 7.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Gmres, SolvesStencilSystemMixed)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->gmres_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({13.0, 7.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Gmres, SolvesStencilSystemComplex)
{
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->gmres_factory->generate(this->mtx);
    auto b =
        gko::initialize<Mtx>({value_type{13.0, -26.0}, value_type{7.0, -14.0},
                              value_type{1.0, -2.0}},
                             this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.0, 0.0}, value_type{0.0, 0.0}, value_type{0.0, 0.0}},
        this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.0, -2.0}, value_type{3.0, -6.0},
                           value_type{2.0, -4.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Gmres, SolvesStencilSystemMixedComplex)
{
    using value_type =
        gko::to_complex<gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->gmres_factory->generate(this->mtx);
    auto b =
        gko::initialize<Mtx>({value_type{13.0, -26.0}, value_type{7.0, -14.0},
                              value_type{1.0, -2.0}},
                             this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.0, 0.0}, value_type{0.0, 0.0}, value_type{0.0, 0.0}},
        this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.0, -2.0}, value_type{3.0, -6.0},
                           value_type{2.0, -4.0}}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Gmres, SolvesMultipleStencilSystems)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto solver = this->gmres_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>(
        {I<T>{13.0, 6.0}, I<T>{7.0, 4.0}, I<T>{1.0, 1.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({{1.0, 1.0}, {3.0, 1.0}, {2.0, 1.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Gmres, SolvesStencilSystemUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gmres_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({13.0, 7.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.5, 5.0, 2.0}), r<value_type>::value * 1e1);
}


TYPED_TEST(Gmres, SolvesStencilSystemUsingAdvancedApplyMixed)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto solver = this->gmres_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>({13.0, 7.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.5, 1.0, 2.0}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.5, 5.0, 2.0}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Gmres, SolvesStencilSystemUsingAdvancedApplyComplex)
{
    using Scalar = typename TestFixture::Mtx;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->gmres_factory->generate(this->mtx);
    auto alpha = gko::initialize<Scalar>({2.0}, this->exec);
    auto beta = gko::initialize<Scalar>({-1.0}, this->exec);
    auto b =
        gko::initialize<Mtx>({value_type{13.0, -26.0}, value_type{7.0, -14.0},
                              value_type{1.0, -2.0}},
                             this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.5, -1.0}, value_type{1.0, -2.0}, value_type{2.0, -4.0}},
        this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.5, -3.0}, value_type{5.0, -10.0},
                           value_type{2.0, -4.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Gmres, SolvesStencilSystemUsingAdvancedApplyMixedComplex)
{
    using Scalar = gko::matrix::Dense<
        gko::next_precision<typename TestFixture::value_type>>;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto solver = this->gmres_factory->generate(this->mtx);
    auto alpha = gko::initialize<Scalar>({2.0}, this->exec);
    auto beta = gko::initialize<Scalar>({-1.0}, this->exec);
    auto b =
        gko::initialize<Mtx>({value_type{13.0, -26.0}, value_type{7.0, -14.0},
                              value_type{1.0, -2.0}},
                             this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.5, -1.0}, value_type{1.0, -2.0}, value_type{2.0, -4.0}},
        this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.5, -3.0}, value_type{5.0, -10.0},
                           value_type{2.0, -4.0}}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Gmres, SolvesMultipleStencilSystemsUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto solver = this->gmres_factory->generate(this->mtx);
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto b = gko::initialize<Mtx>(
        {I<T>{13.0, 6.0}, I<T>{7.0, 4.0}, I<T>{1.0, 1.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.5, 1.0}, I<T>{1.0, 2.0}, I<T>{2.0, 3.0}}, this->exec);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({{1.5, 1.0}, {5.0, 0.0}, {2.0, -1.0}}),
                        r<value_type>::value * 1e1);
}


TYPED_TEST(Gmres, SolvesBigDenseSystem1)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gmres_factory_big->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {72748.36, 297469.88, 347229.24, 36290.66, 82958.82, -80192.15},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({52.7, 85.4, 134.2, -250.0, -16.8, 35.3}),
                        r<value_type>::value * 1e3);
}


TYPED_TEST(Gmres, SolvesBigDenseSystem2)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gmres_factory_big->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {175352.10, 313410.50, 131114.10, -134116.30, 179529.30, -43564.90},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({33.0, -56.0, 81.0, -30.0, 21.0, 40.0}),
                        r<value_type>::value * 1e3);
}


TYPED_TEST(Gmres, SolveWithImplicitResNormCritIsDisabled)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gmres_factory_big2->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {175352.10, 313410.50, 131114.10, -134116.30, 179529.30, -43564.90},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    ASSERT_THROW(solver->apply(b, x), gko::NotSupported);
}


TYPED_TEST(Gmres, SolvesMultipleDenseSystemForDivergenceCheck)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gmres_factory_big->generate(this->mtx_big);
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

    solver->apply(b1, x1);
    solver->apply(b2, x2);
    solver->apply(bc, xc);
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

    this->mtx_big->apply(alpha, x1, beta, residual1);
    this->mtx_big->apply(alpha, x2, beta, residual2);
    this->mtx_big->apply(alpha, xc, beta, residualC);

    auto normS1 = inf_norm(residual1);
    auto normS2 = inf_norm(residual2);
    auto normC1 = inf_norm(residualC, 0);
    auto normC2 = inf_norm(residualC, 1);
    auto normB1 = inf_norm(b1);
    auto normB2 = inf_norm(b2);

    // make sure that all combined solutions are as good or better than the
    // single solutions
    ASSERT_LE(normC1 / normB1, normS1 / normB1 + r<value_type>::value);
    ASSERT_LE(normC2 / normB2, normS2 / normB2 + r<value_type>::value);

    // Not sure if this is necessary, the assertions above should cover what
    // is needed.
    GKO_ASSERT_MTX_NEAR(xc, mergedRes, r<value_type>::value);
}


TYPED_TEST(Gmres, SolvesBigDenseSystem1WithRestart)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto gmres_factory_restart =
        Solver::build()
            .with_krylov_dim(4u)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(200u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(r<value_type>::value))
            .on(this->exec);
    auto solver = gmres_factory_restart->generate(this->mtx_medium);
    auto b = gko::initialize<Mtx>(
        {-13945.16, 11205.66, 16132.96, 24342.18, -10910.98}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-140.20, -142.20, 48.80, -17.70, -19.60}),
                        half_tol * 1e2);
}


TYPED_TEST(Gmres, SolvesWithPreconditioner)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto gmres_factory_preconditioner =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(100u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(r<value_type>::value))
            .with_preconditioner(
                gko::preconditioner::Jacobi<value_type>::build()
                    .with_max_block_size(3u))
            .on(this->exec);
    auto solver = gmres_factory_preconditioner->generate(this->mtx_big);
    auto b = gko::initialize<Mtx>(
        {175352.10, 313410.50, 131114.10, -134116.30, 179529.30, -43564.90},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({33.0, -56.0, 81.0, -30.0, 21.0, 40.0}),
                        r<value_type>::value * 1e3);
}


TYPED_TEST(Gmres, SolvesTransposedBigDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->gmres_factory_big->generate(this->mtx_big->transpose());
    auto b = gko::initialize<Mtx>(
        {72748.36, 297469.88, 347229.24, 36290.66, 82958.82, -80192.15},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({52.7, 85.4, 134.2, -250.0, -16.8, 35.3}),
                        r<value_type>::value * 1e3);
}


TYPED_TEST(Gmres, SolvesConjTransposedBigDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver =
        this->gmres_factory_big->generate(this->mtx_big->conj_transpose());
    auto b = gko::initialize<Mtx>(
        {72748.36, 297469.88, 347229.24, 36290.66, 82958.82, -80192.15},
        this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);

    solver->conj_transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({52.7, 85.4, 134.2, -250.0, -16.8, 35.3}),
                        r<value_type>::value * 1e3);
}


}  // namespace
