// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/batch_multi_vector.hpp>


#include <complex>
#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/batch_multi_vector_kernels.hpp"
#include "core/base/batch_utilities.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_helpers.hpp"


template <typename T>
class MultiVector : public ::testing::Test {
protected:
    using value_type = T;
    using size_type = gko::size_type;
    using Mtx = gko::batch::MultiVector<value_type>;
    using DenseMtx = gko::matrix::Dense<value_type>;
    using ComplexMtx = gko::to_complex<Mtx>;
    MultiVector()
        : exec(gko::ReferenceExecutor::create()),
          mtx_0(gko::batch::initialize<Mtx>(
              {{I<T>({1.0, -1.0, 1.5}), I<T>({-2.0, 2.0, 3.0})},
               {{1.0, -2.0, -0.5}, {1.0, -2.5, 4.0}}},
              exec)),
          mtx_00(gko::initialize<DenseMtx>(
              {I<T>({1.0, -1.0, 1.5}), I<T>({-2.0, 2.0, 3.0})}, exec)),
          mtx_01(gko::initialize<DenseMtx>(
              {I<T>({1.0, -2.0, -0.5}), I<T>({1.0, -2.5, 4.0})}, exec)),
          mtx_1(gko::batch::initialize<Mtx>(
              {{{1.0, -1.0, 2.2}, {-2.0, 2.0, -0.5}},
               {{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}}},
              exec)),
          mtx_10(gko::initialize<DenseMtx>(
              {I<T>({1.0, -1.0, 2.2}), I<T>({-2.0, 2.0, -0.5})}, exec)),
          mtx_11(gko::initialize<DenseMtx>({{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}},
                                           exec)),
          mtx_2(gko::batch::initialize<Mtx>(
              {{{1.0, 1.5}, {6.0, 1.0}, {-0.25, 1.0}},
               {I<T>({2.0, -2.0}), I<T>({1.0, 3.0}), I<T>({4.0, 3.0})}},
              exec)),
          mtx_20(gko::initialize<DenseMtx>(
              {I<T>({1.0, 1.5}), I<T>({6.0, 1.0}), I<T>({-0.25, 1.0})}, exec)),
          mtx_21(gko::initialize<DenseMtx>(
              {I<T>({2.0, -2.0}), I<T>({1.0, 3.0}), I<T>({4.0, 3.0})}, exec)),
          mtx_3(gko::batch::initialize<Mtx>(
              {{I<T>({1.0, 1.5}), I<T>({6.0, 1.0})}, {{2.0, -2.0}, {1.0, 3.0}}},
              exec)),
          mtx_30(gko::initialize<DenseMtx>({I<T>({1.0, 1.5}), I<T>({6.0, 1.0})},
                                           exec)),
          mtx_31(gko::initialize<DenseMtx>(
              {I<T>({2.0, -2.0}), I<T>({1.0, 3.0})}, exec)),
          mtx_4(gko::batch::initialize<Mtx>(
              {{{1.0, 1.5, 3.0}, {6.0, 1.0, 5.0}, {6.0, 1.0, 5.5}},
               {{2.0, -2.0, 1.5}, {4.0, 3.0, 2.2}, {-1.25, 3.0, 0.5}}},
              exec)),
          mtx_5(gko::batch::initialize<Mtx>(
              {{{1.0, 1.5}, {6.0, 1.0}, {7.0, -4.5}},
               {I<T>({2.0, -2.0}), I<T>({1.0, 3.0}), I<T>({4.0, 3.0})}},
              exec)),
          mtx_6(gko::batch::initialize<Mtx>(
              {{{1.0, 0.0, 3.0}, {0.0, 3.0, 0.0}, {0.0, 1.0, 5.0}},
               {{2.0, 0.0, 5.0}, {0.0, 1.0, 0.0}, {0.0, -1.0, 8.0}}},
              exec))
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<Mtx> mtx_0;
    std::unique_ptr<DenseMtx> mtx_00;
    std::unique_ptr<DenseMtx> mtx_01;
    std::unique_ptr<Mtx> mtx_1;
    std::unique_ptr<DenseMtx> mtx_10;
    std::unique_ptr<DenseMtx> mtx_11;
    std::unique_ptr<Mtx> mtx_2;
    std::unique_ptr<DenseMtx> mtx_20;
    std::unique_ptr<DenseMtx> mtx_21;
    std::unique_ptr<Mtx> mtx_3;
    std::unique_ptr<DenseMtx> mtx_30;
    std::unique_ptr<DenseMtx> mtx_31;
    std::unique_ptr<Mtx> mtx_4;
    std::unique_ptr<Mtx> mtx_5;
    std::unique_ptr<Mtx> mtx_6;

    std::default_random_engine rand_engine;
};

TYPED_TEST_SUITE(MultiVector, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(MultiVector, ScalesData)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch::initialize<Mtx>(
        {{{2.0, -2.0, 1.5}}, {{3.0, -1.0, 0.25}}}, this->exec);
    auto ualpha = gko::batch::unbatch<gko::batch::MultiVector<T>>(alpha.get());

    this->mtx_0->scale(alpha.get());

    this->mtx_00->scale(ualpha[0].get());
    this->mtx_01->scale(ualpha[1].get());
    auto res =
        gko::batch::unbatch<gko::batch::MultiVector<T>>(this->mtx_0.get());
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->mtx_00.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->mtx_01.get(), 0.);
}


TYPED_TEST(MultiVector, ScalesDataWithScalar)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch::initialize<Mtx>({{2.0}, {-2.0}}, this->exec);
    auto ualpha = gko::batch::unbatch<gko::batch::MultiVector<T>>(alpha.get());

    this->mtx_1->scale(alpha.get());

    this->mtx_10->scale(ualpha[0].get());
    this->mtx_11->scale(ualpha[1].get());
    auto res =
        gko::batch::unbatch<gko::batch::MultiVector<T>>(this->mtx_1.get());
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->mtx_10.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->mtx_11.get(), 0.);
}


TYPED_TEST(MultiVector, ScalesDataWithMultipleScalars)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch::initialize<Mtx>(
        {{{2.0, -2.0, -1.5}}, {{2.0, -2.0, 3.0}}}, this->exec);
    auto ualpha = gko::batch::unbatch<gko::batch::MultiVector<T>>(alpha.get());

    this->mtx_1->scale(alpha.get());
    this->mtx_10->scale(ualpha[0].get());
    this->mtx_11->scale(ualpha[1].get());

    auto res =
        gko::batch::unbatch<gko::batch::MultiVector<T>>(this->mtx_1.get());
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->mtx_10.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->mtx_11.get(), 0.);
}


TYPED_TEST(MultiVector, ElemWiseScalesData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto alpha =
        gko::batch::initialize<Mtx>({{{1.0, -1.0, 2.2}, {-2.0, 2.0, -0.5}},
                                     {{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}}},
                                    this->exec);

    this->mtx_1->scale(alpha.get());

    auto res =
        gko::batch::initialize<Mtx>({{{1.0, 1.0, 4.84}, {4.0, 4.0, 0.25}},
                                     {{1.0, 6.25, 9.0}, {1.0, 4.0, 9.0}}},
                                    this->exec);
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx_1.get(), res.get(),
                              r<value_type>::value);
}


TYPED_TEST(MultiVector, AddsScaled)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch::initialize<Mtx>(
        {{{2.0, -2.0, 1.5}}, {{2.0, -2.0, 3.0}}}, this->exec);
    auto ualpha = gko::batch::unbatch<gko::batch::MultiVector<T>>(alpha.get());

    this->mtx_1->add_scaled(alpha.get(), this->mtx_0.get());

    this->mtx_10->add_scaled(ualpha[0].get(), this->mtx_00.get());
    this->mtx_11->add_scaled(ualpha[1].get(), this->mtx_01.get());
    auto res =
        gko::batch::unbatch<gko::batch::MultiVector<T>>(this->mtx_1.get());
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->mtx_10.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->mtx_11.get(), 0.);
}


TYPED_TEST(MultiVector, AddsScaledWithScalar)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch::initialize<Mtx>({{2.0}, {-2.0}}, this->exec);
    auto ualpha = gko::batch::unbatch<gko::batch::MultiVector<T>>(alpha.get());

    this->mtx_1->add_scaled(alpha.get(), this->mtx_0.get());

    this->mtx_10->add_scaled(ualpha[0].get(), this->mtx_00.get());
    this->mtx_11->add_scaled(ualpha[1].get(), this->mtx_01.get());
    auto res =
        gko::batch::unbatch<gko::batch::MultiVector<T>>(this->mtx_1.get());
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->mtx_10.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->mtx_11.get(), 0.);
}


TYPED_TEST(MultiVector, AddScaledFailsOnWrongSizes)
{
    using Mtx = typename TestFixture::Mtx;
    auto alpha = gko::batch::initialize<Mtx>(
        {{2.0, 3.0, 4.0, 5.0}, {-2.0, 2.0, 4.0, 5.0}}, this->exec);

    ASSERT_THROW(this->mtx_1->add_scaled(alpha.get(), this->mtx_2.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVector, ComputesDot)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto result =
        Mtx::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>{1, 3}));
    auto ures = gko::batch::unbatch<gko::batch::MultiVector<T>>(result.get());

    this->mtx_0->compute_dot(this->mtx_1.get(), result.get());

    this->mtx_00->compute_dot(this->mtx_10.get(), ures[0].get());
    this->mtx_01->compute_dot(this->mtx_11.get(), ures[1].get());
    auto res = gko::batch::unbatch<gko::batch::MultiVector<T>>(result.get());
    GKO_ASSERT_MTX_NEAR(res[0].get(), ures[0].get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), ures[1].get(), 0.);
}


TYPED_TEST(MultiVector, ComputeDotFailsOnWrongInputSize)
{
    using Mtx = typename TestFixture::Mtx;

    auto result =
        Mtx::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>{1, 3}));

    ASSERT_THROW(this->mtx_1->compute_dot(this->mtx_2.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVector, ComputeDotFailsOnWrongResultSize)
{
    using Mtx = typename TestFixture::Mtx;

    auto result =
        Mtx::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>{1, 2}));

    ASSERT_THROW(this->mtx_0->compute_dot(this->mtx_1.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVector, ComputesConjDot)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto result =
        Mtx::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>{1, 3}));
    auto ures = gko::batch::unbatch<gko::batch::MultiVector<T>>(result.get());

    this->mtx_0->compute_conj_dot(this->mtx_1.get(), result.get());

    this->mtx_00->compute_conj_dot(this->mtx_10.get(), ures[0].get());
    this->mtx_01->compute_conj_dot(this->mtx_11.get(), ures[1].get());
    auto res = gko::batch::unbatch<gko::batch::MultiVector<T>>(result.get());
    GKO_ASSERT_MTX_NEAR(res[0].get(), ures[0].get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), ures[1].get(), 0.);
}


TYPED_TEST(MultiVector, ComputeConjDotFailsOnWrongInputSize)
{
    using Mtx = typename TestFixture::Mtx;

    auto result =
        Mtx::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>{1, 3}));

    ASSERT_THROW(this->mtx_1->compute_conj_dot(this->mtx_2.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVector, ComputeConjDotFailsOnWrongResultSize)
{
    using Mtx = typename TestFixture::Mtx;

    auto result =
        Mtx::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>{1, 2}));

    ASSERT_THROW(this->mtx_0->compute_conj_dot(this->mtx_1.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVector, ComputesNorm2)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using T_nc = gko::remove_complex<T>;
    using NormVector = gko::batch::MultiVector<T_nc>;
    auto mtx(gko::batch::initialize<Mtx>(
        {{I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}},
         {I<T>{-4.0, 2.0}, I<T>{-3.0, -2.0}, I<T>{0.0, 1.0}}},
        this->exec));
    auto batch_size = gko::batch_dim<2>(2, gko::dim<2>{1, 2});
    auto result = NormVector::create(this->exec, batch_size);

    mtx->compute_norm2(result.get());

    EXPECT_EQ(result->at(0, 0, 0), T_nc{3.0});
    EXPECT_EQ(result->at(0, 0, 1), T_nc{5.0});
    EXPECT_EQ(result->at(1, 0, 0), T_nc{5.0});
    EXPECT_EQ(result->at(1, 0, 1), T_nc{3.0});
}


TYPED_TEST(MultiVector, CopiesData)
{
    gko::kernels::reference::batch_multi_vector::copy(
        this->exec, this->mtx_0.get(), this->mtx_1.get());

    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx_1.get(), this->mtx_0.get(), 0.);
}


TYPED_TEST(MultiVector, ConvertsToPrecision)
{
    using MultiVector = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherMultiVector = typename gko::batch::MultiVector<OtherT>;
    auto tmp = OtherMultiVector::create(this->exec);
    auto res = MultiVector::create(this->exec);
    // If OtherT is more precise: 0, otherwise r
    auto residual = r<OtherT>::value < r<T>::value
                        ? gko::remove_complex<T>{0}
                        : gko::remove_complex<T>{r<OtherT>::value};

    this->mtx_1->convert_to(tmp.get());
    tmp->convert_to(res.get());

    auto ures = gko::batch::unbatch<gko::batch::MultiVector<T>>(res.get());
    auto umtx =
        gko::batch::unbatch<gko::batch::MultiVector<T>>(this->mtx_1.get());
    GKO_ASSERT_MTX_NEAR(umtx[0].get(), ures[0].get(), residual);
    GKO_ASSERT_MTX_NEAR(umtx[1].get(), ures[1].get(), residual);
}


TYPED_TEST(MultiVector, MovesToPrecision)
{
    using MultiVector = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherMultiVector = typename gko::batch::MultiVector<OtherT>;
    auto tmp = OtherMultiVector::create(this->exec);
    auto res = MultiVector::create(this->exec);
    // If OtherT is more precise: 0, otherwise r
    auto residual = r<OtherT>::value < r<T>::value
                        ? gko::remove_complex<T>{0}
                        : gko::remove_complex<T>{r<OtherT>::value};

    this->mtx_1->move_to(tmp.get());
    tmp->move_to(res.get());

    auto ures = gko::batch::unbatch<gko::batch::MultiVector<T>>(res.get());
    auto umtx =
        gko::batch::unbatch<gko::batch::MultiVector<T>>(this->mtx_1.get());
    GKO_ASSERT_MTX_NEAR(umtx[0].get(), ures[0].get(), residual);
    GKO_ASSERT_MTX_NEAR(umtx[1].get(), ures[1].get(), residual);
}


TYPED_TEST(MultiVector, ConvertsEmptyToPrecision)
{
    using MultiVector = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherMultiVector = typename gko::batch::MultiVector<OtherT>;
    auto empty = OtherMultiVector::create(this->exec);
    auto res = MultiVector::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_FALSE(res->get_num_batch_items());
}


TYPED_TEST(MultiVector, MovesEmptyToPrecision)
{
    using MultiVector = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherMultiVector = typename gko::batch::MultiVector<OtherT>;
    auto empty = OtherMultiVector::create(this->exec);
    auto res = MultiVector::create(this->exec);

    empty->move_to(res.get());

    ASSERT_FALSE(res->get_num_batch_items());
}
