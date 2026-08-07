// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/multivector_kernels.hpp"

#include <complex>
#include <memory>
#include <numeric>
#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/multivector.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/scaled_permutation.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename T>
class MultiVector : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::MultiVector<value_type>;
    using MixedMtx = gko::matrix::MultiVector<gko::next_precision<value_type>>;
    using ComplexMtx = gko::to_complex<Mtx>;
    using RealMtx = gko::remove_complex<Mtx>;
    MultiVector() : exec(gko::ReferenceExecutor::create())
    {
        mtx1 =
            gko::initialize<Mtx>(4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}}, exec);
        mtx2 =
            gko::initialize<Mtx>({I<T>({1.0, -1.0}), I<T>({-2.0, 2.0})}, exec);
        mtx3 =
            gko::initialize<Mtx>(4, {{1.0, 2.0, 3.0}, {0.5, 1.5, 2.5}}, exec);
        mtx4 =
            gko::initialize<Mtx>(4, {{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}, exec);
        mtx5 = gko::initialize<Mtx>(
            {{1.0, -1.0, -0.5}, {-2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}, exec);
        mtx6 = gko::initialize<Mtx>({{1.0, 2.0, 0.0}, {0.0, 1.5, 0.0}}, exec);
        mtx7 = gko::initialize<Mtx>({{1.0, 2.0, 3.0}, {0.0, 1.5, 0.0}}, exec);
        mtx8 = gko::initialize<Mtx>(
            {I<T>({1.0, -1.0}), I<T>({-2.0, 2.0}), I<T>({-3.0, 3.0})}, exec);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx1;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<Mtx> mtx3;
    std::unique_ptr<Mtx> mtx4;
    std::unique_ptr<Mtx> mtx5;
    std::unique_ptr<Mtx> mtx6;
    std::unique_ptr<Mtx> mtx7;
    std::unique_ptr<Mtx> mtx8;
    std::default_random_engine rand_engine;

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<gko::size_type>(num_cols, num_cols),
            std::normal_distribution<>(0.0, 1.0), rand_engine, exec);
    }
};


TYPED_TEST_SUITE(MultiVector, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(MultiVector, CopyRespectsStride)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::initialize<gko::matrix::MultiVector<TypeParam>>({1.0, 2.0},
                                                                  this->exec);
    auto m2 = gko::matrix::MultiVector<TypeParam>::create(this->exec,
                                                          gko::dim<2>{2, 1}, 2);
    auto original_data = m2->get_values();
    original_data[1] = TypeParam{3.0};

    m->convert_to(m2);

    EXPECT_EQ(m2->at(0, 0), value_type{1.0});
    EXPECT_EQ(m2->get_stride(), 2);
    EXPECT_EQ(m2->at(1, 0), value_type{2.0});
    EXPECT_EQ(m2->get_values(), original_data);
    EXPECT_EQ(original_data[1], TypeParam{3.0});
}


TYPED_TEST(MultiVector, TemporaryOutputCloneWorks)
{
    using value_type = typename TestFixture::value_type;
    auto other = gko::OmpExecutor::create();
    auto m =
        gko::initialize<gko::matrix::MultiVector<TypeParam>>({1.0, 2.0}, other);

    {
        auto clone = gko::make_temporary_output_clone(this->exec, m);
        clone->at(0) = 4.0;
        clone->at(1) = 5.0;

        ASSERT_EQ(m->at(0), value_type{1.0});
        ASSERT_EQ(m->at(1), value_type{2.0});
        ASSERT_EQ(clone->get_size(), m->get_size());
        ASSERT_EQ(clone->get_executor(), this->exec);
    }
    ASSERT_EQ(m->at(0), value_type{4.0});
    ASSERT_EQ(m->at(1), value_type{5.0});
}


TYPED_TEST(MultiVector, CanBeFilledWithValue)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::initialize<gko::matrix::MultiVector<TypeParam>>({1.0, 2.0},
                                                                  this->exec);
    EXPECT_EQ(m->at(0), value_type{1});
    EXPECT_EQ(m->at(1), value_type{2});

    m->fill(value_type{42});

    EXPECT_EQ(m->at(0), value_type{42});
    EXPECT_EQ(m->at(1), value_type{42});
}


TYPED_TEST(MultiVector, CanBeFilledWithValueForStridedMatrices)
{
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto m = gko::initialize<gko::matrix::MultiVector<TypeParam>>(
        4, {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, this->exec);
    T in_stride{-1.0};
    m->get_values()[3] = in_stride;

    ASSERT_EQ(m->get_size(), gko::dim<2>(3, 2));
    ASSERT_EQ(m->get_num_stored_elements(), 12);
    EXPECT_EQ(m->at(0), value_type{1.0});
    EXPECT_EQ(m->at(1), value_type{2.0});
    EXPECT_EQ(m->at(2), value_type{3.0});
    EXPECT_EQ(m->at(3), value_type{4.0});
    EXPECT_EQ(m->at(4), value_type{5.0});
    EXPECT_EQ(m->at(5), value_type{6.0});

    m->fill(value_type{42});

    ASSERT_EQ(m->get_size(), gko::dim<2>(3, 2));
    EXPECT_EQ(m->get_num_stored_elements(), 12);
    EXPECT_EQ(m->at(0), value_type{42.0});
    EXPECT_EQ(m->at(1), value_type{42.0});
    EXPECT_EQ(m->at(2), value_type{42.0});
    EXPECT_EQ(m->at(3), value_type{42.0});
    EXPECT_EQ(m->at(4), value_type{42.0});
    EXPECT_EQ(m->at(5), value_type{42.0});
    ASSERT_EQ(m->get_values()[3], in_stride);
}


TYPED_TEST(MultiVector, ScalesData)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({I<T>{2.0, -2.0}}, this->exec);

    this->mtx2->scale(alpha);

    EXPECT_EQ(this->mtx2->at(0, 0), T{2.0});
    EXPECT_EQ(this->mtx2->at(0, 1), T{2.0});
    EXPECT_EQ(this->mtx2->at(1, 0), T{-4.0});
    EXPECT_EQ(this->mtx2->at(1, 1), T{-4.0});
}


TYPED_TEST(MultiVector, ScalesDataMixed)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using MixedT = typename MixedMtx::value_type;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<MixedMtx>({I<MixedT>{2.0, -2.0}}, this->exec);

    this->mtx2->scale(alpha);

    EXPECT_EQ(this->mtx2->at(0, 0), T{2.0});
    EXPECT_EQ(this->mtx2->at(0, 1), T{2.0});
    EXPECT_EQ(this->mtx2->at(1, 0), T{-4.0});
    EXPECT_EQ(this->mtx2->at(1, 1), T{-4.0});
}


TYPED_TEST(MultiVector, InvScalesData)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({I<T>{0.5, -0.5}}, this->exec);

    this->mtx2->inv_scale(alpha);

    EXPECT_EQ(this->mtx2->at(0, 0), T{2.0});
    EXPECT_EQ(this->mtx2->at(0, 1), T{2.0});
    EXPECT_EQ(this->mtx2->at(1, 0), T{-4.0});
    EXPECT_EQ(this->mtx2->at(1, 1), T{-4.0});
}


TYPED_TEST(MultiVector, ScalesDataWithScalar)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);

    this->mtx2->scale(alpha);

    EXPECT_EQ(this->mtx2->at(0, 0), T{2.0});
    EXPECT_EQ(this->mtx2->at(0, 1), T{-2.0});
    EXPECT_EQ(this->mtx2->at(1, 0), T{-4.0});
    EXPECT_EQ(this->mtx2->at(1, 1), T{4.0});
}


TYPED_TEST(MultiVector, ScalesDataWithZeroScalarNaN)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({I<T>{0.0}}, this->exec);
    this->mtx2->fill(gko::nan<T>());

    this->mtx2->scale(alpha);

    EXPECT_EQ(this->mtx2->at(0, 0), T{0.0});
    EXPECT_EQ(this->mtx2->at(0, 1), T{0.0});
    EXPECT_EQ(this->mtx2->at(1, 0), T{0.0});
    EXPECT_EQ(this->mtx2->at(1, 1), T{0.0});
}


TYPED_TEST(MultiVector, InvScalesDataWithScalar)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({0.5}, this->exec);

    this->mtx2->inv_scale(alpha);

    EXPECT_EQ(this->mtx2->at(0, 0), T{2.0});
    EXPECT_EQ(this->mtx2->at(0, 1), T{-2.0});
    EXPECT_EQ(this->mtx2->at(1, 0), T{-4.0});
    EXPECT_EQ(this->mtx2->at(1, 1), T{4.0});
}


TYPED_TEST(MultiVector, ScalesDataWithStride)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({{-1.0, 1.0, 2.0}}, this->exec);
    T in_stride{-1};
    this->mtx1->get_values()[3] = in_stride;

    this->mtx1->scale(alpha);

    EXPECT_EQ(this->mtx1->at(0, 0), T{-1.0});
    EXPECT_EQ(this->mtx1->at(0, 1), T{2.0});
    EXPECT_EQ(this->mtx1->at(0, 2), T{6.0});
    EXPECT_EQ(this->mtx1->at(1, 0), T{-1.5});
    EXPECT_EQ(this->mtx1->at(1, 1), T{2.5});
    EXPECT_EQ(this->mtx1->at(1, 2), T{7.0});
    ASSERT_EQ(this->mtx1->get_values()[3], in_stride);
}


TYPED_TEST(MultiVector, AddsScaled)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({{2.0, 1.0, -2.0}}, this->exec);
    T in_stride{-1};
    this->mtx1->get_values()[3] = in_stride;

    this->mtx1->add_scaled(alpha, this->mtx3);

    EXPECT_EQ(this->mtx1->at(0, 0), T{3.0});
    EXPECT_EQ(this->mtx1->at(0, 1), T{4.0});
    EXPECT_EQ(this->mtx1->at(0, 2), T{-3.0});
    EXPECT_EQ(this->mtx1->at(1, 0), T{2.5});
    EXPECT_EQ(this->mtx1->at(1, 1), T{4.0});
    EXPECT_EQ(this->mtx1->at(1, 2), T{-1.5});
    ASSERT_EQ(this->mtx1->get_values()[3], in_stride);
}


TYPED_TEST(MultiVector, AddsScaledMixed)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using T = typename TestFixture::value_type;
    auto mmtx3 = MixedMtx::create(this->exec);
    this->mtx3->convert_to(mmtx3);
    auto alpha = gko::initialize<MixedMtx>({{2.0, 1.0, -2.0}}, this->exec);
    T in_stride{-1};
    this->mtx1->get_values()[3] = in_stride;

    this->mtx1->add_scaled(alpha, this->mtx3);

    EXPECT_EQ(this->mtx1->at(0, 0), T{3.0});
    EXPECT_EQ(this->mtx1->at(0, 1), T{4.0});
    EXPECT_EQ(this->mtx1->at(0, 2), T{-3.0});
    EXPECT_EQ(this->mtx1->at(1, 0), T{2.5});
    EXPECT_EQ(this->mtx1->at(1, 1), T{4.0});
    EXPECT_EQ(this->mtx1->at(1, 2), T{-1.5});
    ASSERT_EQ(this->mtx1->get_values()[3], in_stride);
}


TYPED_TEST(MultiVector, SubtractsScaled)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({{-2.0, -1.0, 2.0}}, this->exec);
    T in_stride{-1};
    this->mtx1->get_values()[3] = in_stride;

    this->mtx1->sub_scaled(alpha, this->mtx3);

    EXPECT_EQ(this->mtx1->at(0, 0), T{3.0});
    EXPECT_EQ(this->mtx1->at(0, 1), T{4.0});
    EXPECT_EQ(this->mtx1->at(0, 2), T{-3.0});
    EXPECT_EQ(this->mtx1->at(1, 0), T{2.5});
    EXPECT_EQ(this->mtx1->at(1, 1), T{4.0});
    EXPECT_EQ(this->mtx1->at(1, 2), T{-1.5});
    ASSERT_EQ(this->mtx1->get_values()[3], in_stride);
}


TYPED_TEST(MultiVector, AddsScaledWithScalar)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    T in_stride{-1};
    this->mtx1->get_values()[3] = in_stride;

    this->mtx1->add_scaled(alpha, this->mtx3);

    EXPECT_EQ(this->mtx1->at(0, 0), T{3.0});
    EXPECT_EQ(this->mtx1->at(0, 1), T{6.0});
    EXPECT_EQ(this->mtx1->at(0, 2), T{9.0});
    EXPECT_EQ(this->mtx1->at(1, 0), T{2.5});
    EXPECT_EQ(this->mtx1->at(1, 1), T{5.5});
    EXPECT_EQ(this->mtx1->at(1, 2), T{8.5});
    ASSERT_EQ(this->mtx1->get_values()[3], in_stride);
}


TYPED_TEST(MultiVector, AddsScaledWithZeroScalar)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({0.0}, this->exec);
    this->mtx3->fill(gko::nan<T>());
    const auto expected = this->mtx1->clone();

    this->mtx1->add_scaled(alpha, this->mtx3);

    GKO_ASSERT_MTX_NEAR(this->mtx1, expected, 0.0);
}


TYPED_TEST(MultiVector, SubtractsScaledWithZeroScalar)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({0.0}, this->exec);
    this->mtx3->fill(gko::nan<T>());
    const auto expected = this->mtx1->clone();

    this->mtx1->sub_scaled(alpha, this->mtx3);

    GKO_ASSERT_MTX_NEAR(this->mtx1, expected, 0.0);
}


TYPED_TEST(MultiVector, AddScaledFailsOnWrongSizes)
{
    using Mtx = typename TestFixture::Mtx;
    auto alpha = Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->mtx1->add_scaled(alpha, this->mtx2),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVector, AddsScaledDiag)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto diag = gko::matrix::Diagonal<T>::create(
        this->exec, 2, gko::array<T>{this->exec, {3.0, 2.0}});

    this->mtx2->add_scaled(alpha, diag);

    ASSERT_EQ(this->mtx2->at(0, 0), T{7.0});
    ASSERT_EQ(this->mtx2->at(0, 1), T{-1.0});
    ASSERT_EQ(this->mtx2->at(1, 0), T{-2.0});
    ASSERT_EQ(this->mtx2->at(1, 1), T{6.0});
}


TYPED_TEST(MultiVector, SubtractsScaledDiag)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({-2.0}, this->exec);
    auto diag = gko::matrix::Diagonal<T>::create(
        this->exec, 2, gko::array<T>{this->exec, {3.0, 2.0}});

    this->mtx2->sub_scaled(alpha, diag);

    ASSERT_EQ(this->mtx2->at(0, 0), T{7.0});
    ASSERT_EQ(this->mtx2->at(0, 1), T{-1.0});
    ASSERT_EQ(this->mtx2->at(1, 0), T{-2.0});
    ASSERT_EQ(this->mtx2->at(1, 1), T{6.0});
}


TYPED_TEST(MultiVector, ComputesDot)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 3});

    this->mtx1->compute_dot(this->mtx3, result);

    EXPECT_EQ(result->at(0, 0), T{1.75});
    EXPECT_EQ(result->at(0, 1), T{7.75});
    ASSERT_EQ(result->at(0, 2), T{17.75});
}


TYPED_TEST(MultiVector, ComputesDotMixed)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using MixedT = typename MixedMtx::value_type;
    auto mmtx3 = MixedMtx::create(this->exec);
    this->mtx3->convert_to(mmtx3);
    auto result = MixedMtx::create(this->exec, gko::dim<2>{1, 3});

    this->mtx1->compute_dot(this->mtx3, result);

    EXPECT_EQ(result->at(0, 0), MixedT{1.75});
    EXPECT_EQ(result->at(0, 1), MixedT{7.75});
    ASSERT_EQ(result->at(0, 2), MixedT{17.75});
}


TYPED_TEST(MultiVector, ComputesConjDot)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 3});

    this->mtx1->compute_conj_dot(this->mtx3, result);

    EXPECT_EQ(result->at(0, 0), T{1.75});
    EXPECT_EQ(result->at(0, 1), T{7.75});
    ASSERT_EQ(result->at(0, 2), T{17.75});
}


TYPED_TEST(MultiVector, ComputesConjDotMixed)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using MixedT = typename MixedMtx::value_type;
    auto mmtx3 = MixedMtx::create(this->exec);
    this->mtx3->convert_to(mmtx3);
    auto result = MixedMtx::create(this->exec, gko::dim<2>{1, 3});

    this->mtx1->compute_conj_dot(this->mtx3, result);

    EXPECT_EQ(result->at(0, 0), MixedT{1.75});
    EXPECT_EQ(result->at(0, 1), MixedT{7.75});
    ASSERT_EQ(result->at(0, 2), MixedT{17.75});
}


TYPED_TEST(MultiVector, ComputesNorm2)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using T_nc = gko::remove_complex<T>;
    using NormVector = gko::matrix::MultiVector<T_nc>;
    auto mtx(gko::initialize<Mtx>(
        {I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}}, this->exec));
    auto result = NormVector::create(this->exec, gko::dim<2>{1, 2});

    mtx->compute_norm2(result);

    EXPECT_EQ(result->at(0, 0), T_nc{3.0});
    EXPECT_EQ(result->at(0, 1), T_nc{5.0});
}


TYPED_TEST(MultiVector, ComputesNorm2Mixed)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using MixedMtx = typename TestFixture::MixedMtx;
    using MixedT = typename MixedMtx::value_type;
    using MixedT_nc = gko::remove_complex<MixedT>;
    using MixedNormVector = gko::matrix::MultiVector<MixedT_nc>;
    auto mtx(gko::initialize<Mtx>(
        {I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}}, this->exec));
    auto result = MixedNormVector::create(this->exec, gko::dim<2>{1, 2});

    mtx->compute_norm2(result);

    EXPECT_EQ(result->at(0, 0), MixedT_nc{3.0});
    EXPECT_EQ(result->at(0, 1), MixedT_nc{5.0});
}


TYPED_TEST(MultiVector, ComputesNorm2Squared)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using T_nc = gko::remove_complex<T>;
    using NormVector = gko::matrix::MultiVector<T_nc>;
    gko::array<char> tmp{this->exec};
    auto mtx(gko::initialize<Mtx>(
        {I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}}, this->exec));
    auto result = NormVector::create(this->exec, gko::dim<2>{1, 2});

    gko::kernels::reference::multivector::compute_squared_norm2(
        gko::as<gko::ReferenceExecutor>(this->exec),
        mtx->get_const_device_view(), result->get_device_view(), tmp);

    EXPECT_EQ(result->at(0, 0), T_nc{9.0});
    EXPECT_EQ(result->at(0, 1), T_nc{25.0});
}


TYPED_TEST(MultiVector, ComputesSqrt)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using T_nc = gko::remove_complex<T>;
    using NormVector = gko::matrix::MultiVector<T_nc>;
    auto mtx(gko::initialize<NormVector>(I<I<T_nc>>{{9.0, 25.0}}, this->exec));

    gko::kernels::reference::multivector::compute_sqrt(
        gko::as<gko::ReferenceExecutor>(this->exec), mtx->get_device_view());

    EXPECT_EQ(mtx->at(0, 0), T_nc{3.0});
    EXPECT_EQ(mtx->at(0, 1), T_nc{5.0});
}


TYPED_TEST(MultiVector, ComputesNorm1)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using T_nc = gko::remove_complex<T>;
    using NormVector = gko::matrix::MultiVector<T_nc>;
    auto mtx(gko::initialize<Mtx>(
        {I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}, I<T>{-1.0, -1.0}},
        this->exec));
    auto result = NormVector::create(this->exec, gko::dim<2>{1, 2});

    mtx->compute_norm1(result);

    EXPECT_EQ(result->at(0, 0), T_nc{6.0});
    EXPECT_EQ(result->at(0, 1), T_nc{8.0});
}


TYPED_TEST(MultiVector, ComputesNorm1Mixed)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using MixedT = typename MixedMtx::value_type;
    using MixedT_nc = gko::remove_complex<MixedT>;
    using MixedNormVector = gko::matrix::MultiVector<MixedT_nc>;
    auto mtx(
        gko::initialize<MixedMtx>({I<MixedT>{1.0, 0.0}, I<MixedT>{2.0, 3.0},
                                   I<MixedT>{2.0, 4.0}, I<MixedT>{-1.0, -1.0}},
                                  this->exec));
    auto result = MixedNormVector::create(this->exec, gko::dim<2>{1, 2});

    mtx->compute_norm1(result);

    EXPECT_EQ(result->at(0, 0), MixedT_nc{6.0});
    EXPECT_EQ(result->at(0, 1), MixedT_nc{8.0});
}


TYPED_TEST(MultiVector, ComputesMean)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;

    auto iota = Mtx::create(this->exec, gko::dim<2>{10, 1});
    std::iota(iota->get_values(), iota->get_values() + 10, 1);
    auto iota_result = Mtx::create(this->exec, gko::dim<2>{1, 1});
    iota->compute_mean(iota_result.get());
    GKO_EXPECT_NEAR(iota_result->at(0, 0), T{5.5}, r<T>::value * 10);

    auto result = Mtx::create(this->exec, gko::dim<2>{1, 3});

    this->mtx4->compute_mean(result.get());

    GKO_EXPECT_NEAR(result->at(0, 0), T{0.5}, r<T>::value * 10);
    GKO_EXPECT_NEAR(result->at(0, 1), T{4.0}, r<T>::value * 10);
    GKO_EXPECT_NEAR(result->at(0, 2), T{1.0}, r<T>::value * 10);
}


TYPED_TEST(MultiVector, ComputesMeanFailsOnWrongResultSize)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->mtx4->compute_mean(result), gko::DimensionMismatch);
}


TYPED_TEST(MultiVector, ComputeDotFailsOnWrongInputSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 3});

    ASSERT_THROW(this->mtx1->compute_dot(this->mtx2, result),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVector, ComputeDotFailsOnWrongResultSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->mtx1->compute_dot(this->mtx3, result),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVector, ComputeConjDotFailsOnWrongInputSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 3});

    ASSERT_THROW(this->mtx1->compute_conj_dot(this->mtx2, result),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVector, ComputeConjDotFailsOnWrongResultSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->mtx1->compute_conj_dot(this->mtx3, result),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVector, ConvertsToPrecision)
{
    using MultiVector = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherMultiVector = typename gko::matrix::MultiVector<OtherT>;
    auto tmp = OtherMultiVector::create(this->exec);
    auto res = MultiVector::create(this->exec);
    // If OtherT is more precise: 0, otherwise r
    auto residual =
        r<OtherT>::value < r<T>::value
            ? gko::remove_complex<T>{0}
            : gko::remove_complex<T>{
                  static_cast<gko::remove_complex<T>>(r<OtherT>::value)};

    this->mtx1->convert_to(tmp);
    tmp->convert_to(res);

    GKO_ASSERT_MTX_NEAR(this->mtx1, res, residual);
}


TYPED_TEST(MultiVector, MovesToPrecision)
{
    using MultiVector = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherMultiVector = typename gko::matrix::MultiVector<OtherT>;
    auto tmp = OtherMultiVector::create(this->exec);
    auto res = MultiVector::create(this->exec);
    // If OtherT is more precise: 0, otherwise r
    auto residual =
        r<OtherT>::value < r<T>::value
            ? gko::remove_complex<T>{0}
            : gko::remove_complex<T>{
                  static_cast<gko::remove_complex<T>>(r<OtherT>::value)};

    this->mtx1->move_to(tmp);
    tmp->move_to(res);

    GKO_ASSERT_MTX_NEAR(this->mtx1, res, residual);
}


TYPED_TEST(MultiVector, SquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = gko::as<Mtx>(this->mtx5->transpose());

    GKO_ASSERT_MTX_NEAR(
        trans, l<T>({{1.0, -2.0, 2.1}, {-1.0, 2.0, 3.4}, {-0.5, 4.5, 1.2}}),
        0.0);
}


TYPED_TEST(MultiVector, SquareMatrixIsTransposableIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = Mtx::create(this->exec, this->mtx5->get_size());

    this->mtx5->transpose(trans);

    GKO_ASSERT_MTX_NEAR(
        trans, l<T>({{1.0, -2.0, 2.1}, {-1.0, 2.0, 3.4}, {-0.5, 4.5, 1.2}}),
        0.0);
}


TYPED_TEST(MultiVector, SquareSubmatrixIsTransposableIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = Mtx::create(this->exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {0, 2})->transpose(trans);

    GKO_ASSERT_MTX_NEAR(trans, l<T>({{1.0, -2.0}, {-1.0, 2.0}}), 0.0);
    ASSERT_EQ(trans->get_stride(), 4);
}


TYPED_TEST(MultiVector,
           SquareMatrixIsTransposableIntoMultiVectorFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;

    ASSERT_THROW(this->mtx5->transpose(Mtx::create(this->exec)),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVector, NonSquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = gko::as<Mtx>(this->mtx4->transpose());

    GKO_ASSERT_MTX_NEAR(trans, l<T>({{1.0, 0.0}, {3.0, 5.0}, {2.0, 0.0}}), 0.0);
}


TYPED_TEST(MultiVector, NonSquareMatrixIsTransposableIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans =
        Mtx::create(this->exec, gko::transpose(this->mtx4->get_size()));

    this->mtx4->transpose(trans);

    GKO_ASSERT_MTX_NEAR(trans, l<T>({{1.0, 0.0}, {3.0, 5.0}, {2.0, 0.0}}), 0.0);
}


TYPED_TEST(MultiVector, NonSquareSubmatrixIsTransposableIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = Mtx::create(this->exec, gko::dim<2>{2, 1}, 5);

    this->mtx4->create_submatrix({0, 1}, {0, 2})->transpose(trans);

    GKO_ASSERT_MTX_NEAR(trans, l({1.0, 3.0}), 0.0);
    ASSERT_EQ(trans->get_stride(), 5);
}


TYPED_TEST(MultiVector,
           NonSquareMatrixIsTransposableIntoMultiVectorFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;

    ASSERT_THROW(this->mtx4->transpose(Mtx::create(this->exec)),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVector, ExtractsDiagonalFromSquareMatrix)
{
    using T = typename TestFixture::value_type;

    auto diag = this->mtx5->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 3);
    ASSERT_EQ(diag->get_size()[1], 3);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{2.});
    ASSERT_EQ(diag->get_values()[2], T{1.2});
}


TYPED_TEST(MultiVector, ExtractsDiagonalFromTallSkinnyMatrix)
{
    using T = typename TestFixture::value_type;

    auto diag = this->mtx4->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{5.});
}


TYPED_TEST(MultiVector, ExtractsDiagonalFromShortFatMatrix)
{
    using T = typename TestFixture::value_type;

    auto diag = this->mtx8->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{2.});
}


TYPED_TEST(MultiVector, ExtractsDiagonalFromSquareMatrixIntoDiagonal)
{
    using T = typename TestFixture::value_type;
    auto diag = gko::matrix::Diagonal<T>::create(this->exec, 3);

    this->mtx5->extract_diagonal(diag);

    ASSERT_EQ(diag->get_size()[0], 3);
    ASSERT_EQ(diag->get_size()[1], 3);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{2.});
    ASSERT_EQ(diag->get_values()[2], T{1.2});
}


TYPED_TEST(MultiVector, ExtractsDiagonalFromTallSkinnyMatrixIntoDiagonal)
{
    using T = typename TestFixture::value_type;
    auto diag = gko::matrix::Diagonal<T>::create(this->exec, 2);

    this->mtx4->extract_diagonal(diag);

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{5.});
}


TYPED_TEST(MultiVector, ExtractsDiagonalFromShortFatMatrixIntoDiagonal)
{
    using T = typename TestFixture::value_type;
    auto diag = gko::matrix::Diagonal<T>::create(this->exec, 2);

    this->mtx8->extract_diagonal(diag);

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{2.});
}


TYPED_TEST(MultiVector, InplaceAbsolute)
{
    using T = typename TestFixture::value_type;

    this->mtx5->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        this->mtx5, l<T>({{1.0, 1.0, 0.5}, {2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}),
        0.0);
}


TYPED_TEST(MultiVector, InplaceAbsoluteSubMatrix)
{
    using T = typename TestFixture::value_type;
    auto mtx = this->mtx5->create_submatrix(gko::span{0, 2}, gko::span{0, 2});

    mtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        this->mtx5, l<T>({{1.0, 1.0, -0.5}, {2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}),
        0.0);
}


TYPED_TEST(MultiVector, OutplaceAbsolute)
{
    using T = typename TestFixture::value_type;

    auto abs_mtx = this->mtx5->compute_absolute();

    GKO_ASSERT_MTX_NEAR(
        abs_mtx, l<T>({{1.0, 1.0, 0.5}, {2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}),
        0.0);
}


TYPED_TEST(MultiVector, OutplaceAbsoluteIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto abs_mtx =
        gko::remove_complex<Mtx>::create(this->exec, this->mtx5->get_size());

    this->mtx5->compute_absolute(abs_mtx);

    GKO_ASSERT_MTX_NEAR(
        abs_mtx, l<T>({{1.0, 1.0, 0.5}, {2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}),
        0.0);
}


TYPED_TEST(MultiVector, OutplaceAbsoluteSubMatrix)
{
    using T = typename TestFixture::value_type;
    auto mtx = this->mtx5->create_submatrix(gko::span{0, 2}, gko::span{0, 2});

    auto abs_mtx = mtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx, l<T>({{1.0, 1.0}, {2.0, 2.0}}), 0);
    GKO_ASSERT_EQ(abs_mtx->get_stride(), 2);
}


TYPED_TEST(MultiVector, OutplaceSubmatrixAbsoluteIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto mtx = this->mtx5->create_submatrix(gko::span{0, 2}, gko::span{0, 2});
    auto abs_mtx =
        gko::remove_complex<Mtx>::create(this->exec, gko::dim<2>{2, 2}, 4);

    mtx->compute_absolute(abs_mtx);

    GKO_ASSERT_MTX_NEAR(abs_mtx, l<T>({{1.0, 1.0}, {2.0, 2.0}}), 0);
    GKO_ASSERT_EQ(abs_mtx->get_stride(), 4);
}


TYPED_TEST(MultiVector, MakeComplex)
{
    using T = typename TestFixture::value_type;

    auto complex_mtx = this->mtx5->make_complex();

    GKO_ASSERT_MTX_NEAR(complex_mtx, this->mtx5, 0.0);
}


TYPED_TEST(MultiVector, MakeComplexIntoMultiVector)
{
    using T = typename TestFixture::value_type;
    using ComplexMtx = typename TestFixture::ComplexMtx;
    auto exec = this->mtx5->get_executor();

    auto complex_mtx = ComplexMtx::create(exec, this->mtx5->get_size());
    this->mtx5->make_complex(complex_mtx);

    GKO_ASSERT_MTX_NEAR(complex_mtx, this->mtx5, 0.0);
}


TYPED_TEST(MultiVector, MakeComplexIntoMultiVectorFailsForWrongDimensions)
{
    using T = typename TestFixture::value_type;
    using ComplexMtx = typename TestFixture::ComplexMtx;
    auto exec = this->mtx5->get_executor();

    auto complex_mtx = ComplexMtx::create(exec);

    ASSERT_THROW(this->mtx5->make_complex(complex_mtx), gko::DimensionMismatch);
}


TYPED_TEST(MultiVector, GetReal)
{
    using T = typename TestFixture::value_type;

    auto real_mtx = this->mtx5->get_real();

    GKO_ASSERT_MTX_NEAR(real_mtx, this->mtx5, 0.0);
}


TYPED_TEST(MultiVector, GetRealIntoMultiVector)
{
    using T = typename TestFixture::value_type;
    using RealMtx = typename TestFixture::RealMtx;
    auto exec = this->mtx5->get_executor();

    auto real_mtx = RealMtx::create(exec, this->mtx5->get_size());
    this->mtx5->get_real(real_mtx);

    GKO_ASSERT_MTX_NEAR(real_mtx, this->mtx5, 0.0);
}


TYPED_TEST(MultiVector, GetRealIntoMultiVectorFailsForWrongDimensions)
{
    using T = typename TestFixture::value_type;
    using RealMtx = typename TestFixture::RealMtx;
    auto exec = this->mtx5->get_executor();

    auto real_mtx = RealMtx::create(exec);
    ASSERT_THROW(this->mtx5->get_real(real_mtx), gko::DimensionMismatch);
}


TYPED_TEST(MultiVector, GetImag)
{
    using T = typename TestFixture::value_type;

    auto imag_mtx = this->mtx5->get_imag();

    GKO_ASSERT_MTX_NEAR(
        imag_mtx, l<T>({{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}),
        0.0);
}


TYPED_TEST(MultiVector, GetImagIntoMultiVector)
{
    using T = typename TestFixture::value_type;
    using RealMtx = typename TestFixture::RealMtx;
    auto exec = this->mtx5->get_executor();

    auto imag_mtx = RealMtx::create(exec, this->mtx5->get_size());
    this->mtx5->get_imag(imag_mtx);

    GKO_ASSERT_MTX_NEAR(
        imag_mtx, l<T>({{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}),
        0.0);
}


TYPED_TEST(MultiVector, GetImagIntoMultiVectorFailsForWrongDimensions)
{
    using T = typename TestFixture::value_type;
    using RealMtx = typename TestFixture::RealMtx;
    auto exec = this->mtx5->get_executor();

    auto imag_mtx = RealMtx::create(exec);
    ASSERT_THROW(this->mtx5->get_imag(imag_mtx), gko::DimensionMismatch);
}


TYPED_TEST(MultiVector, MakeTemporaryConversionDoesntConvertOnMatch)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({8.0}, this->exec);

    ASSERT_EQ(gko::make_temporary_conversion<T>(alpha).get(), alpha.get());
}


TYPED_TEST(MultiVector, MakeTemporaryConversionConvertsBack)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using T = typename TestFixture::value_type;
    using MixedT = typename MixedMtx::value_type;
    auto alpha = gko::initialize<MixedMtx>({8.0}, this->exec);

    {
        auto conversion = gko::make_temporary_conversion<T>(alpha);
        conversion->at(0, 0) = T{7.0};
    }

    ASSERT_EQ(alpha->at(0, 0), MixedT{7.0});
}


TYPED_TEST(MultiVector, MakeTemporaryConversionConstDoesntConvertBack)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using T = typename TestFixture::value_type;
    using MixedT = typename MixedMtx::value_type;
    auto alpha = gko::initialize<MixedMtx>({8.0}, this->exec);

    {
        auto conversion = gko::make_temporary_conversion<T>(
            static_cast<const MixedMtx*>(alpha.get()));
        alpha->at(0, 0) = MixedT{7.0};
    }

    ASSERT_EQ(alpha->at(0, 0), MixedT{7.0});
}


TYPED_TEST(MultiVector, ScaleAddIdentityRectangular)
{
    using T = typename TestFixture::value_type;
    using Vec = typename TestFixture::Mtx;
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto b = gko::initialize<Vec>(
        {I<T>{2.0, 0.0}, I<T>{1.0, 2.5}, I<T>{0.0, -4.0}}, this->exec);

    b->add_scaled_identity(alpha, beta);

    GKO_ASSERT_MTX_NEAR(b, l({{0.0, 0.0}, {-1.0, -0.5}, {0.0, 4.0}}), 0.0);
}


template <typename ValueIndexType>
class MultiVectorWithIndexType
    : public MultiVector<
          typename std::tuple_element<0, decltype(ValueIndexType())>::type> {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Permutation = gko::matrix::Permutation<index_type>;
    using ScaledPermutation =
        gko::matrix::ScaledPermutation<value_type, index_type>;


    MultiVectorWithIndexType()
    {
        perm2 = Permutation::create(this->exec,
                                    gko::array<index_type>{this->exec, {1, 0}});
        perm3 = Permutation::create(
            this->exec, gko::array<index_type>{this->exec, {1, 2, 0}});
        perm3_rev = Permutation::create(
            this->exec, gko::array<index_type>{this->exec, {2, 0, 1}});
        perm0 = Permutation::create(this->exec, 0);
        scale_perm2 = ScaledPermutation::create(
            this->exec, gko::array<value_type>{this->exec, {17.0, 19.0}},
            gko::array<index_type>{this->exec, {1, 0}});
        scale_perm3 = ScaledPermutation::create(
            this->exec, gko::array<value_type>{this->exec, {2.0, 3.0, 5.0}},
            gko::array<index_type>{this->exec, {1, 2, 0}});
        scale_perm3_rev = ScaledPermutation::create(
            this->exec, gko::array<value_type>{this->exec, {7.0, 11.0, 13.0}},
            gko::array<index_type>{this->exec, {2, 0, 1}});
        scale_perm0 = ScaledPermutation::create(this->exec, 0);
    }

    std::unique_ptr<Permutation> perm2;
    std::unique_ptr<Permutation> perm3;
    std::unique_ptr<Permutation> perm3_rev;
    std::unique_ptr<Permutation> perm0;
    std::unique_ptr<ScaledPermutation> scale_perm2;
    std::unique_ptr<ScaledPermutation> scale_perm3;
    std::unique_ptr<ScaledPermutation> scale_perm3_rev;
    std::unique_ptr<ScaledPermutation> scale_perm0;
};

TYPED_TEST_SUITE(MultiVectorWithIndexType, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(MultiVector, ConvertsEmptyToPrecision)
{
    using MultiVector = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherMultiVector = typename gko::matrix::MultiVector<OtherT>;
    auto empty = OtherMultiVector::create(this->exec);
    auto res = MultiVector::create(this->exec);

    empty->convert_to(res);

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(MultiVector, MovesEmptyToPrecision)
{
    using MultiVector = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherMultiVector = typename gko::matrix::MultiVector<OtherT>;
    auto empty = OtherMultiVector::create(this->exec);
    auto res = MultiVector::create(this->exec);

    empty->move_to(res);

    ASSERT_FALSE(res->get_size());
}


template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::MultiVector<ValueType>> ref_permute(
    gko::matrix::MultiVector<ValueType>* input,
    gko::matrix::Permutation<IndexType>* permutation,
    gko::matrix::permute_mode mode)
{
    using gko::matrix::permute_mode;
    auto result = input->clone();
    auto permutation_multivector =
        gko::matrix::MultiVector<double>::create(input->get_executor());
    gko::matrix_data<double, IndexType> permutation_data;
    if ((mode & permute_mode::inverse) == permute_mode::inverse) {
        permutation->compute_inverse()->write(permutation_data);
    } else {
        permutation->write(permutation_data);
    }
    permutation_multivector->read(permutation_data);
    if ((mode & permute_mode::rows) == permute_mode::rows) {
        // compute P * A
        permutation_multivector->apply(input, result);
    }
    if ((mode & permute_mode::columns) == permute_mode::columns) {
        // compute A * P^T = (P * A^T)^T
        auto tmp = gko::share(result->transpose());
        auto tmp2 = gko::as<gko::matrix::MultiVector<ValueType>>(
            gko::as<gko::Cloneable>(tmp)->clone());
        permutation_multivector->apply(tmp, tmp2);
        tmp2->transpose(result);
    }
    return result;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::MultiVector<ValueType>> ref_permute(
    gko::matrix::MultiVector<ValueType>* input,
    gko::matrix::Permutation<IndexType>* row_permutation,
    gko::matrix::Permutation<IndexType>* col_permutation, bool invert)
{
    using gko::matrix::permute_mode;
    auto result = input->clone();
    auto row_permutation_multivector =
        gko::matrix::MultiVector<double>::create(input->get_executor());
    auto col_permutation_multivector =
        gko::matrix::MultiVector<double>::create(input->get_executor());
    gko::matrix_data<double, IndexType> row_permutation_data;
    gko::matrix_data<double, IndexType> col_permutation_data;
    if (invert) {
        row_permutation->compute_inverse()->write(row_permutation_data);
        col_permutation->compute_inverse()->write(col_permutation_data);
    } else {
        row_permutation->write(row_permutation_data);
        col_permutation->write(col_permutation_data);
    }
    row_permutation_multivector->read(row_permutation_data);
    col_permutation_multivector->read(col_permutation_data);
    row_permutation_multivector->apply(input, result);
    auto tmp = gko::share(result->transpose());
    auto tmp2 = gko::as<gko::matrix::MultiVector<ValueType>>(
        gko::as<gko::Cloneable>(tmp)->clone());
    col_permutation_multivector->apply(tmp, tmp2);
    tmp2->transpose(result);
    return result;
}


TYPED_TEST(MultiVectorWithIndexType, Permute)
{
    using gko::matrix::permute_mode;

    for (auto mode :
         {permute_mode::none, permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);

        auto permuted = this->mtx5->permute(this->perm3, mode);
        auto ref_permuted =
            ref_permute(this->mtx5.get(), this->perm3.get(), mode);

        GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
    }
}


TYPED_TEST(MultiVectorWithIndexType, PermuteRoundtrip)
{
    using gko::matrix::permute_mode;

    for (auto mode :
         {permute_mode::rows, permute_mode::columns, permute_mode::symmetric}) {
        SCOPED_TRACE(mode);

        auto permuted =
            this->mtx5->permute(this->perm3, mode)
                ->permute(this->perm3, mode | permute_mode::inverse);

        GKO_ASSERT_MTX_NEAR(this->mtx5, permuted, 0.0);
    }
}


TYPED_TEST(MultiVectorWithIndexType, PermuteStridedIntoMultiVector)
{
    using gko::matrix::permute_mode;
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec, this->mtx5->get_size(),
                           this->mtx5->get_size()[1] + 1);
    mtx->copy_from(this->mtx5);

    for (auto mode :
         {permute_mode::none, permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse,
          permute_mode::inverse_rows, permute_mode::inverse_columns,
          permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);
        auto permuted = Mtx::create(this->exec, this->mtx5->get_size(),
                                    this->mtx5->get_size()[1] + 2);

        this->mtx5->permute(this->perm3, permuted, mode);
        auto ref_permuted =
            ref_permute(this->mtx5.get(), this->perm3.get(), mode);

        GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
    }
}


TYPED_TEST(MultiVectorWithIndexType, PermuteRectangular)
{
    using gko::matrix::permute_mode;

    auto rpermuted = this->mtx1->permute(this->perm2, permute_mode::rows);
    auto irpermuted =
        this->mtx1->permute(this->perm2, permute_mode::inverse_rows);
    auto cpermuted = this->mtx1->permute(this->perm3, permute_mode::columns);
    auto icpermuted =
        this->mtx1->permute(this->perm3, permute_mode::inverse_columns);
    auto ref_rpermuted =
        ref_permute(this->mtx1.get(), this->perm2.get(), permute_mode::rows);
    auto ref_irpermuted = ref_permute(this->mtx1.get(), this->perm2.get(),
                                      permute_mode::inverse_rows);
    auto ref_cpermuted =
        ref_permute(this->mtx1.get(), this->perm3.get(), permute_mode::columns);
    auto ref_icpermuted = ref_permute(this->mtx1.get(), this->perm3.get(),
                                      permute_mode::inverse_columns);

    GKO_ASSERT_MTX_NEAR(rpermuted, ref_rpermuted, 0.0);
    GKO_ASSERT_MTX_NEAR(irpermuted, ref_irpermuted, 0.0);
    GKO_ASSERT_MTX_NEAR(cpermuted, ref_cpermuted, 0.0);
    GKO_ASSERT_MTX_NEAR(icpermuted, ref_icpermuted, 0.0);
}


TYPED_TEST(MultiVectorWithIndexType, PermuteFailsWithIncorrectPermutationSize)
{
    using gko::matrix::permute_mode;

    for (auto mode :
         {/* no permute_mode::none */ permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);

        ASSERT_THROW(this->mtx5->permute(this->perm0, mode),
                     gko::DimensionMismatch);
    }
}


TYPED_TEST(MultiVectorWithIndexType, PermuteFailsWithIncorrectOutputSize)
{
    using gko::matrix::permute_mode;
    using Mtx = typename TestFixture::Mtx;
    auto output = Mtx::create(this->exec);

    for (auto mode :
         {permute_mode::none, permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);

        ASSERT_THROW(this->mtx5->permute(this->perm3, output, mode),
                     gko::DimensionMismatch);
    }
}


TYPED_TEST(MultiVectorWithIndexType, NonsymmPermute)
{
    auto permuted = this->mtx5->permute(this->perm3, this->perm3_rev);
    auto ref_permuted = ref_permute(this->mtx5.get(), this->perm3.get(),
                                    this->perm3_rev.get(), false);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(MultiVectorWithIndexType, NonsymmPermuteInverse)
{
    auto permuted = this->mtx5->permute(this->perm3, this->perm3_rev, true);
    auto ref_permuted = ref_permute(this->mtx5.get(), this->perm3.get(),
                                    this->perm3_rev.get(), true);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(MultiVectorWithIndexType, NonsymmPermuteRectangular)
{
    auto permuted = this->mtx1->permute(this->perm2, this->perm3);
    auto ref_permuted = ref_permute(this->mtx1.get(), this->perm2.get(),
                                    this->perm3.get(), false);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(MultiVectorWithIndexType, NonsymmPermuteInverseRectangular)
{
    auto permuted = this->mtx1->permute(this->perm2, this->perm3, true);
    auto ref_permuted = ref_permute(this->mtx1.get(), this->perm2.get(),
                                    this->perm3.get(), true);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(MultiVectorWithIndexType, NonsymmPermuteRoundtrip)
{
    auto permuted = this->mtx5->permute(this->perm3, this->perm3_rev)
                        ->permute(this->perm3, this->perm3_rev, true);

    GKO_ASSERT_MTX_NEAR(this->mtx5, permuted, 0.0);
}


TYPED_TEST(MultiVectorWithIndexType, NonsymmPermuteInverseInverted)
{
    auto inv_permuted = this->mtx5->permute(this->perm3, this->perm3_rev, true);
    auto preinv_permuted = this->mtx5->permute(this->perm3_rev, this->perm3);

    GKO_ASSERT_MTX_NEAR(inv_permuted, preinv_permuted, 0.0);
}


TYPED_TEST(MultiVectorWithIndexType, NonsymmPermuteStridedIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec, this->mtx5->get_size(),
                           this->mtx5->get_size()[1] + 1);
    auto permuted = Mtx::create(this->exec, this->mtx5->get_size(),
                                this->mtx5->get_size()[1] + 2);
    mtx->copy_from(this->mtx5);

    mtx->permute(this->perm3, this->perm3_rev, permuted);
    auto ref_permuted = ref_permute(this->mtx5.get(), this->perm3.get(),
                                    this->perm3_rev.get(), false);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(MultiVectorWithIndexType,
           NonsymmPermuteInverseStridedIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec, this->mtx5->get_size(),
                           this->mtx5->get_size()[1] + 1);
    auto permuted = Mtx::create(this->exec, this->mtx5->get_size(),
                                this->mtx5->get_size()[1] + 2);
    mtx->copy_from(this->mtx5);

    mtx->permute(this->perm3, this->perm3_rev, permuted, true);
    auto ref_permuted = ref_permute(this->mtx5.get(), this->perm3.get(),
                                    this->perm3_rev.get(), true);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(MultiVectorWithIndexType,
           NonsymmPermuteFailsWithIncorrectPermutationSize)
{
    ASSERT_THROW(this->mtx5->permute(this->perm0, this->perm3_rev),
                 gko::DimensionMismatch);
    ASSERT_THROW(this->mtx5->permute(this->perm3_rev, this->perm0),
                 gko::DimensionMismatch);
    ASSERT_THROW(this->mtx5->permute(this->perm0, this->perm0),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVectorWithIndexType, SquareMatrixCanGatherRows)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 0}};

    auto row_collection = this->mtx5->row_gather(&permute_idxs);

    GKO_ASSERT_MTX_NEAR(row_collection,
                        l<value_type>({{-2.0, 2.0, 4.5}, {1.0, -1.0, -0.5}}),
                        0.0);
}


TYPED_TEST(MultiVectorWithIndexType, SquareMatrixCanGatherRowsIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 0}};
    auto row_collection = Mtx::create(exec, gko::dim<2>{2, 3});

    this->mtx5->row_gather(&permute_idxs, row_collection);

    GKO_ASSERT_MTX_NEAR(row_collection,
                        l<value_type>({{-2.0, 2.0, 4.5}, {1.0, -1.0, -0.5}}),
                        0.0);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareSubmatrixCanGatherRowsIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 0}};
    auto row_collection = Mtx::create(exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {1, 3})
        ->row_gather(&permute_idxs, row_collection);

    GKO_ASSERT_MTX_NEAR(row_collection,
                        l<value_type>({{2.0, 4.5}, {-1.0, -0.5}}), 0.0);
    ASSERT_EQ(row_collection->get_stride(), 4);
}


TYPED_TEST(MultiVectorWithIndexType,
           NonSquareSubmatrixCanGatherRowsIntoMixedMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using MixedMtx = typename TestFixture::MixedMtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx4->get_executor();
    gko::array<index_type> gather_index{exec, {1, 0, 1}};
    auto row_collection = MixedMtx::create(exec, gko::dim<2>{3, 3}, 4);

    this->mtx4->row_gather(&gather_index, row_collection);

    GKO_ASSERT_MTX_NEAR(
        row_collection,
        l<typename MixedMtx::value_type>(
            {{0.0, 5.0, 0.0}, {1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}),
        0.0);
}


TYPED_TEST(MultiVectorWithIndexType,
           NonSquareSubmatrixCanAdvancedGatherRowsIntoMixedMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using MixedMtx = typename TestFixture::MixedMtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx4->get_executor();
    gko::array<index_type> gather_index{exec, {1, 0, 1}};
    auto row_collection = gko::initialize<MixedMtx>(
        {{1.0, 0.5, -1.0}, {-1.5, 0.5, 1.0}, {2.0, -3.0, 1.0}}, exec);
    auto alpha = gko::initialize<MixedMtx>({1.0}, exec);
    auto beta = gko::initialize<Mtx>({2.0}, exec);

    this->mtx4->row_gather(alpha, &gather_index, beta, row_collection);

    GKO_ASSERT_MTX_NEAR(
        row_collection,
        l<typename MixedMtx::value_type>(
            {{2.0, 6.0, -2.0}, {-2.0, 4.0, 4.0}, {4.0, -1.0, 2.0}}),
        0.0);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareMatrixGatherRowsIntoMultiVectorFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 0}};

    ASSERT_THROW(this->mtx5->row_gather(&permute_idxs, Mtx::create(exec)),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVectorWithIndexType, SquareMatrixIsPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    auto ref_permuted =
        gko::as<Mtx>(gko::as<Mtx>(this->mtx5->row_permute(&permute_idxs))
                         ->column_permute(&permute_idxs));
    auto permuted = gko::as<Mtx>(this->mtx5->permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(MultiVectorWithIndexType, SquareMatrixIsPermutableIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    auto ref_permuted =
        gko::as<Mtx>(gko::as<Mtx>(this->mtx5->row_permute(&permute_idxs))
                         ->column_permute(&permute_idxs));
    this->mtx5->permute(&permute_idxs, permuted);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(MultiVectorWithIndexType, SquareSubmatrixIsPermutableIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 0}};
    auto permuted = Mtx::create(exec, gko::dim<2>{2, 2}, 4);
    auto mtx = this->mtx5->create_submatrix({0, 2}, {1, 3});

    auto ref_permuted =
        gko::as<Mtx>(gko::as<Mtx>(mtx->row_permute(&permute_idxs))
                         ->column_permute(&permute_idxs));
    mtx->permute(&permute_idxs, permuted);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
    ASSERT_EQ(permuted->get_stride(), 4);
}


TYPED_TEST(MultiVectorWithIndexType, NonSquareMatrixPermuteIntoMultiVectorFails)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx4->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(this->mtx4->permute(&permute_idxs, this->mtx4->clone()),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareMatrixPermuteIntoMultiVectorFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2}};

    ASSERT_THROW(this->mtx5->permute(&permute_idxs, this->mtx5->clone()),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareMatrixPermuteIntoMultiVectorFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(this->mtx5->permute(&permute_idxs, Mtx::create(exec)),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVectorWithIndexType, SquareMatrixIsInversePermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    auto ref_permuted = gko::as<Mtx>(
        gko::as<Mtx>(this->mtx5->inverse_row_permute(&permute_idxs))
            ->inverse_column_permute(&permute_idxs));
    auto permuted = gko::as<Mtx>(this->mtx5->inverse_permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareMatrixIsInversePermutableIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    auto ref_permuted = gko::as<Mtx>(
        gko::as<Mtx>(this->mtx5->inverse_row_permute(&permute_idxs))
            ->inverse_column_permute(&permute_idxs));
    this->mtx5->inverse_permute(&permute_idxs, permuted);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareSubmatrixIsInversePermutableIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 0}};
    auto permuted = Mtx::create(exec, gko::dim<2>{2, 2}, 4);
    auto mtx = this->mtx5->create_submatrix({0, 2}, {1, 3});

    auto ref_permuted =
        gko::as<Mtx>(gko::as<Mtx>(mtx->inverse_row_permute(&permute_idxs))
                         ->inverse_column_permute(&permute_idxs));
    mtx->inverse_permute(&permute_idxs, permuted);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
    ASSERT_EQ(permuted->get_stride(), 4);
}


TYPED_TEST(MultiVectorWithIndexType,
           NonSquareMatrixInversePermuteIntoMultiVectorFails)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx4->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(
        this->mtx4->inverse_permute(&permute_idxs, this->mtx4->clone()),
        gko::DimensionMismatch);
}


TYPED_TEST(
    MultiVectorWithIndexType,
    SquareMatrixInversePermuteIntoMultiVectorFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {0, 1}};

    ASSERT_THROW(
        this->mtx5->inverse_permute(&permute_idxs, this->mtx5->clone()),
        gko::DimensionMismatch);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareMatrixInversePermuteIntoMultiVectorFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(this->mtx5->inverse_permute(&permute_idxs, Mtx::create(exec)),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVectorWithIndexType, SquareMatrixIsRowPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    auto permuted = gko::as<Mtx>(this->mtx5->row_permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(
        permuted,
        l<value_type>({{-2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}, {1.0, -1.0, -0.5}}),
        0.0);
}


TYPED_TEST(MultiVectorWithIndexType, NonSquareMatrixIsRowPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx4->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 0}};

    auto permuted = gko::as<Mtx>(this->mtx4->row_permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(permuted,
                        l<value_type>({{0.0, 5.0, 0.0}, {1.0, 3.0, 2.0}}), 0.0);
}


TYPED_TEST(MultiVectorWithIndexType, SquareMatrixIsRowPermutableIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    this->mtx5->row_permute(&permute_idxs, permuted);

    GKO_ASSERT_MTX_NEAR(
        permuted,
        l<value_type>({{-2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}, {1.0, -1.0, -0.5}}),
        0.0);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareSubmatrixIsRowPermutableIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 0}};
    auto permuted = Mtx::create(exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {0, 2})
        ->row_permute(&permute_idxs, permuted);

    GKO_ASSERT_MTX_NEAR(permuted, l<value_type>({{-2.0, 2.0}, {1.0, -1.0}}),
                        0.0);
    ASSERT_EQ(permuted->get_stride(), 4);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareMatrixRowPermuteIntoMultiVectorFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    ASSERT_THROW(this->mtx5->row_permute(&permute_idxs, permuted),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareMatrixRowPermuteIntoMultiVectorFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(this->mtx5->row_permute(&permute_idxs, Mtx::create(exec)),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVectorWithIndexType, SquareMatrixIsColPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    auto permuted = gko::as<Mtx>(this->mtx5->column_permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(
        permuted,
        l<value_type>({{-1.0, -0.5, 1.0}, {2.0, 4.5, -2.0}, {3.4, 1.2, 2.1}}),
        0.0);
}


TYPED_TEST(MultiVectorWithIndexType, NonSquareMatrixIsColPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx4->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    auto permuted = gko::as<Mtx>(this->mtx4->column_permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(permuted,
                        l<value_type>({{3.0, 2.0, 1.0}, {5.0, 0.0, 0.0}}), 0.0);
}


TYPED_TEST(MultiVectorWithIndexType, SquareMatrixIsColPermutableIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    this->mtx5->column_permute(&permute_idxs, permuted);

    GKO_ASSERT_MTX_NEAR(
        permuted,
        l<value_type>({{-1.0, -0.5, 1.0}, {2.0, 4.5, -2.0}, {3.4, 1.2, 2.1}}),
        0.0);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareSubmatrixIsColPermutableIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 0}};
    auto permuted = Mtx::create(exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {0, 2})
        ->column_permute(&permute_idxs, permuted);

    GKO_ASSERT_MTX_NEAR(permuted, l<value_type>({{-1.0, 1.0}, {2.0, -2.0}}),
                        0.0);
    ASSERT_EQ(permuted->get_stride(), 4);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareMatrixColPermuteIntoMultiVectorFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    ASSERT_THROW(this->mtx5->column_permute(&permute_idxs, permuted),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareMatrixColPermuteIntoMultiVectorFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(this->mtx5->column_permute(&permute_idxs, Mtx::create(exec)),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVectorWithIndexType, SquareMatrixIsInverseRowPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> inverse_permute_idxs{exec, {1, 2, 0}};

    auto permuted =
        gko::as<Mtx>(this->mtx5->inverse_row_permute(&inverse_permute_idxs));

    GKO_ASSERT_MTX_NEAR(
        permuted,
        l<value_type>({{2.1, 3.4, 1.2}, {1.0, -1.0, -0.5}, {-2.0, 2.0, 4.5}}),
        0.0);
}


TYPED_TEST(MultiVectorWithIndexType, NonSquareMatrixIsInverseRowPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx4->get_executor();
    gko::array<index_type> inverse_permute_idxs{exec, {1, 0}};

    auto permuted =
        gko::as<Mtx>(this->mtx4->inverse_row_permute(&inverse_permute_idxs));

    GKO_ASSERT_MTX_NEAR(permuted,
                        l<value_type>({{0.0, 5.0, 0.0}, {1.0, 3.0, 2.0}}), 0.0);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareMatrixIsInverseRowPermutableIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    this->mtx5->inverse_row_permute(&permute_idxs, permuted);

    GKO_ASSERT_MTX_NEAR(
        permuted,
        l<value_type>({{2.1, 3.4, 1.2}, {1.0, -1.0, -0.5}, {-2.0, 2.0, 4.5}}),
        0.0);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareSubmatrixIsInverseRowPermutableIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 0}};
    auto permuted = Mtx::create(exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {0, 2})
        ->inverse_row_permute(&permute_idxs, permuted);

    GKO_ASSERT_MTX_NEAR(permuted, l<value_type>({{-2.0, 2.0}, {1.0, -1.0}}),
                        0.0);
    ASSERT_EQ(permuted->get_stride(), 4);
}


TYPED_TEST(
    MultiVectorWithIndexType,
    SquareMatrixInverseRowPermuteIntoMultiVectorFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    ASSERT_THROW(this->mtx5->inverse_row_permute(&permute_idxs, permuted),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareMatrixInverseRowPermuteIntoMultiVectorFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(
        this->mtx5->inverse_row_permute(&permute_idxs, Mtx::create(exec)),
        gko::DimensionMismatch);
}


TYPED_TEST(MultiVectorWithIndexType, SquareMatrixIsInverseColPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> inverse_permute_idxs{exec, {1, 2, 0}};

    auto permuted =
        gko::as<Mtx>(this->mtx5->inverse_column_permute(&inverse_permute_idxs));

    GKO_ASSERT_MTX_NEAR(
        permuted,
        l<value_type>({{-0.5, 1.0, -1.0}, {4.5, -2.0, 2.0}, {1.2, 2.1, 3.4}}),
        0.0);
}


TYPED_TEST(MultiVectorWithIndexType, NonSquareMatrixIsInverseColPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx4->get_executor();
    gko::array<index_type> inverse_permute_idxs{exec, {1, 2, 0}};

    auto permuted =
        gko::as<Mtx>(this->mtx4->inverse_column_permute(&inverse_permute_idxs));

    GKO_ASSERT_MTX_NEAR(permuted,
                        l<value_type>({{2.0, 1.0, 3.0}, {0.0, 0.0, 5.0}}), 0.0);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareMatrixIsInverseColPermutableIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    this->mtx5->inverse_column_permute(&permute_idxs, permuted);

    GKO_ASSERT_MTX_NEAR(
        permuted,
        l<value_type>({{-0.5, 1.0, -1.0}, {4.5, -2.0, 2.0}, {1.2, 2.1, 3.4}}),
        0.0);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareSubmatrixIsInverseColPermutableIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 0}};
    auto permuted = Mtx::create(exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {0, 2})
        ->column_permute(&permute_idxs, permuted);

    GKO_ASSERT_MTX_NEAR(permuted, l<value_type>({{-1.0, 1.0}, {2.0, -2.0}}),
                        0.0);
    ASSERT_EQ(permuted->get_stride(), 4);
}


TYPED_TEST(
    MultiVectorWithIndexType,
    SquareMatrixInverseColPermuteIntoMultiVectorFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    ASSERT_THROW(this->mtx5->inverse_column_permute(&permute_idxs, permuted),
                 gko::DimensionMismatch);
}


TYPED_TEST(MultiVectorWithIndexType,
           SquareMatrixInverseColPermuteIntoMultiVectorFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(
        this->mtx5->inverse_column_permute(&permute_idxs, Mtx::create(exec)),
        gko::DimensionMismatch);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::MultiVector<ValueType>> ref_scaled_permute(
    gko::matrix::MultiVector<ValueType>* input,
    gko::matrix::ScaledPermutation<ValueType, IndexType>* permutation,
    gko::matrix::permute_mode mode)
{
    using gko::matrix::permute_mode;
    auto result = input->clone();
    auto permutation_multivector =
        gko::matrix::MultiVector<ValueType>::create(input->get_executor());
    gko::matrix_data<ValueType, IndexType> permutation_data;
    if ((mode & permute_mode::inverse) == permute_mode::inverse) {
        permutation->compute_inverse()->write(permutation_data);
    } else {
        permutation->write(permutation_data);
    }
    permutation_multivector->read(permutation_data);
    if ((mode & permute_mode::rows) == permute_mode::rows) {
        // compute P * A
        permutation_multivector->apply(input, result);
    }
    if ((mode & permute_mode::columns) == permute_mode::columns) {
        // compute A * P^T = (P * A^T)^T
        auto tmp = share(result->transpose());
        auto tmp2 = gko::as<gko::matrix::MultiVector<ValueType>>(
            gko::as<gko::Cloneable>(tmp)->clone());
        permutation_multivector->apply(tmp, tmp2);
        tmp2->transpose(result);
    }
    return result;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::MultiVector<ValueType>> ref_scaled_permute(
    gko::matrix::MultiVector<ValueType>* input,
    gko::matrix::ScaledPermutation<ValueType, IndexType>* row_permutation,
    gko::matrix::ScaledPermutation<ValueType, IndexType>* col_permutation,
    bool invert)
{
    using gko::matrix::permute_mode;
    auto result = input->clone();
    auto row_permutation_multivector =
        gko::matrix::MultiVector<ValueType>::create(input->get_executor());
    auto col_permutation_multivector =
        gko::matrix::MultiVector<ValueType>::create(input->get_executor());
    gko::matrix_data<ValueType, IndexType> row_permutation_data;
    gko::matrix_data<ValueType, IndexType> col_permutation_data;
    if (invert) {
        row_permutation->compute_inverse()->write(row_permutation_data);
        col_permutation->compute_inverse()->write(col_permutation_data);
    } else {
        row_permutation->write(row_permutation_data);
        col_permutation->write(col_permutation_data);
    }
    row_permutation_multivector->read(row_permutation_data);
    col_permutation_multivector->read(col_permutation_data);
    row_permutation_multivector->apply(input, result);
    auto tmp = gko::share(result->transpose());
    auto tmp2 = gko::as<gko::matrix::MultiVector<ValueType>>(
        gko::as<gko::Cloneable>(tmp)->clone());
    col_permutation_multivector->apply(tmp, tmp2);
    tmp2->transpose(result);
    return result;
}


TYPED_TEST(MultiVectorWithIndexType, ScaledPermute)
{
    using gko::matrix::permute_mode;
    using value_type = typename TestFixture::value_type;

    for (auto mode :
         {permute_mode::none, permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);

        auto permuted = this->mtx5->scale_permute(this->scale_perm3, mode);
        auto ref_permuted =
            ref_scaled_permute(this->mtx5.get(), this->scale_perm3.get(), mode);

        GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
    }
}


TYPED_TEST(MultiVectorWithIndexType, ScaledPermuteRoundtrip)
{
    using gko::matrix::permute_mode;
    using value_type = typename TestFixture::value_type;

    for (auto mode :
         {permute_mode::rows, permute_mode::columns, permute_mode::symmetric}) {
        SCOPED_TRACE(mode);

        auto permuted = this->mtx5->scale_permute(this->scale_perm3, mode)
                            ->scale_permute(this->scale_perm3,
                                            mode | permute_mode::inverse);

        GKO_ASSERT_MTX_NEAR(this->mtx5, permuted, r<value_type>::value);
    }
}


TYPED_TEST(MultiVectorWithIndexType, ScaledPermuteStridedIntoMultiVector)
{
    using gko::matrix::permute_mode;
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec, this->mtx5->get_size(),
                           this->mtx5->get_size()[1] + 1);
    mtx->copy_from(this->mtx5);

    for (auto mode :
         {permute_mode::none, permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse,
          permute_mode::inverse_rows, permute_mode::inverse_columns,
          permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);
        auto permuted = Mtx::create(this->exec, this->mtx5->get_size(),
                                    this->mtx5->get_size()[1] + 2);

        this->mtx5->scale_permute(this->scale_perm3, permuted, mode);
        auto ref_permuted =
            ref_scaled_permute(this->mtx5.get(), this->scale_perm3.get(), mode);

        GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
    }
}


TYPED_TEST(MultiVectorWithIndexType, ScaledPermuteRectangular)
{
    using gko::matrix::permute_mode;
    using value_type = typename TestFixture::value_type;

    auto rpermuted =
        this->mtx1->scale_permute(this->scale_perm2, permute_mode::rows);
    auto irpermuted = this->mtx1->scale_permute(this->scale_perm2,
                                                permute_mode::inverse_rows);
    auto cpermuted =
        this->mtx1->scale_permute(this->scale_perm3, permute_mode::columns);
    auto icpermuted = this->mtx1->scale_permute(this->scale_perm3,
                                                permute_mode::inverse_columns);
    auto ref_rpermuted = ref_scaled_permute(
        this->mtx1.get(), this->scale_perm2.get(), permute_mode::rows);
    auto ref_irpermuted = ref_scaled_permute(
        this->mtx1.get(), this->scale_perm2.get(), permute_mode::inverse_rows);
    auto ref_cpermuted = ref_scaled_permute(
        this->mtx1.get(), this->scale_perm3.get(), permute_mode::columns);
    auto ref_icpermuted =
        ref_scaled_permute(this->mtx1.get(), this->scale_perm3.get(),
                           permute_mode::inverse_columns);

    GKO_ASSERT_MTX_NEAR(rpermuted, ref_rpermuted, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(irpermuted, ref_irpermuted, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(cpermuted, ref_cpermuted, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(icpermuted, ref_icpermuted, r<value_type>::value);
}


TYPED_TEST(MultiVectorWithIndexType,
           ScaledPermuteFailsWithIncorrectPermutationSize)
{
    using gko::matrix::permute_mode;

    for (auto mode :
         {/* no permute_mode::none */ permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);

        ASSERT_THROW(this->mtx5->scale_permute(this->scale_perm0, mode),
                     gko::DimensionMismatch);
    }
}


TYPED_TEST(MultiVectorWithIndexType, ScaledPermuteFailsWithIncorrectOutputSize)
{
    using gko::matrix::permute_mode;
    using Mtx = typename TestFixture::Mtx;
    auto output = Mtx::create(this->exec);

    for (auto mode :
         {permute_mode::none, permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);

        ASSERT_THROW(this->mtx5->scale_permute(this->scale_perm3, output, mode),
                     gko::DimensionMismatch);
    }
}


TYPED_TEST(MultiVectorWithIndexType, NonsymmScaledPermute)
{
    using value_type = typename TestFixture::value_type;

    auto permuted =
        this->mtx5->scale_permute(this->scale_perm3, this->scale_perm3_rev);
    auto ref_permuted =
        ref_scaled_permute(this->mtx5.get(), this->scale_perm3.get(),
                           this->scale_perm3_rev.get(), false);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
}


TYPED_TEST(MultiVectorWithIndexType, NonsymmScaledPermuteInverse)
{
    using value_type = typename TestFixture::value_type;

    auto permuted = this->mtx5->scale_permute(this->scale_perm3,
                                              this->scale_perm3_rev, true);
    auto ref_permuted =
        ref_scaled_permute(this->mtx5.get(), this->scale_perm3.get(),
                           this->scale_perm3_rev.get(), true);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
}


TYPED_TEST(MultiVectorWithIndexType, NonsymmScaledPermuteRectangular)
{
    using value_type = typename TestFixture::value_type;

    auto permuted =
        this->mtx1->scale_permute(this->scale_perm2, this->scale_perm3);
    auto ref_permuted =
        ref_scaled_permute(this->mtx1.get(), this->scale_perm2.get(),
                           this->scale_perm3.get(), false);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
}


TYPED_TEST(MultiVectorWithIndexType, NonsymmScaledPermuteInverseRectangular)
{
    using value_type = typename TestFixture::value_type;

    auto permuted =
        this->mtx1->scale_permute(this->scale_perm2, this->scale_perm3, true);
    auto ref_permuted =
        ref_scaled_permute(this->mtx1.get(), this->scale_perm2.get(),
                           this->scale_perm3.get(), true);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
}


TYPED_TEST(MultiVectorWithIndexType, NonsymmScaledPermuteRoundtrip)
{
    using value_type = typename TestFixture::value_type;

    auto permuted =
        this->mtx5->scale_permute(this->scale_perm3, this->scale_perm3_rev)
            ->scale_permute(this->scale_perm3, this->scale_perm3_rev, true);

    GKO_ASSERT_MTX_NEAR(this->mtx5, permuted, r<value_type>::value);
}


TYPED_TEST(MultiVectorWithIndexType, NonsymmScaledPermuteInverseInverted)
{
    using value_type = typename TestFixture::value_type;

    auto inv_permuted = this->mtx5->scale_permute(this->scale_perm3,
                                                  this->scale_perm3_rev, true);
    auto preinv_permuted =
        this->mtx5->scale_permute(this->scale_perm3->compute_inverse(),
                                  this->scale_perm3_rev->compute_inverse());

    GKO_ASSERT_MTX_NEAR(inv_permuted, preinv_permuted, r<value_type>::value);
}

TYPED_TEST(MultiVectorWithIndexType, NonsymmScaledPermuteStridedIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto mtx = Mtx::create(this->exec, this->mtx5->get_size(),
                           this->mtx5->get_size()[1] + 1);
    auto permuted = Mtx::create(this->exec, this->mtx5->get_size(),
                                this->mtx5->get_size()[1] + 2);
    mtx->copy_from(this->mtx5);

    mtx->scale_permute(this->scale_perm3, this->scale_perm3_rev, permuted);
    auto ref_permuted =
        ref_scaled_permute(this->mtx5.get(), this->scale_perm3.get(),
                           this->scale_perm3_rev.get(), false);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
}


TYPED_TEST(MultiVectorWithIndexType,
           NonsymmScaledPermuteInverseStridedIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto mtx = Mtx::create(this->exec, this->mtx5->get_size(),
                           this->mtx5->get_size()[1] + 1);
    auto permuted = Mtx::create(this->exec, this->mtx5->get_size(),
                                this->mtx5->get_size()[1] + 2);
    mtx->copy_from(this->mtx5);

    mtx->scale_permute(this->scale_perm3, this->scale_perm3_rev, permuted,
                       true);
    auto ref_permuted =
        ref_scaled_permute(this->mtx5.get(), this->scale_perm3.get(),
                           this->scale_perm3_rev.get(), true);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
}


TYPED_TEST(MultiVectorWithIndexType,
           NonsymmScaledPermuteFailsWithIncorrectOutputSize)
{
    ASSERT_THROW(
        this->mtx5->scale_permute(this->scale_perm3, this->scale_perm3,
                                  TestFixture::Mtx::create(this->exec)),
        gko::DimensionMismatch);
}


TYPED_TEST(MultiVectorWithIndexType,
           NonsymmScaledPermuteFailsWithIncorrectPermutationSize)
{
    ASSERT_THROW(
        this->mtx5->scale_permute(this->scale_perm0, this->scale_perm3_rev),
        gko::DimensionMismatch);
    ASSERT_THROW(
        this->mtx5->scale_permute(this->scale_perm3_rev, this->scale_perm0),
        gko::DimensionMismatch);
    ASSERT_THROW(
        this->mtx5->scale_permute(this->scale_perm0, this->scale_perm0),
        gko::DimensionMismatch);
}


template <typename T>
class MultiVectorComplex : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::MultiVector<value_type>;
    using RealMtx = gko::matrix::MultiVector<gko::remove_complex<value_type>>;
};


TYPED_TEST_SUITE(MultiVectorComplex, gko::test::ComplexValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(MultiVectorComplex, ScalesWithRealScalar)
{
    using MultiVector = typename TestFixture::Mtx;
    using RealMultiVector = gko::remove_complex<MultiVector>;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<MultiVector>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                             {T{-2.0, 1.5}, T{4.5, 0.0}},
                                             {T{1.0, 0.0}, T{0.0, 1.0}}},
                                            exec);
    auto alpha =
        gko::initialize<RealMultiVector>({gko::remove_complex<T>{-2.0}}, exec);

    mtx->scale(alpha);

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-2.0, -4.0}, T{2.0, -4.5}},
                              {T{4.0, -3.0}, T{-9.0, 0.0}},
                              {T{-2.0, 0.0}, T{0.0, -2.0}}}),
                        0.0);
}


TYPED_TEST(MultiVectorComplex, ScalesWithRealVector)
{
    using MultiVector = typename TestFixture::Mtx;
    using RealMultiVector = gko::remove_complex<MultiVector>;
    using T = typename TestFixture::value_type;
    using RealT = gko::remove_complex<T>;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<MultiVector>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                             {T{-2.0, 1.5}, T{4.5, 0.0}},
                                             {T{1.0, 0.0}, T{0.0, 1.0}}},
                                            exec);
    auto alpha =
        gko::initialize<RealMultiVector>({{RealT{-2.0}, RealT{4.0}}}, exec);

    mtx->scale(alpha);

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-2.0, -4.0}, T{-4.0, 9.0}},
                              {T{4.0, -3.0}, T{18.0, 0.0}},
                              {T{-2.0, 0.0}, T{0.0, 4.0}}}),
                        0.0);
}


TYPED_TEST(MultiVectorComplex, InvScalesWithRealScalar)
{
    using MultiVector = typename TestFixture::Mtx;
    using RealMultiVector = gko::remove_complex<MultiVector>;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<MultiVector>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                             {T{-2.0, 1.5}, T{4.5, 0.0}},
                                             {T{1.0, 0.0}, T{0.0, 1.0}}},
                                            exec);
    auto alpha =
        gko::initialize<RealMultiVector>({gko::remove_complex<T>{-0.5}}, exec);

    mtx->inv_scale(alpha);

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-2.0, -4.0}, T{2.0, -4.5}},
                              {T{4.0, -3.0}, T{-9.0, 0.0}},
                              {T{-2.0, 0.0}, T{0.0, -2.0}}}),
                        0.0);
}


TYPED_TEST(MultiVectorComplex, InvScalesWithRealVector)
{
    using MultiVector = typename TestFixture::Mtx;
    using RealMultiVector = gko::remove_complex<MultiVector>;
    using T = typename TestFixture::value_type;
    using RealT = gko::remove_complex<T>;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<MultiVector>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                             {T{-2.0, 1.5}, T{4.5, 0.0}},
                                             {T{1.0, 0.0}, T{0.0, 1.0}}},
                                            exec);
    auto alpha =
        gko::initialize<RealMultiVector>({{RealT{-0.5}, RealT{0.25}}}, exec);

    mtx->inv_scale(alpha);

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-2.0, -4.0}, T{-4.0, 9.0}},
                              {T{4.0, -3.0}, T{18.0, 0.0}},
                              {T{-2.0, 0.0}, T{0.0, 4.0}}}),
                        0.0);
}


TYPED_TEST(MultiVectorComplex, AddsScaledWithRealScalar)
{
    using MultiVector = typename TestFixture::Mtx;
    using RealMultiVector = gko::remove_complex<MultiVector>;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<MultiVector>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                             {T{-2.0, 1.5}, T{4.5, 0.0}},
                                             {T{1.0, 0.0}, T{0.0, 1.0}}},
                                            exec);
    auto mtx2 = gko::initialize<MultiVector>({{T{4.0, -1.0}, T{5.0, 1.5}},
                                              {T{3.0, 1.0}, T{0.0, 2.0}},
                                              {T{-1.0, 1.0}, T{0.5, -2.0}}},
                                             exec);
    auto alpha =
        gko::initialize<RealMultiVector>({gko::remove_complex<T>{-2.0}}, exec);

    mtx->add_scaled(alpha, mtx2);

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-7.0, 4.0}, T{-11.0, -0.75}},
                              {T{-8.0, -0.5}, T{4.5, -4.0}},
                              {T{3.0, -2.0}, T{-1.0, 5.0}}}),
                        0.0);
}


TYPED_TEST(MultiVectorComplex, AddsScaledWithRealVector)
{
    using MultiVector = typename TestFixture::Mtx;
    using RealMultiVector = gko::remove_complex<MultiVector>;
    using T = typename TestFixture::value_type;
    using RealT = gko::remove_complex<T>;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<MultiVector>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                             {T{-2.0, 1.5}, T{4.5, 0.0}},
                                             {T{1.0, 0.0}, T{0.0, 1.0}}},
                                            exec);
    auto mtx2 = gko::initialize<MultiVector>({{T{4.0, -1.0}, T{5.0, 1.5}},
                                              {T{3.0, 1.0}, T{0.0, 2.0}},
                                              {T{-1.0, 1.0}, T{0.5, -2.0}}},
                                             exec);
    auto alpha =
        gko::initialize<RealMultiVector>({{RealT{-2.0}, RealT{4.0}}}, exec);

    mtx->add_scaled(alpha, mtx2);

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-7.0, 4.0}, T{19.0, 8.25}},
                              {T{-8.0, -0.5}, T{4.5, 8.0}},
                              {T{3.0, -2.0}, T{2.0, -7.0}}}),
                        0.0);
}


TYPED_TEST(MultiVectorComplex, SubtractsScaledWithRealScalar)
{
    using MultiVector = typename TestFixture::Mtx;
    using RealMultiVector = gko::remove_complex<MultiVector>;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<MultiVector>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                             {T{-2.0, 1.5}, T{4.5, 0.0}},
                                             {T{1.0, 0.0}, T{0.0, 1.0}}},
                                            exec);
    auto mtx2 = gko::initialize<MultiVector>({{T{4.0, -1.0}, T{5.0, 1.5}},
                                              {T{3.0, 1.0}, T{0.0, 2.0}},
                                              {T{-1.0, 1.0}, T{0.5, -2.0}}},
                                             exec);
    auto alpha =
        gko::initialize<RealMultiVector>({gko::remove_complex<T>{2.0}}, exec);

    mtx->sub_scaled(alpha, mtx2);

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-7.0, 4.0}, T{-11.0, -0.75}},
                              {T{-8.0, -0.5}, T{4.5, -4.0}},
                              {T{3.0, -2.0}, T{-1.0, 5.0}}}),
                        0.0);
}


TYPED_TEST(MultiVectorComplex, SubtractsScaledWithRealVector)
{
    using MultiVector = typename TestFixture::Mtx;
    using RealMultiVector = gko::remove_complex<MultiVector>;
    using T = typename TestFixture::value_type;
    using RealT = gko::remove_complex<T>;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<MultiVector>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                             {T{-2.0, 1.5}, T{4.5, 0.0}},
                                             {T{1.0, 0.0}, T{0.0, 1.0}}},
                                            exec);
    auto mtx2 = gko::initialize<MultiVector>({{T{4.0, -1.0}, T{5.0, 1.5}},
                                              {T{3.0, 1.0}, T{0.0, 2.0}},
                                              {T{-1.0, 1.0}, T{0.5, -2.0}}},
                                             exec);
    auto alpha =
        gko::initialize<RealMultiVector>({{RealT{2.0}, RealT{-4.0}}}, exec);

    mtx->sub_scaled(alpha, mtx2);

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-7.0, 4.0}, T{19.0, 8.25}},
                              {T{-8.0, -0.5}, T{4.5, 8.0}},
                              {T{3.0, -2.0}, T{2.0, -7.0}}}),
                        0.0);
}


TYPED_TEST(MultiVectorComplex, NonSquareMatrixIsConjugateTransposable)
{
    using MultiVector = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<MultiVector>({{T{1.0, 2.0}, T{-1.0, 2.1}},
                                             {T{-2.0, 1.5}, T{4.5, 0.0}},
                                             {T{1.0, 0.0}, T{0.0, 1.0}}},
                                            exec);

    auto trans = gko::as<MultiVector>(mtx->conj_transpose());

    GKO_ASSERT_MTX_NEAR(trans,
                        l<T>({{T{1.0, -2.0}, T{-2.0, -1.5}, T{1.0, 0.0}},
                              {T{-1.0, -2.1}, T{4.5, 0.0}, T{0.0, -1.0}}}),
                        0.0);
}


TYPED_TEST(MultiVectorComplex,
           NonSquareMatrixIsConjugateTransposableIntoMultiVector)
{
    using MultiVector = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<MultiVector>({{T{1.0, 2.0}, T{-1.0, 2.1}},
                                             {T{-2.0, 1.5}, T{4.5, 0.0}},
                                             {T{1.0, 0.0}, T{0.0, 1.0}}},
                                            exec);
    auto trans = MultiVector::create(exec, gko::transpose(mtx->get_size()));

    mtx->conj_transpose(trans);

    GKO_ASSERT_MTX_NEAR(trans,
                        l<T>({{T{1.0, -2.0}, T{-2.0, -1.5}, T{1.0, 0.0}},
                              {T{-1.0, -2.1}, T{4.5, 0.0}, T{0.0, -1.0}}}),
                        0.0);
}


TYPED_TEST(MultiVectorComplex, InplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);

    mtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        mtx, l<T>({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}), 0.0);
}


TYPED_TEST(MultiVectorComplex, OutplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);

    auto abs_mtx = mtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(
        abs_mtx, l<T>({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}),
        0.0);
}


TYPED_TEST(MultiVectorComplex, OutplaceAbsoluteIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);
    auto abs_mtx = gko::remove_complex<Mtx>::create(exec, mtx->get_size());

    mtx->compute_absolute(abs_mtx);

    GKO_ASSERT_MTX_NEAR(
        abs_mtx, l<T>({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}),
        0.0);
}


TYPED_TEST(MultiVectorComplex, MakeComplex)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);

    auto complex_mtx = mtx->make_complex();

    GKO_ASSERT_MTX_NEAR(complex_mtx, mtx, 0.0);
}


TYPED_TEST(MultiVectorComplex, MakeComplexIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);

    auto complex_mtx = Mtx::create(exec, mtx->get_size());
    mtx->make_complex(complex_mtx);

    GKO_ASSERT_MTX_NEAR(complex_mtx, mtx, 0.0);
}


TYPED_TEST(MultiVectorComplex, GetReal)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);

    auto real_mtx = mtx->get_real();

    GKO_ASSERT_MTX_NEAR(
        real_mtx, l<T>({{1.0, 3.0, 0.0}, {-4.0, -1.0, 0.0}, {0.0, 0.0, 2.0}}),
        0.0);
}


TYPED_TEST(MultiVectorComplex, GetRealIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using RealMtx = typename TestFixture::RealMtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);

    auto real_mtx = RealMtx::create(exec, mtx->get_size());
    mtx->get_real(real_mtx);

    GKO_ASSERT_MTX_NEAR(
        real_mtx, l<T>({{1.0, 3.0, 0.0}, {-4.0, -1.0, 0.0}, {0.0, 0.0, 2.0}}),
        0.0);
}


TYPED_TEST(MultiVectorComplex, GetImag)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);

    auto imag_mtx = mtx->get_imag();

    GKO_ASSERT_MTX_NEAR(
        imag_mtx, l<T>({{0.0, 4.0, 2.0}, {-3.0, 0.0, 0.0}, {0.0, -1.5, 0.0}}),
        0.0);
}


TYPED_TEST(MultiVectorComplex, GetImagIntoMultiVector)
{
    using Mtx = typename TestFixture::Mtx;
    using RealMtx = typename TestFixture::RealMtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);

    auto imag_mtx = RealMtx::create(exec, mtx->get_size());
    mtx->get_imag(imag_mtx);

    GKO_ASSERT_MTX_NEAR(
        imag_mtx, l<T>({{0.0, 4.0, 2.0}, {-3.0, 0.0, 0.0}, {0.0, -1.5, 0.0}}),
        0.0);
}


TYPED_TEST(MultiVectorComplex, Dot)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto a =
        gko::initialize<Mtx>({T{1.0, 0.0}, T{3.0, 4.0}, T{1.0, 2.0}}, exec);
    auto b =
        gko::initialize<Mtx>({T{1.0, -2.0}, T{5.0, 0.0}, T{0.0, -3.0}}, exec);
    auto result = gko::initialize<Mtx>({T{0.0, 0.0}}, exec);

    a->compute_dot(b, result);

    GKO_ASSERT_MTX_NEAR(result, l({T{22.0, 15.0}}), 0.0);
}


TYPED_TEST(MultiVectorComplex, ConjDot)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto a =
        gko::initialize<Mtx>({T{1.0, 0.0}, T{3.0, 4.0}, T{1.0, 2.0}}, exec);
    auto b =
        gko::initialize<Mtx>({T{1.0, -2.0}, T{5.0, 0.0}, T{0.0, -3.0}}, exec);
    auto result = gko::initialize<Mtx>({T{0.0, 0.0}}, exec);

    a->compute_conj_dot(b, result);

    GKO_ASSERT_MTX_NEAR(result, l({T{10.0, -25.0}}), 0.0);
}


}  // namespace
