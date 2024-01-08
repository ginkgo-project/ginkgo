// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/dense.hpp>


#include <complex>
#include <memory>
#include <numeric>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/scaled_permutation.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename T>
class Dense : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using MixedMtx = gko::matrix::Dense<gko::next_precision<value_type>>;
    using ComplexMtx = gko::to_complex<Mtx>;
    using RealMtx = gko::remove_complex<Mtx>;
    Dense()
        : exec(gko::ReferenceExecutor::create()),
          mtx1(gko::initialize<Mtx>(4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}},
                                    exec)),
          mtx2(gko::initialize<Mtx>({I<T>({1.0, -1.0}), I<T>({-2.0, 2.0})},
                                    exec)),
          mtx3(gko::initialize<Mtx>(4, {{1.0, 2.0, 3.0}, {0.5, 1.5, 2.5}},
                                    exec)),
          mtx4(gko::initialize<Mtx>(4, {{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}},
                                    exec)),
          mtx5(gko::initialize<Mtx>(
              {{1.0, -1.0, -0.5}, {-2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}, exec)),
          mtx6(gko::initialize<Mtx>({{1.0, 2.0, 0.0}, {0.0, 1.5, 0.0}}, exec)),
          mtx7(gko::initialize<Mtx>({{1.0, 2.0, 3.0}, {0.0, 1.5, 0.0}}, exec)),
          mtx8(gko::initialize<Mtx>(
              {I<T>({1.0, -1.0}), I<T>({-2.0, 2.0}), I<T>({-3.0, 3.0})}, exec))
    {}

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
            std::normal_distribution<gko::remove_complex<value_type>>(0.0, 1.0),
            rand_engine, exec);
    }
};


TYPED_TEST_SUITE(Dense, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Dense, CopyRespectsStride)
{
    using value_type = typename TestFixture::value_type;
    auto m =
        gko::initialize<gko::matrix::Dense<TypeParam>>({1.0, 2.0}, this->exec);
    auto m2 =
        gko::matrix::Dense<TypeParam>::create(this->exec, gko::dim<2>{2, 1}, 2);
    auto original_data = m2->get_values();
    original_data[1] = TypeParam{3.0};

    m->convert_to(m2);

    EXPECT_EQ(m2->at(0, 0), value_type{1.0});
    EXPECT_EQ(m2->get_stride(), 2);
    EXPECT_EQ(m2->at(1, 0), value_type{2.0});
    EXPECT_EQ(m2->get_values(), original_data);
    EXPECT_EQ(original_data[1], TypeParam{3.0});
}


TYPED_TEST(Dense, TemporaryOutputCloneWorks)
{
    using value_type = typename TestFixture::value_type;
    auto other = gko::OmpExecutor::create();
    auto m = gko::initialize<gko::matrix::Dense<TypeParam>>({1.0, 2.0}, other);

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


TYPED_TEST(Dense, CanBeFilledWithValue)
{
    using value_type = typename TestFixture::value_type;
    auto m =
        gko::initialize<gko::matrix::Dense<TypeParam>>({1.0, 2.0}, this->exec);
    EXPECT_EQ(m->at(0), value_type{1});
    EXPECT_EQ(m->at(1), value_type{2});

    m->fill(value_type{42});

    EXPECT_EQ(m->at(0), value_type{42});
    EXPECT_EQ(m->at(1), value_type{42});
}


TYPED_TEST(Dense, CanBeFilledWithValueForStridedMatrices)
{
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto m = gko::initialize<gko::matrix::Dense<TypeParam>>(
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


TYPED_TEST(Dense, AppliesToDense)
{
    using T = typename TestFixture::value_type;
    T in_stride{-1};
    this->mtx3->get_values()[3] = in_stride;

    this->mtx2->apply(this->mtx1, this->mtx3);

    EXPECT_EQ(this->mtx3->at(0, 0), T{-0.5});
    EXPECT_EQ(this->mtx3->at(0, 1), T{-0.5});
    EXPECT_EQ(this->mtx3->at(0, 2), T{-0.5});
    EXPECT_EQ(this->mtx3->at(1, 0), T{1.0});
    EXPECT_EQ(this->mtx3->at(1, 1), T{1.0});
    EXPECT_EQ(this->mtx3->at(1, 2), T{1.0});
    ASSERT_EQ(this->mtx3->get_values()[3], in_stride);
}


TYPED_TEST(Dense, AppliesToMixedDense)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using MixedT = typename MixedMtx::value_type;
    auto mmtx1 = MixedMtx::create(this->exec);
    auto mmtx3 = MixedMtx::create(this->exec);
    this->mtx1->convert_to(mmtx1);
    this->mtx3->convert_to(mmtx3);

    this->mtx2->apply(mmtx1, mmtx3);

    EXPECT_EQ(mmtx3->at(0, 0), MixedT{-0.5});
    EXPECT_EQ(mmtx3->at(0, 1), MixedT{-0.5});
    EXPECT_EQ(mmtx3->at(0, 2), MixedT{-0.5});
    EXPECT_EQ(mmtx3->at(1, 0), MixedT{1.0});
    EXPECT_EQ(mmtx3->at(1, 1), MixedT{1.0});
    ASSERT_EQ(mmtx3->at(1, 2), MixedT{1.0});
}


TYPED_TEST(Dense, AppliesLinearCombinationToDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({-1.0}, this->exec);
    auto beta = gko::initialize<Mtx>({2.0}, this->exec);
    T in_stride{-1};
    this->mtx3->get_values()[3] = in_stride;

    this->mtx2->apply(alpha, this->mtx1, beta, this->mtx3);

    EXPECT_EQ(this->mtx3->at(0, 0), T{2.5});
    EXPECT_EQ(this->mtx3->at(0, 1), T{4.5});
    EXPECT_EQ(this->mtx3->at(0, 2), T{6.5});
    EXPECT_EQ(this->mtx3->at(1, 0), T{0.0});
    EXPECT_EQ(this->mtx3->at(1, 1), T{2.0});
    EXPECT_EQ(this->mtx3->at(1, 2), T{4.0});
    ASSERT_EQ(this->mtx3->get_values()[3], in_stride);
}


TYPED_TEST(Dense, AppliesLinearCombinationToMixedDense)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using MixedT = typename MixedMtx::value_type;
    auto mmtx1 = MixedMtx::create(this->exec);
    auto mmtx3 = MixedMtx::create(this->exec);
    this->mtx1->convert_to(mmtx1);
    this->mtx3->convert_to(mmtx3);
    auto alpha = gko::initialize<MixedMtx>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedMtx>({2.0}, this->exec);

    this->mtx2->apply(alpha, mmtx1, beta, mmtx3);

    EXPECT_EQ(mmtx3->at(0, 0), MixedT{2.5});
    EXPECT_EQ(mmtx3->at(0, 1), MixedT{4.5});
    EXPECT_EQ(mmtx3->at(0, 2), MixedT{6.5});
    EXPECT_EQ(mmtx3->at(1, 0), MixedT{0.0});
    EXPECT_EQ(mmtx3->at(1, 1), MixedT{2.0});
    ASSERT_EQ(mmtx3->at(1, 2), MixedT{4.0});
}


TYPED_TEST(Dense, ApplyFailsOnWrongInnerDimension)
{
    using Mtx = typename TestFixture::Mtx;
    auto res = Mtx::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx2->apply(this->mtx1, res), gko::DimensionMismatch);
}


TYPED_TEST(Dense, ApplyFailsOnWrongNumberOfRows)
{
    using Mtx = typename TestFixture::Mtx;
    auto res = Mtx::create(this->exec, gko::dim<2>{3});

    ASSERT_THROW(this->mtx1->apply(this->mtx2, res), gko::DimensionMismatch);
}


TYPED_TEST(Dense, ApplyFailsOnWrongNumberOfCols)
{
    using Mtx = typename TestFixture::Mtx;
    auto res = Mtx::create(this->exec, gko::dim<2>{2}, 3);

    ASSERT_THROW(this->mtx1->apply(this->mtx2, res), gko::DimensionMismatch);
}


TYPED_TEST(Dense, ScalesData)
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


TYPED_TEST(Dense, ScalesDataMixed)
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


TYPED_TEST(Dense, InvScalesData)
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


TYPED_TEST(Dense, ScalesDataWithScalar)
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


TYPED_TEST(Dense, InvScalesDataWithScalar)
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


TYPED_TEST(Dense, ScalesDataWithStride)
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


TYPED_TEST(Dense, AddsScaled)
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


TYPED_TEST(Dense, AddsScaledMixed)
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


TYPED_TEST(Dense, SubtractsScaled)
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


TYPED_TEST(Dense, AddsScaledWithScalar)
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


TYPED_TEST(Dense, AddScaledFailsOnWrongSizes)
{
    using Mtx = typename TestFixture::Mtx;
    auto alpha = Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->mtx1->add_scaled(alpha, this->mtx2),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, AddsScaledDiag)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto diag = gko::matrix::Diagonal<T>::create(this->exec, 2, I<T>{3.0, 2.0});

    this->mtx2->add_scaled(alpha, diag);

    ASSERT_EQ(this->mtx2->at(0, 0), T{7.0});
    ASSERT_EQ(this->mtx2->at(0, 1), T{-1.0});
    ASSERT_EQ(this->mtx2->at(1, 0), T{-2.0});
    ASSERT_EQ(this->mtx2->at(1, 1), T{6.0});
}


TYPED_TEST(Dense, SubtractsScaledDiag)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({-2.0}, this->exec);
    auto diag = gko::matrix::Diagonal<T>::create(this->exec, 2, I<T>{3.0, 2.0});

    this->mtx2->sub_scaled(alpha, diag);

    ASSERT_EQ(this->mtx2->at(0, 0), T{7.0});
    ASSERT_EQ(this->mtx2->at(0, 1), T{-1.0});
    ASSERT_EQ(this->mtx2->at(1, 0), T{-2.0});
    ASSERT_EQ(this->mtx2->at(1, 1), T{6.0});
}


TYPED_TEST(Dense, ComputesDot)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 3});

    this->mtx1->compute_dot(this->mtx3, result);

    EXPECT_EQ(result->at(0, 0), T{1.75});
    EXPECT_EQ(result->at(0, 1), T{7.75});
    ASSERT_EQ(result->at(0, 2), T{17.75});
}


TYPED_TEST(Dense, ComputesDotMixed)
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


TYPED_TEST(Dense, ComputesConjDot)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 3});

    this->mtx1->compute_conj_dot(this->mtx3, result);

    EXPECT_EQ(result->at(0, 0), T{1.75});
    EXPECT_EQ(result->at(0, 1), T{7.75});
    ASSERT_EQ(result->at(0, 2), T{17.75});
}


TYPED_TEST(Dense, ComputesConjDotMixed)
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


TYPED_TEST(Dense, ComputesNorm2)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using T_nc = gko::remove_complex<T>;
    using NormVector = gko::matrix::Dense<T_nc>;
    auto mtx(gko::initialize<Mtx>(
        {I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}}, this->exec));
    auto result = NormVector::create(this->exec, gko::dim<2>{1, 2});

    mtx->compute_norm2(result);

    EXPECT_EQ(result->at(0, 0), T_nc{3.0});
    EXPECT_EQ(result->at(0, 1), T_nc{5.0});
}


TYPED_TEST(Dense, ComputesNorm2Mixed)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using MixedMtx = typename TestFixture::MixedMtx;
    using MixedT = typename MixedMtx::value_type;
    using MixedT_nc = gko::remove_complex<MixedT>;
    using MixedNormVector = gko::matrix::Dense<MixedT_nc>;
    auto mtx(gko::initialize<Mtx>(
        {I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}}, this->exec));
    auto result = MixedNormVector::create(this->exec, gko::dim<2>{1, 2});

    mtx->compute_norm2(result);

    EXPECT_EQ(result->at(0, 0), MixedT_nc{3.0});
    EXPECT_EQ(result->at(0, 1), MixedT_nc{5.0});
}


TYPED_TEST(Dense, ComputesNorm2Squared)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using T_nc = gko::remove_complex<T>;
    using NormVector = gko::matrix::Dense<T_nc>;
    gko::array<char> tmp{this->exec};
    auto mtx(gko::initialize<Mtx>(
        {I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}}, this->exec));
    auto result = NormVector::create(this->exec, gko::dim<2>{1, 2});

    gko::kernels::reference::dense::compute_squared_norm2(
        gko::as<gko::ReferenceExecutor>(this->exec), mtx.get(), result.get(),
        tmp);

    EXPECT_EQ(result->at(0, 0), T_nc{9.0});
    EXPECT_EQ(result->at(0, 1), T_nc{25.0});
}


TYPED_TEST(Dense, ComputesSqrt)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using T_nc = gko::remove_complex<T>;
    using NormVector = gko::matrix::Dense<T_nc>;
    auto mtx(gko::initialize<NormVector>(I<I<T_nc>>{{9.0, 25.0}}, this->exec));

    gko::kernels::reference::dense::compute_sqrt(
        gko::as<gko::ReferenceExecutor>(this->exec), mtx.get());

    EXPECT_EQ(mtx->at(0, 0), T_nc{3.0});
    EXPECT_EQ(mtx->at(0, 1), T_nc{5.0});
}


TYPED_TEST(Dense, ComputesNorm1)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using T_nc = gko::remove_complex<T>;
    using NormVector = gko::matrix::Dense<T_nc>;
    auto mtx(gko::initialize<Mtx>(
        {I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}, I<T>{-1.0, -1.0}},
        this->exec));
    auto result = NormVector::create(this->exec, gko::dim<2>{1, 2});

    mtx->compute_norm1(result);

    EXPECT_EQ(result->at(0, 0), T_nc{6.0});
    EXPECT_EQ(result->at(0, 1), T_nc{8.0});
}


TYPED_TEST(Dense, ComputesNorm1Mixed)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using MixedT = typename MixedMtx::value_type;
    using MixedT_nc = gko::remove_complex<MixedT>;
    using MixedNormVector = gko::matrix::Dense<MixedT_nc>;
    auto mtx(
        gko::initialize<MixedMtx>({I<MixedT>{1.0, 0.0}, I<MixedT>{2.0, 3.0},
                                   I<MixedT>{2.0, 4.0}, I<MixedT>{-1.0, -1.0}},
                                  this->exec));
    auto result = MixedNormVector::create(this->exec, gko::dim<2>{1, 2});

    mtx->compute_norm1(result);

    EXPECT_EQ(result->at(0, 0), MixedT_nc{6.0});
    EXPECT_EQ(result->at(0, 1), MixedT_nc{8.0});
}


TYPED_TEST(Dense, ComputesMean)
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


TYPED_TEST(Dense, ComputesMeanFailsOnWrongResultSize)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->mtx4->compute_mean(result), gko::DimensionMismatch);
}


TYPED_TEST(Dense, ComputeDotFailsOnWrongInputSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 3});

    ASSERT_THROW(this->mtx1->compute_dot(this->mtx2, result),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ComputeDotFailsOnWrongResultSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->mtx1->compute_dot(this->mtx3, result),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ComputeConjDotFailsOnWrongInputSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 3});

    ASSERT_THROW(this->mtx1->compute_conj_dot(this->mtx2, result),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ComputeConjDotFailsOnWrongResultSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->mtx1->compute_conj_dot(this->mtx3, result),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ConvertsToPrecision)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherDense = typename gko::matrix::Dense<OtherT>;
    auto tmp = OtherDense::create(this->exec);
    auto res = Dense::create(this->exec);
    // If OtherT is more precise: 0, otherwise r
    auto residual = r<OtherT>::value < r<T>::value
                        ? gko::remove_complex<T>{0}
                        : gko::remove_complex<T>{r<OtherT>::value};

    this->mtx1->convert_to(tmp);
    tmp->convert_to(res);

    GKO_ASSERT_MTX_NEAR(this->mtx1, res, residual);
}


TYPED_TEST(Dense, MovesToPrecision)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherDense = typename gko::matrix::Dense<OtherT>;
    auto tmp = OtherDense::create(this->exec);
    auto res = Dense::create(this->exec);
    // If OtherT is more precise: 0, otherwise r
    auto residual = r<OtherT>::value < r<T>::value
                        ? gko::remove_complex<T>{0}
                        : gko::remove_complex<T>{r<OtherT>::value};

    this->mtx1->move_to(tmp);
    tmp->move_to(res);

    GKO_ASSERT_MTX_NEAR(this->mtx1, res, residual);
}


TYPED_TEST(Dense, SquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = gko::as<Mtx>(this->mtx5->transpose());

    GKO_ASSERT_MTX_NEAR(
        trans, l<T>({{1.0, -2.0, 2.1}, {-1.0, 2.0, 3.4}, {-0.5, 4.5, 1.2}}),
        0.0);
}


TYPED_TEST(Dense, SquareMatrixIsTransposableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = Mtx::create(this->exec, this->mtx5->get_size());

    this->mtx5->transpose(trans);

    GKO_ASSERT_MTX_NEAR(
        trans, l<T>({{1.0, -2.0, 2.1}, {-1.0, 2.0, 3.4}, {-0.5, 4.5, 1.2}}),
        0.0);
}


TYPED_TEST(Dense, SquareSubmatrixIsTransposableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = Mtx::create(this->exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {0, 2})->transpose(trans);

    GKO_ASSERT_MTX_NEAR(trans, l<T>({{1.0, -2.0}, {-1.0, 2.0}}), 0.0);
    ASSERT_EQ(trans->get_stride(), 4);
}


TYPED_TEST(Dense, SquareMatrixIsTransposableIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;

    ASSERT_THROW(this->mtx5->transpose(Mtx::create(this->exec)),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, NonSquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = gko::as<Mtx>(this->mtx4->transpose());

    GKO_ASSERT_MTX_NEAR(trans, l<T>({{1.0, 0.0}, {3.0, 5.0}, {2.0, 0.0}}), 0.0);
}


TYPED_TEST(Dense, NonSquareMatrixIsTransposableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans =
        Mtx::create(this->exec, gko::transpose(this->mtx4->get_size()));

    this->mtx4->transpose(trans);

    GKO_ASSERT_MTX_NEAR(trans, l<T>({{1.0, 0.0}, {3.0, 5.0}, {2.0, 0.0}}), 0.0);
}


TYPED_TEST(Dense, NonSquareSubmatrixIsTransposableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = Mtx::create(this->exec, gko::dim<2>{2, 1}, 5);

    this->mtx4->create_submatrix({0, 1}, {0, 2})->transpose(trans);

    GKO_ASSERT_MTX_NEAR(trans, l({1.0, 3.0}), 0.0);
    ASSERT_EQ(trans->get_stride(), 5);
}


TYPED_TEST(Dense, NonSquareMatrixIsTransposableIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;

    ASSERT_THROW(this->mtx4->transpose(Mtx::create(this->exec)),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ExtractsDiagonalFromSquareMatrix)
{
    using T = typename TestFixture::value_type;

    auto diag = this->mtx5->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 3);
    ASSERT_EQ(diag->get_size()[1], 3);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{2.});
    ASSERT_EQ(diag->get_values()[2], T{1.2});
}


TYPED_TEST(Dense, ExtractsDiagonalFromTallSkinnyMatrix)
{
    using T = typename TestFixture::value_type;

    auto diag = this->mtx4->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{5.});
}


TYPED_TEST(Dense, ExtractsDiagonalFromShortFatMatrix)
{
    using T = typename TestFixture::value_type;

    auto diag = this->mtx8->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{2.});
}


TYPED_TEST(Dense, ExtractsDiagonalFromSquareMatrixIntoDiagonal)
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


TYPED_TEST(Dense, ExtractsDiagonalFromTallSkinnyMatrixIntoDiagonal)
{
    using T = typename TestFixture::value_type;
    auto diag = gko::matrix::Diagonal<T>::create(this->exec, 2);

    this->mtx4->extract_diagonal(diag);

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{5.});
}


TYPED_TEST(Dense, ExtractsDiagonalFromShortFatMatrixIntoDiagonal)
{
    using T = typename TestFixture::value_type;
    auto diag = gko::matrix::Diagonal<T>::create(this->exec, 2);

    this->mtx8->extract_diagonal(diag);

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{2.});
}


TYPED_TEST(Dense, InplaceAbsolute)
{
    using T = typename TestFixture::value_type;

    this->mtx5->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        this->mtx5, l<T>({{1.0, 1.0, 0.5}, {2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}),
        0.0);
}


TYPED_TEST(Dense, InplaceAbsoluteSubMatrix)
{
    using T = typename TestFixture::value_type;
    auto mtx = this->mtx5->create_submatrix(gko::span{0, 2}, gko::span{0, 2});

    mtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        this->mtx5, l<T>({{1.0, 1.0, -0.5}, {2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}),
        0.0);
}


TYPED_TEST(Dense, OutplaceAbsolute)
{
    using T = typename TestFixture::value_type;

    auto abs_mtx = this->mtx5->compute_absolute();

    GKO_ASSERT_MTX_NEAR(
        abs_mtx, l<T>({{1.0, 1.0, 0.5}, {2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}),
        0.0);
}


TYPED_TEST(Dense, OutplaceAbsoluteIntoDense)
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


TYPED_TEST(Dense, OutplaceAbsoluteSubMatrix)
{
    using T = typename TestFixture::value_type;
    auto mtx = this->mtx5->create_submatrix(gko::span{0, 2}, gko::span{0, 2});

    auto abs_mtx = mtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx, l<T>({{1.0, 1.0}, {2.0, 2.0}}), 0);
    GKO_ASSERT_EQ(abs_mtx->get_stride(), 2);
}


TYPED_TEST(Dense, OutplaceSubmatrixAbsoluteIntoDense)
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


TYPED_TEST(Dense, AppliesToComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Vec = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();
    auto b =
        gko::initialize<Vec>({{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
                              {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
                              {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}},
                             exec);
    auto x = Vec::create(exec, gko::dim<2>{2, 2});

    this->mtx1->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{complex_type{14.0, 16.0}, complex_type{20.0, 22.0}},
           {complex_type{17.0, 19.0}, complex_type{24.5, 26.5}}}),
        0.0);
}


TYPED_TEST(Dense, AppliesToMixedComplex)
{
    using mixed_value_type =
        gko::next_precision<typename TestFixture::value_type>;
    using mixed_complex_type = gko::to_complex<mixed_value_type>;
    using Vec = gko::matrix::Dense<mixed_complex_type>;
    auto exec = gko::ReferenceExecutor::create();
    auto b = gko::initialize<Vec>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}},
         {mixed_complex_type{3.0, 4.0}, mixed_complex_type{4.0, 5.0}}},
        exec);
    auto x = Vec::create(exec, gko::dim<2>{2, 2});

    this->mtx1->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{mixed_complex_type{14.0, 16.0}, mixed_complex_type{20.0, 22.0}},
           {mixed_complex_type{17.0, 19.0}, mixed_complex_type{24.5, 26.5}}}),
        0.0);
}


TYPED_TEST(Dense, AdvancedAppliesToComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using DenseComplex = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    auto b = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}},
        exec);
    auto x = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}},
        exec);
    auto alpha = gko::initialize<Dense>({-1.0}, this->exec);
    auto beta = gko::initialize<Dense>({2.0}, this->exec);

    this->mtx1->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{complex_type{-12.0, -16.0}, complex_type{-16.0, -20.0}},
           {complex_type{-13.0, -15.0}, complex_type{-18.5, -20.5}}}),
        0.0);
}


TYPED_TEST(Dense, AdvancedAppliesToMixedComplex)
{
    using mixed_value_type =
        gko::next_precision<typename TestFixture::value_type>;
    using mixed_complex_type = gko::to_complex<mixed_value_type>;
    using MixedDense = gko::matrix::Dense<mixed_value_type>;
    using MixedDenseComplex = gko::matrix::Dense<mixed_complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    auto b = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}},
         {mixed_complex_type{3.0, 4.0}, mixed_complex_type{4.0, 5.0}}},
        exec);
    auto x = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}}},
        exec);
    auto alpha = gko::initialize<MixedDense>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedDense>({2.0}, this->exec);

    this->mtx1->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{mixed_complex_type{-12.0, -16.0}, mixed_complex_type{-16.0, -20.0}},
           {mixed_complex_type{-13.0, -15.0},
            mixed_complex_type{-18.5, -20.5}}}),
        0.0);
}


TYPED_TEST(Dense, MakeComplex)
{
    using T = typename TestFixture::value_type;

    auto complex_mtx = this->mtx5->make_complex();

    GKO_ASSERT_MTX_NEAR(complex_mtx, this->mtx5, 0.0);
}


TYPED_TEST(Dense, MakeComplexIntoDense)
{
    using T = typename TestFixture::value_type;
    using ComplexMtx = typename TestFixture::ComplexMtx;
    auto exec = this->mtx5->get_executor();

    auto complex_mtx = ComplexMtx::create(exec, this->mtx5->get_size());
    this->mtx5->make_complex(complex_mtx);

    GKO_ASSERT_MTX_NEAR(complex_mtx, this->mtx5, 0.0);
}


TYPED_TEST(Dense, MakeComplexIntoDenseFailsForWrongDimensions)
{
    using T = typename TestFixture::value_type;
    using ComplexMtx = typename TestFixture::ComplexMtx;
    auto exec = this->mtx5->get_executor();

    auto complex_mtx = ComplexMtx::create(exec);

    ASSERT_THROW(this->mtx5->make_complex(complex_mtx), gko::DimensionMismatch);
}


TYPED_TEST(Dense, GetReal)
{
    using T = typename TestFixture::value_type;

    auto real_mtx = this->mtx5->get_real();

    GKO_ASSERT_MTX_NEAR(real_mtx, this->mtx5, 0.0);
}


TYPED_TEST(Dense, GetRealIntoDense)
{
    using T = typename TestFixture::value_type;
    using RealMtx = typename TestFixture::RealMtx;
    auto exec = this->mtx5->get_executor();

    auto real_mtx = RealMtx::create(exec, this->mtx5->get_size());
    this->mtx5->get_real(real_mtx);

    GKO_ASSERT_MTX_NEAR(real_mtx, this->mtx5, 0.0);
}


TYPED_TEST(Dense, GetRealIntoDenseFailsForWrongDimensions)
{
    using T = typename TestFixture::value_type;
    using RealMtx = typename TestFixture::RealMtx;
    auto exec = this->mtx5->get_executor();

    auto real_mtx = RealMtx::create(exec);
    ASSERT_THROW(this->mtx5->get_real(real_mtx), gko::DimensionMismatch);
}


TYPED_TEST(Dense, GetImag)
{
    using T = typename TestFixture::value_type;

    auto imag_mtx = this->mtx5->get_imag();

    GKO_ASSERT_MTX_NEAR(
        imag_mtx, l<T>({{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}),
        0.0);
}


TYPED_TEST(Dense, GetImagIntoDense)
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


TYPED_TEST(Dense, GetImagIntoDenseFailsForWrongDimensions)
{
    using T = typename TestFixture::value_type;
    using RealMtx = typename TestFixture::RealMtx;
    auto exec = this->mtx5->get_executor();

    auto imag_mtx = RealMtx::create(exec);
    ASSERT_THROW(this->mtx5->get_imag(imag_mtx), gko::DimensionMismatch);
}


TYPED_TEST(Dense, MakeTemporaryConversionDoesntConvertOnMatch)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({8.0}, this->exec);

    ASSERT_EQ(gko::make_temporary_conversion<T>(alpha).get(), alpha.get());
}


TYPED_TEST(Dense, MakeTemporaryConversionConvertsBack)
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


TYPED_TEST(Dense, MakeTemporaryConversionConstDoesntConvertBack)
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


TYPED_TEST(Dense, ScaleAddIdentityRectangular)
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
class DenseWithIndexType
    : public Dense<
          typename std::tuple_element<0, decltype(ValueIndexType())>::type> {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Permutation = gko::matrix::Permutation<index_type>;
    using ScaledPermutation =
        gko::matrix::ScaledPermutation<value_type, index_type>;


    DenseWithIndexType()
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

TYPED_TEST_SUITE(DenseWithIndexType, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


template <typename ValueType, typename IndexType>
void assert_coo_eq_mtx4(const gko::matrix::Coo<ValueType, IndexType>* coo_mtx)
{
    auto v = coo_mtx->get_const_values();
    auto c = coo_mtx->get_const_col_idxs();
    auto r = coo_mtx->get_const_row_idxs();

    ASSERT_EQ(coo_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], ValueType{1.0});
    EXPECT_EQ(v[1], ValueType{3.0});
    EXPECT_EQ(v[2], ValueType{2.0});
    EXPECT_EQ(v[3], ValueType{5.0});
}


TYPED_TEST(DenseWithIndexType, ConvertsToCoo)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Coo = typename gko::matrix::Coo<value_type, index_type>;
    auto coo_mtx = Coo::create(this->mtx4->get_executor());

    this->mtx4->convert_to(coo_mtx);

    assert_coo_eq_mtx4(coo_mtx.get());
}


TYPED_TEST(DenseWithIndexType, MovesToCoo)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Coo = typename gko::matrix::Coo<value_type, index_type>;
    auto coo_mtx = Coo::create(this->mtx4->get_executor());

    this->mtx4->move_to(coo_mtx);

    assert_coo_eq_mtx4(coo_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_csr_eq_mtx4(const gko::matrix::Csr<ValueType, IndexType>* csr_mtx)
{
    auto v = csr_mtx->get_const_values();
    auto c = csr_mtx->get_const_col_idxs();
    auto r = csr_mtx->get_const_row_ptrs();
    ASSERT_EQ(csr_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(csr_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], ValueType{1.0});
    EXPECT_EQ(v[1], ValueType{3.0});
    EXPECT_EQ(v[2], ValueType{2.0});
    EXPECT_EQ(v[3], ValueType{5.0});
}


TYPED_TEST(DenseWithIndexType, ConvertsToCsr)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Csr = typename gko::matrix::Csr<value_type, index_type>;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_c = Csr::create(this->mtx4->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx4->get_executor(), csr_s_merge);

    this->mtx4->convert_to(csr_mtx_c);
    this->mtx4->convert_to(csr_mtx_m);

    assert_csr_eq_mtx4(csr_mtx_c.get());
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_c, csr_mtx_m, 0.0);
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(DenseWithIndexType, MovesToCsr)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Csr = typename gko::matrix::Csr<value_type, index_type>;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_c = Csr::create(this->mtx4->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx4->get_executor(), csr_s_merge);
    auto mtx_clone = this->mtx4->clone();

    this->mtx4->move_to(csr_mtx_c);
    mtx_clone->move_to(csr_mtx_m);

    assert_csr_eq_mtx4(csr_mtx_c.get());
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_c, csr_mtx_m, 0.0);
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


template <typename ValueType, typename IndexType>
void assert_sparsity_csr_eq_mtx4(
    const gko::matrix::SparsityCsr<ValueType, IndexType>* sparsity_csr_mtx)
{
    auto v = sparsity_csr_mtx->get_const_value();
    auto c = sparsity_csr_mtx->get_const_col_idxs();
    auto r = sparsity_csr_mtx->get_const_row_ptrs();

    ASSERT_EQ(sparsity_csr_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sparsity_csr_mtx->get_num_nonzeros(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], ValueType{1.0});
}


TYPED_TEST(DenseWithIndexType, ConvertsToSparsityCsr)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using SparsityCsr =
        typename gko::matrix::SparsityCsr<value_type, index_type>;
    auto sparsity_csr_mtx = SparsityCsr::create(this->mtx4->get_executor());

    this->mtx4->convert_to(sparsity_csr_mtx);

    assert_sparsity_csr_eq_mtx4(sparsity_csr_mtx.get());
}


TYPED_TEST(DenseWithIndexType, MovesToSparsityCsr)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using SparsityCsr =
        typename gko::matrix::SparsityCsr<value_type, index_type>;
    auto sparsity_csr_mtx = SparsityCsr::create(this->mtx4->get_executor());

    this->mtx4->move_to(sparsity_csr_mtx);

    assert_sparsity_csr_eq_mtx4(sparsity_csr_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_ell_eq_mtx6(const gko::matrix::Ell<ValueType, IndexType>* ell_mtx)
{
    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 4);
    ASSERT_EQ(ell_mtx->get_stride(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], gko::invalid_index<IndexType>());
    EXPECT_EQ(v[0], ValueType{1.0});
    EXPECT_EQ(v[1], ValueType{1.5});
    EXPECT_EQ(v[2], ValueType{2.0});
    EXPECT_EQ(v[3], ValueType{0.0});
}


TYPED_TEST(DenseWithIndexType, ConvertsToEll)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Ell = typename gko::matrix::Ell<value_type, index_type>;
    auto ell_mtx = Ell::create(this->mtx6->get_executor());

    this->mtx6->convert_to(ell_mtx);

    assert_ell_eq_mtx6(ell_mtx.get());
}


TYPED_TEST(DenseWithIndexType, MovesToEll)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Ell = typename gko::matrix::Ell<value_type, index_type>;
    auto ell_mtx = Ell::create(this->mtx6->get_executor());

    this->mtx6->move_to(ell_mtx);

    assert_ell_eq_mtx6(ell_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_strided_ell_eq_mtx6(
    const gko::matrix::Ell<ValueType, IndexType>* ell_mtx)
{
    constexpr auto invalid_index = gko::invalid_index<IndexType>();
    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 6);
    ASSERT_EQ(ell_mtx->get_stride(), 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], invalid_index);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], invalid_index);
    EXPECT_EQ(c[5], invalid_index);
    EXPECT_EQ(v[0], ValueType{1.0});
    EXPECT_EQ(v[1], ValueType{1.5});
    EXPECT_EQ(v[2], ValueType{0.0});
    EXPECT_EQ(v[3], ValueType{2.0});
    EXPECT_EQ(v[4], ValueType{0.0});
    EXPECT_EQ(v[5], ValueType{0.0});
}


TYPED_TEST(DenseWithIndexType, ConvertsToEllWithStride)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Ell = typename gko::matrix::Ell<value_type, index_type>;
    auto ell_mtx =
        Ell::create(this->mtx6->get_executor(), gko::dim<2>{2, 3}, 2, 3);

    this->mtx6->convert_to(ell_mtx);

    assert_strided_ell_eq_mtx6(ell_mtx.get());
}


TYPED_TEST(DenseWithIndexType, MovesToEllWithStride)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Ell = typename gko::matrix::Ell<value_type, index_type>;
    auto ell_mtx =
        Ell::create(this->mtx6->get_executor(), gko::dim<2>{2, 3}, 2, 3);

    this->mtx6->move_to(ell_mtx);

    assert_strided_ell_eq_mtx6(ell_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_hybrid_auto_eq_mtx4(
    const gko::matrix::Hybrid<ValueType, IndexType>* hybrid_mtx)
{
    auto v = hybrid_mtx->get_const_coo_values();
    auto c = hybrid_mtx->get_const_coo_col_idxs();
    auto r = hybrid_mtx->get_const_coo_row_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 0);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 4);
    EXPECT_EQ(n, 0);
    EXPECT_EQ(p, 2);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], ValueType{1.0});
    EXPECT_EQ(v[1], ValueType{3.0});
    EXPECT_EQ(v[2], ValueType{2.0});
    EXPECT_EQ(v[3], ValueType{5.0});
}


TYPED_TEST(DenseWithIndexType, MovesToHybridAutomatically)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Hybrid = typename gko::matrix::Hybrid<value_type, index_type>;
    auto hybrid_mtx = Hybrid::create(this->mtx4->get_executor());

    this->mtx4->move_to(hybrid_mtx);

    assert_hybrid_auto_eq_mtx4(hybrid_mtx.get());
}


TYPED_TEST(DenseWithIndexType, ConvertsToHybridAutomatically)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Hybrid = typename gko::matrix::Hybrid<value_type, index_type>;
    auto hybrid_mtx = Hybrid::create(this->mtx4->get_executor());

    this->mtx4->convert_to(hybrid_mtx);

    assert_hybrid_auto_eq_mtx4(hybrid_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_hybrid_strided_eq_mtx4(
    const gko::matrix::Hybrid<ValueType, IndexType>* hybrid_mtx)
{
    auto v = hybrid_mtx->get_const_coo_values();
    auto c = hybrid_mtx->get_const_coo_col_idxs();
    auto r = hybrid_mtx->get_const_coo_row_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 0);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 4);
    EXPECT_EQ(n, 0);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], ValueType{1.0});
    EXPECT_EQ(v[1], ValueType{3.0});
    EXPECT_EQ(v[2], ValueType{2.0});
    EXPECT_EQ(v[3], ValueType{5.0});
}


TYPED_TEST(DenseWithIndexType, MovesToHybridWithStrideAutomatically)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Hybrid = typename gko::matrix::Hybrid<value_type, index_type>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{2, 3}, 0, 3);

    this->mtx4->move_to(hybrid_mtx);

    assert_hybrid_strided_eq_mtx4(hybrid_mtx.get());
}


TYPED_TEST(DenseWithIndexType, ConvertsToHybridWithStrideAutomatically)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Hybrid = typename gko::matrix::Hybrid<value_type, index_type>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{2, 3}, 0, 3);

    this->mtx4->convert_to(hybrid_mtx);

    assert_hybrid_strided_eq_mtx4(hybrid_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_hybrid_limited_eq_mtx4(
    const gko::matrix::Hybrid<ValueType, IndexType>* hybrid_mtx)
{
    constexpr auto invalid_index = gko::invalid_index<IndexType>();
    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 6);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 1);
    EXPECT_EQ(n, 2);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], invalid_index);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], invalid_index);
    EXPECT_EQ(c[5], invalid_index);
    EXPECT_EQ(v[0], ValueType{1.0});
    EXPECT_EQ(v[1], ValueType{5.0});
    EXPECT_EQ(v[2], ValueType{0.0});
    EXPECT_EQ(v[3], ValueType{3.0});
    EXPECT_EQ(v[4], ValueType{0.0});
    EXPECT_EQ(v[5], ValueType{0.0});
    EXPECT_EQ(hybrid_mtx->get_const_coo_values()[0], ValueType{2.0});
    EXPECT_EQ(hybrid_mtx->get_const_coo_row_idxs()[0], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_col_idxs()[0], 2);
}


TYPED_TEST(DenseWithIndexType, MovesToHybridWithStrideAndCooLengthByColumns2)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Hybrid = typename gko::matrix::Hybrid<value_type, index_type>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{2, 3}, 2, 3, 3,
                       std::make_shared<typename Hybrid::column_limit>(2));

    this->mtx4->move_to(hybrid_mtx);

    assert_hybrid_limited_eq_mtx4(hybrid_mtx.get());
}


TYPED_TEST(DenseWithIndexType, ConvertsToHybridWithStrideAndCooLengthByColumns2)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Hybrid = typename gko::matrix::Hybrid<value_type, index_type>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{2, 3}, 2, 3, 3,
                       std::make_shared<typename Hybrid::column_limit>(2));

    this->mtx4->convert_to(hybrid_mtx);

    assert_hybrid_limited_eq_mtx4(hybrid_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_hybrid_percent_eq_mtx4(
    const gko::matrix::Hybrid<ValueType, IndexType>* hybrid_mtx)
{
    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();
    auto coo_v = hybrid_mtx->get_const_coo_values();
    auto coo_c = hybrid_mtx->get_const_coo_col_idxs();
    auto coo_r = hybrid_mtx->get_const_coo_row_idxs();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 3);
    EXPECT_EQ(n, 1);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], gko::invalid_index<IndexType>());
    EXPECT_EQ(v[0], ValueType{1.0});
    EXPECT_EQ(v[1], ValueType{5.0});
    EXPECT_EQ(v[2], ValueType{0.0});
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 2);
    EXPECT_EQ(coo_v[0], ValueType{3.0});
    EXPECT_EQ(coo_v[1], ValueType{2.0});
    EXPECT_EQ(coo_c[0], 1);
    EXPECT_EQ(coo_c[1], 2);
    EXPECT_EQ(coo_r[0], 0);
    EXPECT_EQ(coo_r[1], 0);
}


TYPED_TEST(DenseWithIndexType, MovesToHybridWithStrideByPercent40)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Hybrid = typename gko::matrix::Hybrid<value_type, index_type>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{2, 3}, 1, 3,
                       std::make_shared<typename Hybrid::imbalance_limit>(0.4));

    this->mtx4->move_to(hybrid_mtx);

    assert_hybrid_percent_eq_mtx4(hybrid_mtx.get());
}


TYPED_TEST(DenseWithIndexType, ConvertsToHybridWithStrideByPercent40)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Hybrid = typename gko::matrix::Hybrid<value_type, index_type>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{2, 3}, 1, 3,
                       std::make_shared<typename Hybrid::imbalance_limit>(0.4));

    this->mtx4->convert_to(hybrid_mtx);

    assert_hybrid_percent_eq_mtx4(hybrid_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_sellp_eq_mtx7(
    const gko::matrix::Sellp<ValueType, IndexType>* sellp_mtx)
{
    constexpr auto invalid_index = gko::invalid_index<IndexType>();
    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 3);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(),
              3 * gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_slice_size(), gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_stride_factor(),
              gko::matrix::default_stride_factor);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size + 1], invalid_index);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size], 2);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size + 1], invalid_index);
    EXPECT_EQ(v[0], ValueType{1.0});
    EXPECT_EQ(v[1], ValueType{1.5});
    EXPECT_EQ(v[gko::matrix::default_slice_size], ValueType{2.0});
    EXPECT_EQ(v[gko::matrix::default_slice_size + 1], ValueType{0.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size], ValueType{3.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size + 1], ValueType{0.0});
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 3);
    EXPECT_EQ(l[0], 3);
}


TYPED_TEST(DenseWithIndexType, ConvertsToSellp)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Sellp = typename gko::matrix::Sellp<value_type, index_type>;
    auto sellp_mtx = Sellp::create(this->mtx7->get_executor());

    this->mtx7->convert_to(sellp_mtx);

    assert_sellp_eq_mtx7(sellp_mtx.get());
}


TYPED_TEST(DenseWithIndexType, MovesToSellp)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Sellp = typename gko::matrix::Sellp<value_type, index_type>;
    auto sellp_mtx = Sellp::create(this->mtx7->get_executor());

    this->mtx7->move_to(sellp_mtx);

    assert_sellp_eq_mtx7(sellp_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_sellp_strided_eq_mtx7(
    const gko::matrix::Sellp<ValueType, IndexType>* sellp_mtx)
{
    constexpr auto invalid_index = gko::invalid_index<IndexType>();
    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 4);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(), 8);
    ASSERT_EQ(sellp_mtx->get_slice_size(), 2);
    ASSERT_EQ(sellp_mtx->get_stride_factor(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], invalid_index);
    EXPECT_EQ(c[4], 2);
    EXPECT_EQ(c[5], invalid_index);
    EXPECT_EQ(c[6], invalid_index);
    EXPECT_EQ(c[7], invalid_index);
    EXPECT_EQ(v[0], ValueType{1.0});
    EXPECT_EQ(v[1], ValueType{1.5});
    EXPECT_EQ(v[2], ValueType{2.0});
    EXPECT_EQ(v[3], ValueType{0.0});
    EXPECT_EQ(v[4], ValueType{3.0});
    EXPECT_EQ(v[5], ValueType{0.0});
    EXPECT_EQ(v[6], ValueType{0.0});
    EXPECT_EQ(v[7], ValueType{0.0});
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 4);
    EXPECT_EQ(l[0], 4);
}


TYPED_TEST(DenseWithIndexType, ConvertsToSellpWithSliceSizeAndStrideFactor)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Sellp = typename gko::matrix::Sellp<value_type, index_type>;
    auto sellp_mtx =
        Sellp::create(this->mtx7->get_executor(), gko::dim<2>{}, 2, 2, 0);

    this->mtx7->convert_to(sellp_mtx);

    assert_sellp_strided_eq_mtx7(sellp_mtx.get());
}


TYPED_TEST(DenseWithIndexType, MovesToSellpWithSliceSizeAndStrideFactor)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Sellp = typename gko::matrix::Sellp<value_type, index_type>;
    auto sellp_mtx =
        Sellp::create(this->mtx7->get_executor(), gko::dim<2>{}, 2, 2, 0);

    this->mtx7->move_to(sellp_mtx);

    assert_sellp_strided_eq_mtx7(sellp_mtx.get());
}


TYPED_TEST(DenseWithIndexType, ConvertsToAndFromSellpWithMoreThanOneSlice)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using Sellp = typename gko::matrix::Sellp<value_type, index_type>;
    auto x = this->template gen_mtx<Mtx>(65, 25);

    auto sellp_mtx = Sellp::create(this->exec);
    auto dense_mtx = Mtx::create(this->exec);
    x->convert_to(sellp_mtx);
    sellp_mtx->convert_to(dense_mtx);

    GKO_ASSERT_MTX_NEAR(dense_mtx, x, 0.0);
}


TYPED_TEST(Dense, ConvertsEmptyToPrecision)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherDense = typename gko::matrix::Dense<OtherT>;
    auto empty = OtherDense::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->convert_to(res);

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, MovesEmptyToPrecision)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherDense = typename gko::matrix::Dense<OtherT>;
    auto empty = OtherDense::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->move_to(res);

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, ConvertsEmptyToCoo)
{
    using Dense = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Coo = typename gko::matrix::Coo<value_type, index_type>;
    auto empty = Dense::create(this->exec);
    auto res = Coo::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, MovesEmptyToCoo)
{
    using Dense = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Coo = typename gko::matrix::Coo<value_type, index_type>;
    auto empty = Dense::create(this->exec);
    auto res = Coo::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, ConvertsEmptyMatrixToCsr)
{
    using Dense = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Csr = typename gko::matrix::Csr<value_type, index_type>;
    auto empty = Dense::create(this->exec);
    auto res = Csr::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, MovesEmptyMatrixToCsr)
{
    using Dense = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Csr = typename gko::matrix::Csr<value_type, index_type>;
    auto empty = Dense::create(this->exec);
    auto res = Csr::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, ConvertsEmptyToSparsityCsr)
{
    using Dense = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using SparsityCsr =
        typename gko::matrix::SparsityCsr<value_type, index_type>;
    auto empty = Dense::create(this->exec);
    auto res = SparsityCsr::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_nonzeros(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, MovesEmptyToSparsityCsr)
{
    using Dense = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using SparsityCsr =
        typename gko::matrix::SparsityCsr<value_type, index_type>;
    auto empty = Dense::create(this->exec);
    auto res = SparsityCsr::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_nonzeros(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, ConvertsEmptyToEll)
{
    using Dense = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Ell = typename gko::matrix::Ell<value_type, index_type>;
    auto empty = Dense::create(this->exec);
    auto res = Ell::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, MovesEmptyToEll)
{
    using Dense = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Ell = typename gko::matrix::Ell<value_type, index_type>;
    auto empty = Dense::create(this->exec);
    auto res = Ell::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, ConvertsEmptyToHybrid)
{
    using Dense = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Hybrid = typename gko::matrix::Hybrid<value_type, index_type>;
    auto empty = Dense::create(this->exec);
    auto res = Hybrid::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, MovesEmptyToHybrid)
{
    using Dense = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Hybrid = typename gko::matrix::Hybrid<value_type, index_type>;
    auto empty = Dense::create(this->exec);
    auto res = Hybrid::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, ConvertsEmptyToSellp)
{
    using Dense = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Sellp = typename gko::matrix::Sellp<value_type, index_type>;
    auto empty = Dense::create(this->exec);
    auto res = Sellp::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_slice_sets(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, MovesEmptyToSellp)
{
    using Dense = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Sellp = typename gko::matrix::Sellp<value_type, index_type>;
    auto empty = Dense::create(this->exec);
    auto res = Sellp::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_slice_sets(), 0);
    ASSERT_FALSE(res->get_size());
}


template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Dense<ValueType>> ref_permute(
    gko::matrix::Dense<ValueType>* input,
    gko::matrix::Permutation<IndexType>* permutation,
    gko::matrix::permute_mode mode)
{
    using gko::matrix::permute_mode;
    auto result = input->clone();
    auto permutation_dense =
        gko::matrix::Dense<double>::create(input->get_executor());
    gko::matrix_data<double, IndexType> permutation_data;
    if ((mode & permute_mode::inverse) == permute_mode::inverse) {
        permutation->compute_inverse()->write(permutation_data);
    } else {
        permutation->write(permutation_data);
    }
    permutation_dense->read(permutation_data);
    if ((mode & permute_mode::rows) == permute_mode::rows) {
        // compute P * A
        permutation_dense->apply(input, result);
    }
    if ((mode & permute_mode::columns) == permute_mode::columns) {
        // compute A * P^T = (P * A^T)^T
        auto tmp = result->transpose();
        auto tmp2 = gko::as<gko::matrix::Dense<ValueType>>(tmp->clone());
        permutation_dense->apply(tmp, tmp2);
        tmp2->transpose(result);
    }
    return result;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Dense<ValueType>> ref_permute(
    gko::matrix::Dense<ValueType>* input,
    gko::matrix::Permutation<IndexType>* row_permutation,
    gko::matrix::Permutation<IndexType>* col_permutation, bool invert)
{
    using gko::matrix::permute_mode;
    auto result = input->clone();
    auto row_permutation_dense =
        gko::matrix::Dense<double>::create(input->get_executor());
    auto col_permutation_dense =
        gko::matrix::Dense<double>::create(input->get_executor());
    gko::matrix_data<double, IndexType> row_permutation_data;
    gko::matrix_data<double, IndexType> col_permutation_data;
    if (invert) {
        row_permutation->compute_inverse()->write(row_permutation_data);
        col_permutation->compute_inverse()->write(col_permutation_data);
    } else {
        row_permutation->write(row_permutation_data);
        col_permutation->write(col_permutation_data);
    }
    row_permutation_dense->read(row_permutation_data);
    col_permutation_dense->read(col_permutation_data);
    row_permutation_dense->apply(input, result);
    auto tmp = result->transpose();
    auto tmp2 = gko::as<gko::matrix::Dense<ValueType>>(tmp->clone());
    col_permutation_dense->apply(tmp, tmp2);
    tmp2->transpose(result);
    return result;
}


TYPED_TEST(DenseWithIndexType, Permute)
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


TYPED_TEST(DenseWithIndexType, PermuteRoundtrip)
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


TYPED_TEST(DenseWithIndexType, PermuteStridedIntoDense)
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


TYPED_TEST(DenseWithIndexType, PermuteRectangular)
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


TYPED_TEST(DenseWithIndexType, PermuteFailsWithIncorrectPermutationSize)
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


TYPED_TEST(DenseWithIndexType, PermuteFailsWithIncorrectOutputSize)
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


TYPED_TEST(DenseWithIndexType, NonsymmPermute)
{
    auto permuted = this->mtx5->permute(this->perm3, this->perm3_rev);
    auto ref_permuted = ref_permute(this->mtx5.get(), this->perm3.get(),
                                    this->perm3_rev.get(), false);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(DenseWithIndexType, NonsymmPermuteInverse)
{
    auto permuted = this->mtx5->permute(this->perm3, this->perm3_rev, true);
    auto ref_permuted = ref_permute(this->mtx5.get(), this->perm3.get(),
                                    this->perm3_rev.get(), true);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(DenseWithIndexType, NonsymmPermuteRectangular)
{
    auto permuted = this->mtx1->permute(this->perm2, this->perm3);
    auto ref_permuted = ref_permute(this->mtx1.get(), this->perm2.get(),
                                    this->perm3.get(), false);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(DenseWithIndexType, NonsymmPermuteInverseRectangular)
{
    auto permuted = this->mtx1->permute(this->perm2, this->perm3, true);
    auto ref_permuted = ref_permute(this->mtx1.get(), this->perm2.get(),
                                    this->perm3.get(), true);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(DenseWithIndexType, NonsymmPermuteRoundtrip)
{
    auto permuted = this->mtx5->permute(this->perm3, this->perm3_rev)
                        ->permute(this->perm3, this->perm3_rev, true);

    GKO_ASSERT_MTX_NEAR(this->mtx5, permuted, 0.0);
}


TYPED_TEST(DenseWithIndexType, NonsymmPermuteInverseInverted)
{
    auto inv_permuted = this->mtx5->permute(this->perm3, this->perm3_rev, true);
    auto preinv_permuted = this->mtx5->permute(this->perm3_rev, this->perm3);

    GKO_ASSERT_MTX_NEAR(inv_permuted, preinv_permuted, 0.0);
}


TYPED_TEST(DenseWithIndexType, NonsymmPermuteStridedIntoDense)
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


TYPED_TEST(DenseWithIndexType, NonsymmPermuteInverseStridedIntoDense)
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


TYPED_TEST(DenseWithIndexType, NonsymmPermuteFailsWithIncorrectPermutationSize)
{
    ASSERT_THROW(this->mtx5->permute(this->perm0, this->perm3_rev),
                 gko::DimensionMismatch);
    ASSERT_THROW(this->mtx5->permute(this->perm3_rev, this->perm0),
                 gko::DimensionMismatch);
    ASSERT_THROW(this->mtx5->permute(this->perm0, this->perm0),
                 gko::DimensionMismatch);
}


TYPED_TEST(DenseWithIndexType, SquareMatrixCanGatherRows)
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


TYPED_TEST(DenseWithIndexType, SquareMatrixCanGatherRowsIntoDense)
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


TYPED_TEST(DenseWithIndexType, SquareSubmatrixCanGatherRowsIntoDense)
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


TYPED_TEST(DenseWithIndexType, NonSquareSubmatrixCanGatherRowsIntoMixedDense)
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


TYPED_TEST(DenseWithIndexType,
           NonSquareSubmatrixCanAdvancedGatherRowsIntoMixedDense)
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


TYPED_TEST(DenseWithIndexType,
           SquareMatrixGatherRowsIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 0}};

    ASSERT_THROW(this->mtx5->row_gather(&permute_idxs, Mtx::create(exec)),
                 gko::DimensionMismatch);
}


TYPED_TEST(DenseWithIndexType, SquareMatrixIsPermutable)
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


TYPED_TEST(DenseWithIndexType, SquareMatrixIsPermutableIntoDense)
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


TYPED_TEST(DenseWithIndexType, SquareSubmatrixIsPermutableIntoDense)
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


TYPED_TEST(DenseWithIndexType, NonSquareMatrixPermuteIntoDenseFails)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx4->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(this->mtx4->permute(&permute_idxs, this->mtx4->clone()),
                 gko::DimensionMismatch);
}


TYPED_TEST(DenseWithIndexType,
           SquareMatrixPermuteIntoDenseFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2}};

    ASSERT_THROW(this->mtx5->permute(&permute_idxs, this->mtx5->clone()),
                 gko::DimensionMismatch);
}


TYPED_TEST(DenseWithIndexType,
           SquareMatrixPermuteIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(this->mtx5->permute(&permute_idxs, Mtx::create(exec)),
                 gko::DimensionMismatch);
}


TYPED_TEST(DenseWithIndexType, SquareMatrixIsInversePermutable)
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


TYPED_TEST(DenseWithIndexType, SquareMatrixIsInversePermutableIntoDense)
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


TYPED_TEST(DenseWithIndexType, SquareSubmatrixIsInversePermutableIntoDense)
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


TYPED_TEST(DenseWithIndexType, NonSquareMatrixInversePermuteIntoDenseFails)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx4->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(
        this->mtx4->inverse_permute(&permute_idxs, this->mtx4->clone()),
        gko::DimensionMismatch);
}


TYPED_TEST(DenseWithIndexType,
           SquareMatrixInversePermuteIntoDenseFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {0, 1}};

    ASSERT_THROW(
        this->mtx5->inverse_permute(&permute_idxs, this->mtx5->clone()),
        gko::DimensionMismatch);
}


TYPED_TEST(DenseWithIndexType,
           SquareMatrixInversePermuteIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(this->mtx5->inverse_permute(&permute_idxs, Mtx::create(exec)),
                 gko::DimensionMismatch);
}


TYPED_TEST(DenseWithIndexType, SquareMatrixIsRowPermutable)
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


TYPED_TEST(DenseWithIndexType, NonSquareMatrixIsRowPermutable)
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


TYPED_TEST(DenseWithIndexType, SquareMatrixIsRowPermutableIntoDense)
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


TYPED_TEST(DenseWithIndexType, SquareSubmatrixIsRowPermutableIntoDense)
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


TYPED_TEST(DenseWithIndexType,
           SquareMatrixRowPermuteIntoDenseFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    ASSERT_THROW(this->mtx5->row_permute(&permute_idxs, permuted),
                 gko::DimensionMismatch);
}


TYPED_TEST(DenseWithIndexType,
           SquareMatrixRowPermuteIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(this->mtx5->row_permute(&permute_idxs, Mtx::create(exec)),
                 gko::DimensionMismatch);
}


TYPED_TEST(DenseWithIndexType, SquareMatrixIsColPermutable)
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


TYPED_TEST(DenseWithIndexType, NonSquareMatrixIsColPermutable)
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


TYPED_TEST(DenseWithIndexType, SquareMatrixIsColPermutableIntoDense)
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


TYPED_TEST(DenseWithIndexType, SquareSubmatrixIsColPermutableIntoDense)
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


TYPED_TEST(DenseWithIndexType,
           SquareMatrixColPermuteIntoDenseFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    ASSERT_THROW(this->mtx5->column_permute(&permute_idxs, permuted),
                 gko::DimensionMismatch);
}


TYPED_TEST(DenseWithIndexType,
           SquareMatrixColPermuteIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(this->mtx5->column_permute(&permute_idxs, Mtx::create(exec)),
                 gko::DimensionMismatch);
}


TYPED_TEST(DenseWithIndexType, SquareMatrixIsInverseRowPermutable)
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


TYPED_TEST(DenseWithIndexType, NonSquareMatrixIsInverseRowPermutable)
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


TYPED_TEST(DenseWithIndexType, SquareMatrixIsInverseRowPermutableIntoDense)
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


TYPED_TEST(DenseWithIndexType, SquareSubmatrixIsInverseRowPermutableIntoDense)
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


TYPED_TEST(DenseWithIndexType,
           SquareMatrixInverseRowPermuteIntoDenseFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    ASSERT_THROW(this->mtx5->inverse_row_permute(&permute_idxs, permuted),
                 gko::DimensionMismatch);
}


TYPED_TEST(DenseWithIndexType,
           SquareMatrixInverseRowPermuteIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(
        this->mtx5->inverse_row_permute(&permute_idxs, Mtx::create(exec)),
        gko::DimensionMismatch);
}


TYPED_TEST(DenseWithIndexType, SquareMatrixIsInverseColPermutable)
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


TYPED_TEST(DenseWithIndexType, NonSquareMatrixIsInverseColPermutable)
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


TYPED_TEST(DenseWithIndexType, SquareMatrixIsInverseColPermutableIntoDense)
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


TYPED_TEST(DenseWithIndexType, SquareSubmatrixIsInverseColPermutableIntoDense)
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


TYPED_TEST(DenseWithIndexType,
           SquareMatrixInverseColPermuteIntoDenseFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto exec = this->mtx5->get_executor();
    gko::array<index_type> permute_idxs{exec, {1, 2}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    ASSERT_THROW(this->mtx5->inverse_column_permute(&permute_idxs, permuted),
                 gko::DimensionMismatch);
}


TYPED_TEST(DenseWithIndexType,
           SquareMatrixInverseColPermuteIntoDenseFailsForWrongDimensions)
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
std::unique_ptr<gko::matrix::Dense<ValueType>> ref_scaled_permute(
    gko::matrix::Dense<ValueType>* input,
    gko::matrix::ScaledPermutation<ValueType, IndexType>* permutation,
    gko::matrix::permute_mode mode)
{
    using gko::matrix::permute_mode;
    auto result = input->clone();
    auto permutation_dense =
        gko::matrix::Dense<ValueType>::create(input->get_executor());
    gko::matrix_data<ValueType, IndexType> permutation_data;
    if ((mode & permute_mode::inverse) == permute_mode::inverse) {
        permutation->compute_inverse()->write(permutation_data);
    } else {
        permutation->write(permutation_data);
    }
    permutation_dense->read(permutation_data);
    if ((mode & permute_mode::rows) == permute_mode::rows) {
        // compute P * A
        permutation_dense->apply(input, result);
    }
    if ((mode & permute_mode::columns) == permute_mode::columns) {
        // compute A * P^T = (P * A^T)^T
        auto tmp = result->transpose();
        auto tmp2 = gko::as<gko::matrix::Dense<ValueType>>(tmp->clone());
        permutation_dense->apply(tmp, tmp2);
        tmp2->transpose(result);
    }
    return result;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Dense<ValueType>> ref_scaled_permute(
    gko::matrix::Dense<ValueType>* input,
    gko::matrix::ScaledPermutation<ValueType, IndexType>* row_permutation,
    gko::matrix::ScaledPermutation<ValueType, IndexType>* col_permutation,
    bool invert)
{
    using gko::matrix::permute_mode;
    auto result = input->clone();
    auto row_permutation_dense =
        gko::matrix::Dense<ValueType>::create(input->get_executor());
    auto col_permutation_dense =
        gko::matrix::Dense<ValueType>::create(input->get_executor());
    gko::matrix_data<ValueType, IndexType> row_permutation_data;
    gko::matrix_data<ValueType, IndexType> col_permutation_data;
    if (invert) {
        row_permutation->compute_inverse()->write(row_permutation_data);
        col_permutation->compute_inverse()->write(col_permutation_data);
    } else {
        row_permutation->write(row_permutation_data);
        col_permutation->write(col_permutation_data);
    }
    row_permutation_dense->read(row_permutation_data);
    col_permutation_dense->read(col_permutation_data);
    row_permutation_dense->apply(input, result);
    auto tmp = result->transpose();
    auto tmp2 = gko::as<gko::matrix::Dense<ValueType>>(tmp->clone());
    col_permutation_dense->apply(tmp, tmp2);
    tmp2->transpose(result);
    return result;
}


TYPED_TEST(DenseWithIndexType, ScaledPermute)
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


TYPED_TEST(DenseWithIndexType, ScaledPermuteRoundtrip)
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


TYPED_TEST(DenseWithIndexType, ScaledPermuteStridedIntoDense)
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


TYPED_TEST(DenseWithIndexType, ScaledPermuteRectangular)
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


TYPED_TEST(DenseWithIndexType, ScaledPermuteFailsWithIncorrectPermutationSize)
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


TYPED_TEST(DenseWithIndexType, ScaledPermuteFailsWithIncorrectOutputSize)
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


TYPED_TEST(DenseWithIndexType, NonsymmScaledPermute)
{
    using value_type = typename TestFixture::value_type;

    auto permuted =
        this->mtx5->scale_permute(this->scale_perm3, this->scale_perm3_rev);
    auto ref_permuted =
        ref_scaled_permute(this->mtx5.get(), this->scale_perm3.get(),
                           this->scale_perm3_rev.get(), false);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
}


TYPED_TEST(DenseWithIndexType, NonsymmScaledPermuteInverse)
{
    using value_type = typename TestFixture::value_type;

    auto permuted = this->mtx5->scale_permute(this->scale_perm3,
                                              this->scale_perm3_rev, true);
    auto ref_permuted =
        ref_scaled_permute(this->mtx5.get(), this->scale_perm3.get(),
                           this->scale_perm3_rev.get(), true);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
}


TYPED_TEST(DenseWithIndexType, NonsymmScaledPermuteRectangular)
{
    using value_type = typename TestFixture::value_type;

    auto permuted =
        this->mtx1->scale_permute(this->scale_perm2, this->scale_perm3);
    auto ref_permuted =
        ref_scaled_permute(this->mtx1.get(), this->scale_perm2.get(),
                           this->scale_perm3.get(), false);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
}


TYPED_TEST(DenseWithIndexType, NonsymmScaledPermuteInverseRectangular)
{
    using value_type = typename TestFixture::value_type;

    auto permuted =
        this->mtx1->scale_permute(this->scale_perm2, this->scale_perm3, true);
    auto ref_permuted =
        ref_scaled_permute(this->mtx1.get(), this->scale_perm2.get(),
                           this->scale_perm3.get(), true);

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, r<value_type>::value);
}


TYPED_TEST(DenseWithIndexType, NonsymmScaledPermuteRoundtrip)
{
    using value_type = typename TestFixture::value_type;

    auto permuted =
        this->mtx5->scale_permute(this->scale_perm3, this->scale_perm3_rev)
            ->scale_permute(this->scale_perm3, this->scale_perm3_rev, true);

    GKO_ASSERT_MTX_NEAR(this->mtx5, permuted, r<value_type>::value);
}


TYPED_TEST(DenseWithIndexType, NonsymmScaledPermuteInverseInverted)
{
    using value_type = typename TestFixture::value_type;

    auto inv_permuted = this->mtx5->scale_permute(this->scale_perm3,
                                                  this->scale_perm3_rev, true);
    auto preinv_permuted =
        this->mtx5->scale_permute(this->scale_perm3->compute_inverse(),
                                  this->scale_perm3_rev->compute_inverse());

    GKO_ASSERT_MTX_NEAR(inv_permuted, preinv_permuted, r<value_type>::value);
}

TYPED_TEST(DenseWithIndexType, NonsymmScaledPermuteStridedIntoDense)
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


TYPED_TEST(DenseWithIndexType, NonsymmScaledPermuteInverseStridedIntoDense)
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


TYPED_TEST(DenseWithIndexType, NonsymmScaledPermuteFailsWithIncorrectOutputSize)
{
    ASSERT_THROW(
        this->mtx5->scale_permute(this->scale_perm3, this->scale_perm3,
                                  TestFixture::Mtx::create(this->exec)),
        gko::DimensionMismatch);
}


TYPED_TEST(DenseWithIndexType,
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
class DenseComplex : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using RealMtx = gko::matrix::Dense<gko::remove_complex<value_type>>;
};


TYPED_TEST_SUITE(DenseComplex, gko::test::ComplexValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(DenseComplex, ScalesWithRealScalar)
{
    using Dense = typename TestFixture::Mtx;
    using RealDense = gko::remove_complex<Dense>;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto alpha =
        gko::initialize<RealDense>({gko::remove_complex<T>{-2.0}}, exec);

    mtx->scale(alpha);

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-2.0, -4.0}, T{2.0, -4.5}},
                              {T{4.0, -3.0}, T{-9.0, 0.0}},
                              {T{-2.0, 0.0}, T{0.0, -2.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, ScalesWithRealVector)
{
    using Dense = typename TestFixture::Mtx;
    using RealDense = gko::remove_complex<Dense>;
    using T = typename TestFixture::value_type;
    using RealT = gko::remove_complex<T>;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto alpha = gko::initialize<RealDense>({{RealT{-2.0}, RealT{4.0}}}, exec);

    mtx->scale(alpha);

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-2.0, -4.0}, T{-4.0, 9.0}},
                              {T{4.0, -3.0}, T{18.0, 0.0}},
                              {T{-2.0, 0.0}, T{0.0, 4.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, InvScalesWithRealScalar)
{
    using Dense = typename TestFixture::Mtx;
    using RealDense = gko::remove_complex<Dense>;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto alpha =
        gko::initialize<RealDense>({gko::remove_complex<T>{-0.5}}, exec);

    mtx->inv_scale(alpha);

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-2.0, -4.0}, T{2.0, -4.5}},
                              {T{4.0, -3.0}, T{-9.0, 0.0}},
                              {T{-2.0, 0.0}, T{0.0, -2.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, InvScalesWithRealVector)
{
    using Dense = typename TestFixture::Mtx;
    using RealDense = gko::remove_complex<Dense>;
    using T = typename TestFixture::value_type;
    using RealT = gko::remove_complex<T>;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto alpha = gko::initialize<RealDense>({{RealT{-0.5}, RealT{0.25}}}, exec);

    mtx->inv_scale(alpha);

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-2.0, -4.0}, T{-4.0, 9.0}},
                              {T{4.0, -3.0}, T{18.0, 0.0}},
                              {T{-2.0, 0.0}, T{0.0, 4.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, AddsScaledWithRealScalar)
{
    using Dense = typename TestFixture::Mtx;
    using RealDense = gko::remove_complex<Dense>;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto mtx2 = gko::initialize<Dense>({{T{4.0, -1.0}, T{5.0, 1.5}},
                                        {T{3.0, 1.0}, T{0.0, 2.0}},
                                        {T{-1.0, 1.0}, T{0.5, -2.0}}},
                                       exec);
    auto alpha =
        gko::initialize<RealDense>({gko::remove_complex<T>{-2.0}}, exec);

    mtx->add_scaled(alpha, mtx2);

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-7.0, 4.0}, T{-11.0, -0.75}},
                              {T{-8.0, -0.5}, T{4.5, -4.0}},
                              {T{3.0, -2.0}, T{-1.0, 5.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, AddsScaledWithRealVector)
{
    using Dense = typename TestFixture::Mtx;
    using RealDense = gko::remove_complex<Dense>;
    using T = typename TestFixture::value_type;
    using RealT = gko::remove_complex<T>;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto mtx2 = gko::initialize<Dense>({{T{4.0, -1.0}, T{5.0, 1.5}},
                                        {T{3.0, 1.0}, T{0.0, 2.0}},
                                        {T{-1.0, 1.0}, T{0.5, -2.0}}},
                                       exec);
    auto alpha = gko::initialize<RealDense>({{RealT{-2.0}, RealT{4.0}}}, exec);

    mtx->add_scaled(alpha, mtx2);

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-7.0, 4.0}, T{19.0, 8.25}},
                              {T{-8.0, -0.5}, T{4.5, 8.0}},
                              {T{3.0, -2.0}, T{2.0, -7.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, SubtractsScaledWithRealScalar)
{
    using Dense = typename TestFixture::Mtx;
    using RealDense = gko::remove_complex<Dense>;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto mtx2 = gko::initialize<Dense>({{T{4.0, -1.0}, T{5.0, 1.5}},
                                        {T{3.0, 1.0}, T{0.0, 2.0}},
                                        {T{-1.0, 1.0}, T{0.5, -2.0}}},
                                       exec);
    auto alpha =
        gko::initialize<RealDense>({gko::remove_complex<T>{2.0}}, exec);

    mtx->sub_scaled(alpha, mtx2);

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-7.0, 4.0}, T{-11.0, -0.75}},
                              {T{-8.0, -0.5}, T{4.5, -4.0}},
                              {T{3.0, -2.0}, T{-1.0, 5.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, SubtractsScaledWithRealVector)
{
    using Dense = typename TestFixture::Mtx;
    using RealDense = gko::remove_complex<Dense>;
    using T = typename TestFixture::value_type;
    using RealT = gko::remove_complex<T>;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto mtx2 = gko::initialize<Dense>({{T{4.0, -1.0}, T{5.0, 1.5}},
                                        {T{3.0, 1.0}, T{0.0, 2.0}},
                                        {T{-1.0, 1.0}, T{0.5, -2.0}}},
                                       exec);
    auto alpha = gko::initialize<RealDense>({{RealT{2.0}, RealT{-4.0}}}, exec);

    mtx->sub_scaled(alpha, mtx2);

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-7.0, 4.0}, T{19.0, 8.25}},
                              {T{-8.0, -0.5}, T{4.5, 8.0}},
                              {T{3.0, -2.0}, T{2.0, -7.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, NonSquareMatrixIsConjugateTransposable)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.1}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);

    auto trans = gko::as<Dense>(mtx->conj_transpose());

    GKO_ASSERT_MTX_NEAR(trans,
                        l<T>({{T{1.0, -2.0}, T{-2.0, -1.5}, T{1.0, 0.0}},
                              {T{-1.0, -2.1}, T{4.5, 0.0}, T{0.0, -1.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, NonSquareMatrixIsConjugateTransposableIntoDense)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.1}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto trans = Dense::create(exec, gko::transpose(mtx->get_size()));

    mtx->conj_transpose(trans);

    GKO_ASSERT_MTX_NEAR(trans,
                        l<T>({{T{1.0, -2.0}, T{-2.0, -1.5}, T{1.0, 0.0}},
                              {T{-1.0, -2.1}, T{4.5, 0.0}, T{0.0, -1.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, InplaceAbsolute)
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


TYPED_TEST(DenseComplex, OutplaceAbsolute)
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


TYPED_TEST(DenseComplex, OutplaceAbsoluteIntoDense)
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


TYPED_TEST(DenseComplex, MakeComplex)
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


TYPED_TEST(DenseComplex, MakeComplexIntoDense)
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


TYPED_TEST(DenseComplex, GetReal)
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


TYPED_TEST(DenseComplex, GetRealIntoDense)
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


TYPED_TEST(DenseComplex, GetImag)
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


TYPED_TEST(DenseComplex, GetImagIntoDense)
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


TYPED_TEST(DenseComplex, Dot)
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


TYPED_TEST(DenseComplex, ConjDot)
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
