// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/dense_kernels.hpp"

#include <complex>
#include <memory>
#include <numeric>
#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename T>
class Dense : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Vec = gko::matrix::MultiVector<value_type>;
    using MixedVec = gko::matrix::MultiVector<gko::next_precision<value_type>>;
    using ComplexMtx = gko::to_complex<Mtx>;
    using RealMtx = gko::remove_complex<Mtx>;

    Dense() : exec(gko::ReferenceExecutor::create()) {}

    void SetUp() override
    {
        mtx1 =
            gko::initialize<Mtx>(4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}}, exec);
        mtx2 =
            gko::initialize<Mtx>({I<T>({1.0, -1.0}), I<T>({-2.0, 2.0})}, exec);
        mtx3 =
            gko::initialize<Mtx>(4, {{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}, exec);
        mtx4 = gko::initialize<Mtx>(
            {{1.0, -1.0, -0.5}, {-2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}, exec);
        mtx5 = gko::initialize<Mtx>({{1.0, 2.0, 0.0}, {0.0, 1.5, 0.0}}, exec);
        mtx6 = gko::initialize<Mtx>({{1.0, 2.0, 3.0}, {0.0, 1.5, 0.0}}, exec);
        mtx7 = gko::initialize<Mtx>(
            {I<T>({1.0, -1.0}), I<T>({-2.0, 2.0}), I<T>({-3.0, 3.0})}, exec);
        vec1 =
            gko::initialize<Vec>(4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}}, exec);
        vec2 =
            gko::initialize<Vec>(4, {{1.0, 2.0, 3.0}, {0.5, 1.5, 2.5}}, exec);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx1;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<Mtx> mtx3;
    std::unique_ptr<Mtx> mtx4;
    std::unique_ptr<Mtx> mtx5;
    std::unique_ptr<Mtx> mtx6;
    std::unique_ptr<Mtx> mtx7;
    std::unique_ptr<Vec> vec1;
    std::unique_ptr<Vec> vec2;
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


TYPED_TEST(Dense, CanBeFilledWithValue)
{
    using value_type = typename TestFixture::value_type;
    auto m =
        gko::initialize<gko::matrix::Dense<TypeParam>>({1.0, 2.0}, this->exec);
    EXPECT_EQ(m->at(0, 0), value_type{1});
    EXPECT_EQ(m->at(0, 1), value_type{2});

    m->fill(value_type{42});

    EXPECT_EQ(m->at(0, 0), value_type{42});
    EXPECT_EQ(m->at(0, 1), value_type{42});
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
    EXPECT_EQ(m->at(0, 0), value_type{1.0});
    EXPECT_EQ(m->at(0, 1), value_type{2.0});
    EXPECT_EQ(m->at(1, 0), value_type{3.0});
    EXPECT_EQ(m->at(1, 1), value_type{4.0});
    EXPECT_EQ(m->at(2, 0), value_type{5.0});
    EXPECT_EQ(m->at(2, 1), value_type{6.0});

    m->fill(value_type{42});

    ASSERT_EQ(m->get_size(), gko::dim<2>(3, 2));
    EXPECT_EQ(m->get_num_stored_elements(), 12);
    EXPECT_EQ(m->at(0, 0), value_type{42.0});
    EXPECT_EQ(m->at(0, 1), value_type{42.0});
    EXPECT_EQ(m->at(1, 0), value_type{42.0});
    EXPECT_EQ(m->at(1, 1), value_type{42.0});
    EXPECT_EQ(m->at(2, 0), value_type{42.0});
    EXPECT_EQ(m->at(2, 1), value_type{42.0});
    ASSERT_EQ(m->get_values()[3], in_stride);
}


TYPED_TEST(Dense, AppliesToDense)
{
    using T = typename TestFixture::value_type;
    T in_stride{-1};
    this->vec2->get_values()[3] = in_stride;

    this->mtx2->apply(this->vec1, this->vec2);

    EXPECT_EQ(this->vec2->at(0, 0), T{-0.5});
    EXPECT_EQ(this->vec2->at(0, 1), T{-0.5});
    EXPECT_EQ(this->vec2->at(0, 2), T{-0.5});
    EXPECT_EQ(this->vec2->at(1, 0), T{1.0});
    EXPECT_EQ(this->vec2->at(1, 1), T{1.0});
    EXPECT_EQ(this->vec2->at(1, 2), T{1.0});
    ASSERT_EQ(this->vec2->get_values()[3], in_stride);
}


TYPED_TEST(Dense, AppliesToMixedDense)
{
    using MixedMtx = typename TestFixture::MixedVec;
    using MixedT = typename MixedMtx::value_type;
    auto mvec1 = MixedMtx::create(this->exec);
    auto mvec2 = MixedMtx::create(this->exec);
    this->vec1->convert_to(mvec1);
    this->vec2->convert_to(mvec2);

    this->mtx2->apply(mvec1, mvec2);

    EXPECT_EQ(mvec2->at(0, 0), MixedT{-0.5});
    EXPECT_EQ(mvec2->at(0, 1), MixedT{-0.5});
    EXPECT_EQ(mvec2->at(0, 2), MixedT{-0.5});
    EXPECT_EQ(mvec2->at(1, 0), MixedT{1.0});
    EXPECT_EQ(mvec2->at(1, 1), MixedT{1.0});
    ASSERT_EQ(mvec2->at(1, 2), MixedT{1.0});
}


TYPED_TEST(Dense, AppliesLinearCombinationToDense)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    T in_stride{-1};
    this->vec2->get_values()[3] = in_stride;

    this->mtx2->apply(alpha, this->vec1, beta, this->vec2);

    EXPECT_EQ(this->vec2->at(0, 0), T{2.5});
    EXPECT_EQ(this->vec2->at(0, 1), T{4.5});
    EXPECT_EQ(this->vec2->at(0, 2), T{6.5});
    EXPECT_EQ(this->vec2->at(1, 0), T{0.0});
    EXPECT_EQ(this->vec2->at(1, 1), T{2.0});
    EXPECT_EQ(this->vec2->at(1, 2), T{4.0});
    ASSERT_EQ(this->vec2->get_values()[3], in_stride);
}


TYPED_TEST(Dense, AppliesLinearCombinationToDenseWithZeroBetaNan)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({0.0}, this->exec);
    this->vec2->fill(gko::nan<T>());

    this->mtx2->apply(alpha, this->vec1, beta, this->vec2);

    EXPECT_EQ(this->vec2->at(0, 0), T{0.5});
    EXPECT_EQ(this->vec2->at(0, 1), T{0.5});
    EXPECT_EQ(this->vec2->at(0, 2), T{0.5});
    EXPECT_EQ(this->vec2->at(1, 0), T{-1.0});
    EXPECT_EQ(this->vec2->at(1, 1), T{-1.0});
    EXPECT_EQ(this->vec2->at(1, 2), T{-1.0});
}


TYPED_TEST(Dense, AppliesLinearCombinationToMixedDense)
{
    using MixedVec = typename TestFixture::MixedVec;
    using MixedT = typename MixedVec::value_type;
    auto mvec1 = MixedVec::create(this->exec);
    auto mvec2 = MixedVec::create(this->exec);
    this->vec1->convert_to(mvec1);
    this->vec2->convert_to(mvec2);
    auto alpha = gko::initialize<MixedVec>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedVec>({2.0}, this->exec);

    this->mtx2->apply(alpha, mvec1, beta, mvec2);

    EXPECT_EQ(mvec2->at(0, 0), MixedT{2.5});
    EXPECT_EQ(mvec2->at(0, 1), MixedT{4.5});
    EXPECT_EQ(mvec2->at(0, 2), MixedT{6.5});
    EXPECT_EQ(mvec2->at(1, 0), MixedT{0.0});
    EXPECT_EQ(mvec2->at(1, 1), MixedT{2.0});
    ASSERT_EQ(mvec2->at(1, 2), MixedT{4.0});
}


GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS


TYPED_TEST(Dense, AppliesToDenseDeprecated)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto mtx2 = Mtx::create(this->exec, this->vec2->get_size(),
                            this->vec2->get_stride());
    this->vec2->as_const_dense_view()->convert_to(mtx2);
    T in_stride{-1};
    mtx2->get_values()[3] = in_stride;

    this->mtx2->apply(this->mtx1, mtx2);

    EXPECT_EQ(mtx2->at(0, 0), T{-0.5});
    EXPECT_EQ(mtx2->at(0, 1), T{-0.5});
    EXPECT_EQ(mtx2->at(0, 2), T{-0.5});
    EXPECT_EQ(mtx2->at(1, 0), T{1.0});
    EXPECT_EQ(mtx2->at(1, 1), T{1.0});
    EXPECT_EQ(mtx2->at(1, 2), T{1.0});
    ASSERT_EQ(mtx2->get_values()[3], in_stride);
}


TYPED_TEST(Dense, AppliesLinearCombinationToDenseDeprecated)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto mtx2 = Mtx::create(this->exec, this->vec2->get_size(),
                            this->vec2->get_stride());
    this->vec2->as_const_dense_view()->convert_to(mtx2);
    auto alpha = gko::initialize<Mtx>({-1.0}, this->exec);
    auto beta = gko::initialize<Mtx>({2.0}, this->exec);
    T in_stride{-1};
    mtx2->get_values()[3] = in_stride;

    this->mtx2->apply(alpha, this->mtx1, beta, mtx2);

    EXPECT_EQ(mtx2->at(0, 0), T{2.5});
    EXPECT_EQ(mtx2->at(0, 1), T{4.5});
    EXPECT_EQ(mtx2->at(0, 2), T{6.5});
    EXPECT_EQ(mtx2->at(1, 0), T{0.0});
    EXPECT_EQ(mtx2->at(1, 1), T{2.0});
    EXPECT_EQ(mtx2->at(1, 2), T{4.0});
    ASSERT_EQ(mtx2->get_values()[3], in_stride);
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


GKO_END_DISABLE_DEPRECATION_WARNINGS


TYPED_TEST(Dense, SquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = gko::as<Mtx>(this->mtx4->transpose());

    GKO_ASSERT_MTX_NEAR(
        trans, l<T>({{1.0, -2.0, 2.1}, {-1.0, 2.0, 3.4}, {-0.5, 4.5, 1.2}}),
        0.0);
}


TYPED_TEST(Dense, SquareMatrixIsTransposableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = Mtx::create(this->exec, this->mtx4->get_size());

    this->mtx4->transpose(trans);

    GKO_ASSERT_MTX_NEAR(
        trans, l<T>({{1.0, -2.0, 2.1}, {-1.0, 2.0, 3.4}, {-0.5, 4.5, 1.2}}),
        0.0);
}


TYPED_TEST(Dense, SquareSubmatrixIsTransposableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = Mtx::create(this->exec, gko::dim<2>{2, 2}, 4);

    this->mtx4->create_subview({0, 2}, {0, 2})->transpose(trans);

    GKO_ASSERT_MTX_NEAR(trans, l<T>({{1.0, -2.0}, {-1.0, 2.0}}), 0.0);
    ASSERT_EQ(trans->get_stride(), 4);
}


TYPED_TEST(Dense, SquareMatrixIsTransposableIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;

    ASSERT_THROW(this->mtx4->transpose(Mtx::create(this->exec)),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, NonSquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = gko::as<Mtx>(this->mtx3->transpose());

    GKO_ASSERT_MTX_NEAR(trans, l<T>({{1.0, 0.0}, {3.0, 5.0}, {2.0, 0.0}}), 0.0);
}


TYPED_TEST(Dense, NonSquareMatrixIsTransposableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans =
        Mtx::create(this->exec, gko::transpose(this->mtx3->get_size()));

    this->mtx3->transpose(trans);

    GKO_ASSERT_MTX_NEAR(trans, l<T>({{1.0, 0.0}, {3.0, 5.0}, {2.0, 0.0}}), 0.0);
}


TYPED_TEST(Dense, NonSquareSubmatrixIsTransposableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = Mtx::create(this->exec, gko::dim<2>{2, 1}, 5);

    this->mtx3->create_subview({0, 1}, {0, 2})->transpose(trans);

    GKO_ASSERT_MTX_NEAR(trans, l({1.0, 3.0}), 0.0);
    ASSERT_EQ(trans->get_stride(), 5);
}


TYPED_TEST(Dense, NonSquareMatrixIsTransposableIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;

    ASSERT_THROW(this->mtx3->transpose(Mtx::create(this->exec)),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ExtractsDiagonalFromSquareMatrix)
{
    using T = typename TestFixture::value_type;

    auto diag = this->mtx4->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 3);
    ASSERT_EQ(diag->get_size()[1], 3);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{2.});
    ASSERT_EQ(diag->get_values()[2], T{1.2});
}


TYPED_TEST(Dense, ExtractsDiagonalFromTallSkinnyMatrix)
{
    using T = typename TestFixture::value_type;

    auto diag = this->mtx3->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{5.});
}


TYPED_TEST(Dense, ExtractsDiagonalFromShortFatMatrix)
{
    using T = typename TestFixture::value_type;

    auto diag = this->mtx7->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{2.});
}


TYPED_TEST(Dense, ExtractsDiagonalFromSquareMatrixIntoDiagonal)
{
    using T = typename TestFixture::value_type;
    auto diag = gko::matrix::Diagonal<T>::create(this->exec, 3);

    this->mtx4->extract_diagonal(diag);

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

    this->mtx3->extract_diagonal(diag);

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{5.});
}


TYPED_TEST(Dense, ExtractsDiagonalFromShortFatMatrixIntoDiagonal)
{
    using T = typename TestFixture::value_type;
    auto diag = gko::matrix::Diagonal<T>::create(this->exec, 2);

    this->mtx7->extract_diagonal(diag);

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{2.});
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
    using Vector = gko::matrix::MultiVector<value_type>;
    using VectorComplex = gko::matrix::MultiVector<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    auto b = gko::initialize<VectorComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}},
        exec);
    auto x = gko::initialize<VectorComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}},
        exec);
    auto alpha = gko::initialize<Vector>({-1.0}, this->exec);
    auto beta = gko::initialize<Vector>({2.0}, this->exec);

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
    using MixedVector = gko::matrix::MultiVector<mixed_value_type>;
    using MixedVectorComplex = gko::matrix::MultiVector<mixed_complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    auto b = gko::initialize<MixedVectorComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}},
         {mixed_complex_type{3.0, 4.0}, mixed_complex_type{4.0, 5.0}}},
        exec);
    auto x = gko::initialize<MixedVectorComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}}},
        exec);
    auto alpha = gko::initialize<MixedVector>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedVector>({2.0}, this->exec);

    this->mtx1->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{mixed_complex_type{-12.0, -16.0}, mixed_complex_type{-16.0, -20.0}},
           {mixed_complex_type{-13.0, -15.0},
            mixed_complex_type{-18.5, -20.5}}}),
        0.0);
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
};

TYPED_TEST_SUITE(DenseWithIndexType, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


template <typename ValueType, typename IndexType>
void assert_coo_eq_mtx3(const gko::matrix::Coo<ValueType, IndexType>* coo_mtx)
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
    auto coo_mtx = Coo::create(this->mtx3->get_executor());

    this->mtx3->convert_to(coo_mtx);

    assert_coo_eq_mtx3(coo_mtx.get());
}


TYPED_TEST(DenseWithIndexType, MovesToCoo)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Coo = typename gko::matrix::Coo<value_type, index_type>;
    auto coo_mtx = Coo::create(this->mtx4->get_executor());

    this->mtx3->move_to(coo_mtx);

    assert_coo_eq_mtx3(coo_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_csr_eq_mtx3(const gko::matrix::Csr<ValueType, IndexType>* csr_mtx)
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
    auto csr_s_classical = gko::matrix::csr::spmv_strategy::classical;
    auto csr_s_merge = gko::matrix::csr::spmv_strategy::merge_path;
    auto csr_mtx_c = Csr::create(this->mtx3->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx3->get_executor(), csr_s_merge);

    this->mtx3->convert_to(csr_mtx_c);
    this->mtx3->convert_to(csr_mtx_m);

    assert_csr_eq_mtx3(csr_mtx_c.get());
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_c, csr_mtx_m, 0.0);
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(DenseWithIndexType, MovesToCsr)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Csr = typename gko::matrix::Csr<value_type, index_type>;
    auto csr_s_classical = gko::matrix::csr::spmv_strategy::classical;
    auto csr_s_merge = gko::matrix::csr::spmv_strategy::merge_path;
    auto csr_mtx_c = Csr::create(this->mtx3->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx3->get_executor(), csr_s_merge);
    auto mtx_clone = this->mtx3->clone();

    this->mtx3->move_to(csr_mtx_c);
    mtx_clone->move_to(csr_mtx_m);

    assert_csr_eq_mtx3(csr_mtx_c.get());
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_c, csr_mtx_m, 0.0);
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


template <typename ValueType, typename IndexType>
void assert_sparsity_csr_eq_mtx3(
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
    auto sparsity_csr_mtx = SparsityCsr::create(this->mtx3->get_executor());

    this->mtx3->convert_to(sparsity_csr_mtx);

    assert_sparsity_csr_eq_mtx3(sparsity_csr_mtx.get());
}


TYPED_TEST(DenseWithIndexType, MovesToSparsityCsr)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using SparsityCsr =
        typename gko::matrix::SparsityCsr<value_type, index_type>;
    auto sparsity_csr_mtx = SparsityCsr::create(this->mtx3->get_executor());

    this->mtx3->move_to(sparsity_csr_mtx);

    assert_sparsity_csr_eq_mtx3(sparsity_csr_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_ell_eq_mtx5(const gko::matrix::Ell<ValueType, IndexType>* ell_mtx)
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
    auto ell_mtx = Ell::create(this->mtx5->get_executor());

    this->mtx5->convert_to(ell_mtx);

    assert_ell_eq_mtx5(ell_mtx.get());
}


TYPED_TEST(DenseWithIndexType, MovesToEll)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Ell = typename gko::matrix::Ell<value_type, index_type>;
    auto ell_mtx = Ell::create(this->mtx5->get_executor());

    this->mtx5->move_to(ell_mtx);

    assert_ell_eq_mtx5(ell_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_strided_ell_eq_mtx5(
    const gko::matrix::Ell<ValueType, IndexType>* ell_mtx)
{
    constexpr auto invalid_index = gko::invalid_index<IndexType>();
    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 6);
    ASSERT_EQ(ell_mtx->get_stride(), 3);
    // only check the actual matrix entries.
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], invalid_index);
    EXPECT_EQ(v[0], ValueType{1.0});
    EXPECT_EQ(v[1], ValueType{1.5});
    EXPECT_EQ(v[3], ValueType{2.0});
    EXPECT_EQ(v[4], ValueType{0.0});
}


TYPED_TEST(DenseWithIndexType, ConvertsToEllWithStride)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Ell = typename gko::matrix::Ell<value_type, index_type>;
    auto ell_mtx =
        Ell::create(this->mtx5->get_executor(), gko::dim<2>{2, 3}, 2, 3);

    this->mtx5->convert_to(ell_mtx);

    assert_strided_ell_eq_mtx5(ell_mtx.get());
}


TYPED_TEST(DenseWithIndexType, MovesToEllWithStride)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Ell = typename gko::matrix::Ell<value_type, index_type>;
    auto ell_mtx =
        Ell::create(this->mtx5->get_executor(), gko::dim<2>{2, 3}, 2, 3);

    this->mtx5->move_to(ell_mtx);

    assert_strided_ell_eq_mtx5(ell_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_hybrid_auto_eq_mtx3(
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
    auto hybrid_mtx = Hybrid::create(this->mtx3->get_executor());

    this->mtx3->move_to(hybrid_mtx);

    assert_hybrid_auto_eq_mtx3(hybrid_mtx.get());
}


TYPED_TEST(DenseWithIndexType, ConvertsToHybridAutomatically)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Hybrid = typename gko::matrix::Hybrid<value_type, index_type>;
    auto hybrid_mtx = Hybrid::create(this->mtx3->get_executor());

    this->mtx3->convert_to(hybrid_mtx);

    assert_hybrid_auto_eq_mtx3(hybrid_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_hybrid_strided_eq_mtx3(
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
        Hybrid::create(this->mtx3->get_executor(), gko::dim<2>{2, 3}, 0, 3);

    this->mtx3->move_to(hybrid_mtx);

    assert_hybrid_strided_eq_mtx3(hybrid_mtx.get());
}


TYPED_TEST(DenseWithIndexType, ConvertsToHybridWithStrideAutomatically)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Hybrid = typename gko::matrix::Hybrid<value_type, index_type>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx3->get_executor(), gko::dim<2>{2, 3}, 0, 3);

    this->mtx3->convert_to(hybrid_mtx);

    assert_hybrid_strided_eq_mtx3(hybrid_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_hybrid_limited_eq_mtx3(
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
        Hybrid::create(this->mtx3->get_executor(), gko::dim<2>{2, 3}, 2, 3, 3,
                       std::make_shared<typename Hybrid::column_limit>(2));

    this->mtx3->move_to(hybrid_mtx);

    assert_hybrid_limited_eq_mtx3(hybrid_mtx.get());
}


TYPED_TEST(DenseWithIndexType, ConvertsToHybridWithStrideAndCooLengthByColumns2)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Hybrid = typename gko::matrix::Hybrid<value_type, index_type>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx3->get_executor(), gko::dim<2>{2, 3}, 2, 3, 3,
                       std::make_shared<typename Hybrid::column_limit>(2));

    this->mtx3->convert_to(hybrid_mtx);

    assert_hybrid_limited_eq_mtx3(hybrid_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_hybrid_percent_eq_mtx3(
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
        Hybrid::create(this->mtx3->get_executor(), gko::dim<2>{2, 3}, 1, 3,
                       std::make_shared<typename Hybrid::imbalance_limit>(0.4));

    this->mtx3->move_to(hybrid_mtx);

    assert_hybrid_percent_eq_mtx3(hybrid_mtx.get());
}


TYPED_TEST(DenseWithIndexType, ConvertsToHybridWithStrideByPercent40)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Hybrid = typename gko::matrix::Hybrid<value_type, index_type>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx3->get_executor(), gko::dim<2>{2, 3}, 1, 3,
                       std::make_shared<typename Hybrid::imbalance_limit>(0.4));

    this->mtx3->convert_to(hybrid_mtx);

    assert_hybrid_percent_eq_mtx3(hybrid_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_sellp_eq_mtx6(
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
    auto sellp_mtx = Sellp::create(this->mtx6->get_executor());

    this->mtx6->convert_to(sellp_mtx);

    assert_sellp_eq_mtx6(sellp_mtx.get());
}


TYPED_TEST(DenseWithIndexType, MovesToSellp)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Sellp = typename gko::matrix::Sellp<value_type, index_type>;
    auto sellp_mtx = Sellp::create(this->mtx6->get_executor());

    this->mtx6->move_to(sellp_mtx);

    assert_sellp_eq_mtx6(sellp_mtx.get());
}


template <typename ValueType, typename IndexType>
void assert_sellp_strided_eq_mtx6(
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
        Sellp::create(this->mtx6->get_executor(), gko::dim<2>{}, 2, 2, 0);

    this->mtx6->convert_to(sellp_mtx);

    assert_sellp_strided_eq_mtx6(sellp_mtx.get());
}


TYPED_TEST(DenseWithIndexType, MovesToSellpWithSliceSizeAndStrideFactor)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Sellp = typename gko::matrix::Sellp<value_type, index_type>;
    auto sellp_mtx =
        Sellp::create(this->mtx6->get_executor(), gko::dim<2>{}, 2, 2, 0);

    this->mtx6->move_to(sellp_mtx);

    assert_sellp_strided_eq_mtx6(sellp_mtx.get());
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


TYPED_TEST(DenseWithIndexType, ConvertsEmptyToCoo)
{
    using MultiVector = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Coo = typename gko::matrix::Coo<value_type, index_type>;
    auto empty = MultiVector::create(this->exec);
    auto res = Coo::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, MovesEmptyToCoo)
{
    using MultiVector = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Coo = typename gko::matrix::Coo<value_type, index_type>;
    auto empty = MultiVector::create(this->exec);
    auto res = Coo::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, ConvertsEmptyMatrixToCsr)
{
    using MultiVector = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Csr = typename gko::matrix::Csr<value_type, index_type>;
    auto empty = MultiVector::create(this->exec);
    auto res = Csr::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, MovesEmptyMatrixToCsr)
{
    using MultiVector = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Csr = typename gko::matrix::Csr<value_type, index_type>;
    auto empty = MultiVector::create(this->exec);
    auto res = Csr::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, ConvertsEmptyToSparsityCsr)
{
    using MultiVector = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using SparsityCsr =
        typename gko::matrix::SparsityCsr<value_type, index_type>;
    auto empty = MultiVector::create(this->exec);
    auto res = SparsityCsr::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_nonzeros(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, MovesEmptyToSparsityCsr)
{
    using MultiVector = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using SparsityCsr =
        typename gko::matrix::SparsityCsr<value_type, index_type>;
    auto empty = MultiVector::create(this->exec);
    auto res = SparsityCsr::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_nonzeros(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, ConvertsEmptyToEll)
{
    using MultiVector = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Ell = typename gko::matrix::Ell<value_type, index_type>;
    auto empty = MultiVector::create(this->exec);
    auto res = Ell::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, MovesEmptyToEll)
{
    using MultiVector = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Ell = typename gko::matrix::Ell<value_type, index_type>;
    auto empty = MultiVector::create(this->exec);
    auto res = Ell::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, ConvertsEmptyToHybrid)
{
    using MultiVector = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Hybrid = typename gko::matrix::Hybrid<value_type, index_type>;
    auto empty = MultiVector::create(this->exec);
    auto res = Hybrid::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, MovesEmptyToHybrid)
{
    using MultiVector = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Hybrid = typename gko::matrix::Hybrid<value_type, index_type>;
    auto empty = MultiVector::create(this->exec);
    auto res = Hybrid::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, ConvertsEmptyToSellp)
{
    using MultiVector = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Sellp = typename gko::matrix::Sellp<value_type, index_type>;
    auto empty = MultiVector::create(this->exec);
    auto res = Sellp::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_slice_sets(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(DenseWithIndexType, MovesEmptyToSellp)
{
    using MultiVector = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Sellp = typename gko::matrix::Sellp<value_type, index_type>;
    auto empty = MultiVector::create(this->exec);
    auto res = Sellp::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_slice_sets(), 0);
    ASSERT_FALSE(res->get_size());
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


}  // namespace
