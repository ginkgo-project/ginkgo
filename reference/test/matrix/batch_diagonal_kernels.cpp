// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/batch_diagonal.hpp>


// Copyright (c) 2017-2023, the Ginkgo authors
#include <complex>
#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/matrix/batch_diagonal_kernels.hpp"
#include "core/test/utils.hpp"


template <typename T>
class Diagonal : public ::testing::Test {
protected:
    using value_type = T;
    using size_type = gko::size_type;
    using BMtx = gko::batch::matrix::Diagonal<value_type>;
    using BMVec = gko::batch::MultiVector<value_type>;
    using DiagonalMtx = gko::matrix::Diagonal<value_type>;
    using DenseMtx = gko::matrix::Dense<value_type>;
    Diagonal()
        : exec(gko::ReferenceExecutor::create()),
          mtx_0(gko::batch::initialize<BMtx>(
              {
                  // clang-format off
                {{2.0, 0.0, 0.0},
                 {0.0, -1.0, 0.0},
                 {0.0, 0.0, 3.0}},
                {{4.0, 0.0, 0.0},
                 {0.0, -2.0, 0.0},
                 {0.0, 0.0, 0.5}},
                  // clang-format on
              },
              exec)),
          mtx_00(DiagonalMtx::create(exec, 3)),
          mtx_01(DiagonalMtx::create(exec, 3)),
          b_0(gko::batch::initialize<BMVec>(
              {
                  // clang-format off
                  {{1.0, 0.0, 1.0},
                   {2.0, 0.0, 1.0},
                   {1.0, 0.0, 2.0}},
                  {{-1.0, 1.0, 1.0},
                   {1.0, -1.0, 1.0},
                   {1.0, 0.0, 2.0}}
                  // clang-format on
              },
              exec)),
          b_00(gko::initialize<DenseMtx>(
              {
                  // clang-format off
                  {1.0, 0.0, 1.0},
                  {2.0, 0.0, 1.0},
                  {1.0, 0.0, 2.0},
                  // clang-format on
              },
              exec)),
          b_01(gko::initialize<DenseMtx>(
              {
                  // clang-format off
                  {-1.0, 1.0, 1.0},
                  {1.0, -1.0, 1.0},
                  {1.0, 0.0, 2.0}
                  // clang-format on
              },
              exec)),
          x_0(gko::batch::initialize<BMVec>(
              {
                  // clang-format off
                  {{2.0, 0.0, 1.0},
                   {2.0, 0.0, 2.0},
                   {2.0, 0.0, 2.0}},
                  {{-2.0, 1.0, 1.0},
                   {2.0, 0.0, 2.0},
                   {1.0, -1.0, -1.0}}
                  // clang-format on
              },
              exec)),
          x_00(gko::initialize<DenseMtx>(
              {
                  // clang-format off
                  {2.0, 0.0, 1.0},
                   {2.0, 0.0, 2.0},
                  {2.0, 0.0, 2.0}
                  // clang-format on
              },
              exec)),
          x_01(gko::initialize<DenseMtx>(
              {
                  // clang-format off
                  {-2.0, 1.0, 1.0},
                  {2.0, 0.0, 2.0},
                  {1.0, -1.0, -1.0}
                  // clang-format on
              },
              exec))
    {
        auto values_00 = mtx_00->get_values();
        values_00[0] = value_type{2.0};
        values_00[1] = value_type{-1.0};
        values_00[2] = value_type{3.0};
        auto values_01 = mtx_01->get_values();
        values_01[0] = value_type{4.0};
        values_01[1] = value_type{-2.0};
        values_01[2] = value_type{0.5};
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<BMtx> mtx_0;
    std::unique_ptr<DiagonalMtx> mtx_00;
    std::unique_ptr<DiagonalMtx> mtx_01;
    std::unique_ptr<BMVec> b_0;
    std::unique_ptr<DenseMtx> b_00;
    std::unique_ptr<DenseMtx> b_01;
    std::unique_ptr<BMVec> x_0;
    std::unique_ptr<DenseMtx> x_00;
    std::unique_ptr<DenseMtx> x_01;

    std::default_random_engine rand_engine;
};


TYPED_TEST_SUITE(Diagonal, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Diagonal, AppliesToBatchMultiVector)
{
    using T = typename TestFixture::value_type;

    this->mtx_0->apply(this->b_0.get(), this->x_0.get());

    this->mtx_00->apply(this->b_00.get(), this->x_00.get());
    this->mtx_01->apply(this->b_01.get(), this->x_01.get());
    auto res = gko::batch::unbatch<gko::batch::MultiVector<T>>(this->x_0.get());
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->x_00.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->x_01.get(), 0.);
}


TYPED_TEST(Diagonal, AppliesLinearCombinationToBatchMultiVector)
{
    using BMtx = typename TestFixture::BMtx;
    using BMVec = typename TestFixture::BMVec;
    using DenseMtx = typename TestFixture::DenseMtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch::initialize<BMVec>({{1.5}, {-1.0}}, this->exec);
    auto beta = gko::batch::initialize<BMVec>({{2.5}, {-4.0}}, this->exec);
    auto alpha0 = gko::initialize<DenseMtx>({1.5}, this->exec);
    auto alpha1 = gko::initialize<DenseMtx>({-1.0}, this->exec);
    auto beta0 = gko::initialize<DenseMtx>({2.5}, this->exec);
    auto beta1 = gko::initialize<DenseMtx>({-4.0}, this->exec);

    this->mtx_0->apply(alpha.get(), this->b_0.get(), beta.get(),
                       this->x_0.get());

    this->mtx_00->apply(alpha0.get(), this->b_00.get(), beta0.get(),
                        this->x_00.get());
    this->mtx_01->apply(alpha1.get(), this->b_01.get(), beta1.get(),
                        this->x_01.get());
    auto res = gko::batch::unbatch<gko::batch::MultiVector<T>>(this->x_0.get());
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->x_00.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->x_01.get(), 0.);
}


TYPED_TEST(Diagonal, ApplyFailsOnWrongNumberOfResultCols)
{
    using BMVec = typename TestFixture::BMVec;

    auto res = BMVec::create(this->exec, gko::batch_dim<2>{2, gko::dim<2>{2}});

    ASSERT_THROW(this->mtx_0->apply(this->b_0.get(), res.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Diagonal, ApplyFailsOnWrongNumberOfResultRows)
{
    using BMVec = typename TestFixture::BMVec;

    auto res = BMVec::create(this->exec, gko::batch_dim<2>{2, gko::dim<2>{4}});

    ASSERT_THROW(this->mtx_0->apply(this->b_0.get(), res.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Diagonal, ApplyFailsOnWrongInnerDimension)
{
    using BMVec = typename TestFixture::BMVec;

    auto res =
        BMVec::create(this->exec, gko::batch_dim<2>{2, gko::dim<2>{2, 3}});

    ASSERT_THROW(this->mtx_0->apply(res.get(), this->x_0.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Diagonal, AdvancedApplyFailsOnWrongInnerDimension)
{
    using BMVec = typename TestFixture::BMVec;
    auto res =
        BMVec::create(this->exec, gko::batch_dim<2>{2, gko::dim<2>{2, 3}});
    auto alpha =
        BMVec::create(this->exec, gko::batch_dim<2>{2, gko::dim<2>{1, 1}});
    auto beta =
        BMVec::create(this->exec, gko::batch_dim<2>{2, gko::dim<2>{1, 1}});

    ASSERT_THROW(
        this->mtx_0->apply(alpha.get(), res.get(), beta.get(), this->x_0.get()),
        gko::DimensionMismatch);
}


TYPED_TEST(Diagonal, AdvancedApplyFailsOnWrongAlphaDimension)
{
    using BMVec = typename TestFixture::BMVec;
    auto res =
        BMVec::create(this->exec, gko::batch_dim<2>{2, gko::dim<2>{3, 3}});
    auto alpha =
        BMVec::create(this->exec, gko::batch_dim<2>{2, gko::dim<2>{2, 1}});
    auto beta =
        BMVec::create(this->exec, gko::batch_dim<2>{2, gko::dim<2>{1, 1}});

    ASSERT_THROW(
        this->mtx_0->apply(alpha.get(), res.get(), beta.get(), this->x_0.get()),
        gko::DimensionMismatch);
}
