/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/matrix/batch_dense.hpp>


#include <complex>
#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/batch_dense_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename T>
class BatchDense : public ::testing::Test {
protected:
    using value_type = T;
    using size_type = gko::size_type;
    using Mtx = gko::matrix::BatchDense<value_type>;
    using DenseMtx = gko::matrix::Dense<value_type>;
    using ComplexMtx = gko::to_complex<Mtx>;
    using RealMtx = gko::remove_complex<Mtx>;
    BatchDense()
        : exec(gko::ReferenceExecutor::create()),
          mtx_0(gko::batch_initialize<Mtx>(
              {{I<T>({1.0, -1.0}), I<T>({-2.0, 2.0})},
               {{1.0, -2.0, 3.0}, {1.0, 2.5, 3.0}}},
              exec)),
          mtx_00(gko::initialize<DenseMtx>(
              {I<T>({1.0, -1.0}), I<T>({-2.0, 2.0})}, exec)),
          mtx_01(gko::initialize<DenseMtx>({{1.0, -2.0, 3.0}, {1.0, 2.5, 3.0}},
                                           exec)),
          mtx_1(gko::batch_initialize<Mtx>(
              std::vector<size_type>{2, 4},
              {{I<T>({1.0, -1.0}), I<T>({-2.0, 2.0})},
               {{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}}},
              exec)),
          mtx_10(gko::initialize<DenseMtx>(
              {I<T>({1.0, -1.0}), I<T>({-2.0, 2.0})}, exec)),
          mtx_11(gko::initialize<DenseMtx>(
              4, {{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}}, exec)),
          mtx_2(gko::batch_initialize<Mtx>(
              std::vector<size_type>{4, 2},
              {{{1.0, 1.5, 3.0}, {6.0, 1.0, 5.0}},
               {I<T>({2.0, -2.0}), I<T>({1.0, 3.0}), I<T>({4.0, 3.0})}},
              exec)),
          mtx_20(gko::initialize<DenseMtx>(
              4, {{1.0, 1.5, 3.0}, {6.0, 1.0, 5.0}}, exec)),
          mtx_21(gko::initialize<DenseMtx>(
              {I<T>({2.0, -2.0}), I<T>({1.0, 3.0}), I<T>({4.0, 3.0})}, exec)),
          mtx_3(gko::batch_initialize<Mtx>(
              std::vector<size_type>{4, 2},
              {{{1.0, 1.5, 3.0}, {6.0, 1.0, 5.0}},
               {I<T>({2.0, -2.0}), I<T>({1.0, 3.0})}},
              exec)),
          mtx_30(gko::initialize<DenseMtx>(
              4, {{1.0, 1.5, 3.0}, {6.0, 1.0, 5.0}}, exec)),
          mtx_31(gko::initialize<DenseMtx>(
              {I<T>({2.0, -2.0}), I<T>({1.0, 3.0})}, exec)),
          mtx_4(gko::batch_initialize<Mtx>(
              {{{1.0, 1.5, 3.0}, {6.0, 1.0, 5.0}, {6.0, 1.0, 5.0}},
               {I<T>({2.0, -2.0}), I<T>({4.0, 3.0})}},
              exec)),
          mtx_5(gko::batch_initialize<Mtx>(
              {{{1.0, 1.5, 3.0}, {6.0, 1.0, 5.0}},
               {I<T>({2.0, -2.0}), I<T>({1.0, 3.0}), I<T>({4.0, 3.0})}},
              exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
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

    std::ranlux48 rand_engine;

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


TYPED_TEST_SUITE(BatchDense, gko::test::ValueTypes);


TYPED_TEST(BatchDense, AppliesToBatchDense)
{
    using T = typename TestFixture::value_type;
    this->mtx_1->apply(this->mtx_2.get(), this->mtx_3.get());
    this->mtx_10->apply(this->mtx_20.get(), this->mtx_30.get());
    this->mtx_11->apply(this->mtx_21.get(), this->mtx_31.get());


    auto res = this->mtx_3->unbatch();
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->mtx_30.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->mtx_31.get(), 0.);
}


TYPED_TEST(BatchDense, AppliesLinearCombinationToBatchDense)
{
    using Mtx = typename TestFixture::Mtx;
    using DenseMtx = typename TestFixture::DenseMtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch_initialize<Mtx>({{1.5}, {-1.0}}, this->exec);
    auto beta = gko::batch_initialize<Mtx>({{2.5}, {-4.0}}, this->exec);
    auto alpha0 = gko::initialize<DenseMtx>({1.5}, this->exec);
    auto alpha1 = gko::initialize<DenseMtx>({-1.0}, this->exec);
    auto beta0 = gko::initialize<DenseMtx>({2.5}, this->exec);
    auto beta1 = gko::initialize<DenseMtx>({-4.0}, this->exec);

    this->mtx_1->apply(alpha.get(), this->mtx_2.get(), beta.get(),
                       this->mtx_3.get());
    this->mtx_10->apply(alpha0.get(), this->mtx_20.get(), beta0.get(),
                        this->mtx_30.get());
    this->mtx_11->apply(alpha1.get(), this->mtx_21.get(), beta1.get(),
                        this->mtx_31.get());

    auto res = this->mtx_3->unbatch();
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->mtx_30.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->mtx_31.get(), 0.);
}


TYPED_TEST(BatchDense, ApplyFailsOnWrongInnerDimension)
{
    using Mtx = typename TestFixture::Mtx;
    auto res = Mtx::create(
        this->exec, std::vector<gko::dim<2>>{gko::dim<2>{2}, gko::dim<2>{2}});

    ASSERT_THROW(this->mtx_2->apply(this->mtx_1.get(), res.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(BatchDense, ApplyFailsOnWrongNumberOfRows)
{
    using Mtx = typename TestFixture::Mtx;
    auto res = Mtx::create(
        this->exec, std::vector<gko::dim<2>>{gko::dim<2>{3}, gko::dim<2>{3}});

    ASSERT_THROW(this->mtx_1->apply(this->mtx_2.get(), res.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(BatchDense, ApplyFailsOnWrongNumberOfCols)
{
    using Mtx = typename TestFixture::Mtx;
    auto res = Mtx::create(
        this->exec, std::vector<gko::dim<2>>{gko::dim<2>{2}, gko::dim<2>{2}},
        std::vector<gko::size_type>{3, 3});


    ASSERT_THROW(this->mtx_1->apply(this->mtx_2.get(), res.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(BatchDense, ScalesData)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch_initialize<Mtx>({{{2.0, -2.0}}, {{2.0, -2.0, 3.0}}},
                                            this->exec);

    auto ualpha = alpha->unbatch();

    this->mtx_0->scale(alpha.get());
    this->mtx_00->scale(ualpha[0].get());
    this->mtx_01->scale(ualpha[1].get());

    auto res = this->mtx_0->unbatch();
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->mtx_00.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->mtx_01.get(), 0.);
}


TYPED_TEST(BatchDense, ScalesDataWithScalar)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch_initialize<Mtx>({{2.0}, {-2.0}}, this->exec);

    auto ualpha = alpha->unbatch();

    this->mtx_1->scale(alpha.get());
    this->mtx_10->scale(ualpha[0].get());
    this->mtx_11->scale(ualpha[1].get());

    auto res = this->mtx_1->unbatch();
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->mtx_10.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->mtx_11.get(), 0.);
}


TYPED_TEST(BatchDense, ScalesDataWithStride)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch_initialize<Mtx>({{{2.0, -2.0}}, {{2.0, -2.0, 3.0}}},
                                            this->exec);

    auto ualpha = alpha->unbatch();

    this->mtx_1->scale(alpha.get());
    this->mtx_10->scale(ualpha[0].get());
    this->mtx_11->scale(ualpha[1].get());

    auto res = this->mtx_1->unbatch();
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->mtx_10.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->mtx_11.get(), 0.);
}


TYPED_TEST(BatchDense, AddsScaled)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch_initialize<Mtx>({{{2.0, -2.0}}, {{2.0, -2.0, 3.0}}},
                                            this->exec);

    auto ualpha = alpha->unbatch();

    this->mtx_1->add_scaled(alpha.get(), this->mtx_0.get());
    this->mtx_10->add_scaled(ualpha[0].get(), this->mtx_00.get());
    this->mtx_11->add_scaled(ualpha[1].get(), this->mtx_01.get());

    auto res = this->mtx_0->unbatch();
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->mtx_00.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->mtx_01.get(), 0.);
}


TYPED_TEST(BatchDense, AddsScaledWithScalar)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch_initialize<Mtx>({{2.0}, {-2.0}}, this->exec);

    auto ualpha = alpha->unbatch();

    this->mtx_1->add_scaled(alpha.get(), this->mtx_0.get());
    this->mtx_10->add_scaled(ualpha[0].get(), this->mtx_00.get());
    this->mtx_11->add_scaled(ualpha[1].get(), this->mtx_01.get());

    auto res = this->mtx_0->unbatch();
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->mtx_00.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->mtx_01.get(), 0.);
}


TYPED_TEST(BatchDense, AddScaledFailsOnWrongSizes)
{
    using Mtx = typename TestFixture::Mtx;
    auto alpha =
        gko::batch_initialize<Mtx>({{2.0, 3.0, 4.0, 5.0}, {-2.0}}, this->exec);

    ASSERT_THROW(this->mtx_1->add_scaled(alpha.get(), this->mtx_2.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(BatchDense, ComputesDot)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto result = Mtx::create(
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{1, 2}, gko::dim<2>{1, 3}});

    auto ures = result->unbatch();

    this->mtx_0->compute_dot(this->mtx_1.get(), result.get());
    this->mtx_00->compute_dot(this->mtx_10.get(), ures[0].get());
    this->mtx_01->compute_dot(this->mtx_11.get(), ures[1].get());

    auto res = result->unbatch();
    GKO_ASSERT_MTX_NEAR(res[0].get(), ures[0].get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), ures[1].get(), 0.);
}


TYPED_TEST(BatchDense, ComputesNorm2)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using T_nc = gko::remove_complex<T>;
    using NormVector = gko::matrix::BatchDense<T_nc>;
    auto mtx(gko::batch_initialize<Mtx>(
        {{I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}},
         {I<T>{-4.0, 2.0}, I<T>{-3.0, -2.0}, I<T>{0.0, 1.0}}},
        this->exec));
    auto result = NormVector::create(
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{1, 2}, gko::dim<2>{1, 2}});

    mtx->compute_norm2(result.get());

    EXPECT_EQ(result->at(0, 0, 0), T_nc{3.0});
    EXPECT_EQ(result->at(0, 0, 1), T_nc{5.0});
    EXPECT_EQ(result->at(1, 0, 0), T_nc{5.0});
    EXPECT_EQ(result->at(1, 0, 1), T_nc{3.0});
}


TYPED_TEST(BatchDense, ComputDotFailsOnWrongInputSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result = Mtx::create(
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{1, 2}, gko::dim<2>{1, 3}});

    ASSERT_THROW(this->mtx_1->compute_dot(this->mtx_2.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(BatchDense, ComputDotFailsOnWrongResultSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result = Mtx::create(
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{1, 3}, gko::dim<2>{1, 3}});

    ASSERT_THROW(this->mtx_0->compute_dot(this->mtx_1.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(BatchDense, ConvertsToPrecision)
{
    using BatchDense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherBatchDense = typename gko::matrix::BatchDense<OtherT>;
    auto tmp = OtherBatchDense::create(this->exec);
    auto res = BatchDense::create(this->exec);
    // If OtherT is more precise: 0, otherwise r
    auto residual = r<OtherT>::value < r<T>::value
                        ? gko::remove_complex<T>{0}
                        : gko::remove_complex<T>{r<OtherT>::value};

    this->mtx_1->convert_to(tmp.get());
    tmp->convert_to(res.get());

    auto ures = res->unbatch();
    auto umtx = this->mtx_1->unbatch();
    GKO_ASSERT_MTX_NEAR(umtx[0].get(), ures[0].get(), residual);
    GKO_ASSERT_MTX_NEAR(umtx[1].get(), ures[1].get(), residual);
}


TYPED_TEST(BatchDense, MovesToPrecision)
{
    using BatchDense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherBatchDense = typename gko::matrix::BatchDense<OtherT>;
    auto tmp = OtherBatchDense::create(this->exec);
    auto res = BatchDense::create(this->exec);
    // If OtherT is more precise: 0, otherwise r
    auto residual = r<OtherT>::value < r<T>::value
                        ? gko::remove_complex<T>{0}
                        : gko::remove_complex<T>{r<OtherT>::value};

    this->mtx_1->move_to(tmp.get());
    tmp->move_to(res.get());

    auto ures = res->unbatch();
    auto umtx = this->mtx_1->unbatch();
    GKO_ASSERT_MTX_NEAR(umtx[0].get(), ures[0].get(), residual);
    GKO_ASSERT_MTX_NEAR(umtx[1].get(), ures[1].get(), residual);
}


TYPED_TEST(BatchDense, ConvertsEmptyToPrecision)
{
    using BatchDense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherBatchDense = typename gko::matrix::BatchDense<OtherT>;
    auto empty = OtherBatchDense::create(this->exec);
    auto res = BatchDense::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_FALSE(res->get_batch_sizes().size());
}


TYPED_TEST(BatchDense, MovesEmptyToPrecision)
{
    using BatchDense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherBatchDense = typename gko::matrix::BatchDense<OtherT>;
    auto empty = OtherBatchDense::create(this->exec);
    auto res = BatchDense::create(this->exec);

    empty->move_to(res.get());

    ASSERT_FALSE(res->get_batch_sizes().size());
}


TYPED_TEST(BatchDense, SquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    auto trans = this->mtx_4->transpose();
    auto trans_as_batch_dense = static_cast<Mtx *>(trans.get());

    auto utb = trans_as_batch_dense->unbatch();
    GKO_ASSERT_MTX_NEAR(utb[0].get(),
                        l({{1.0, 6.0, 6.0}, {1.5, 1.0, 1.0}, {3.0, 5.0, 5.0}}),
                        r<TypeParam>::value);
    GKO_ASSERT_MTX_NEAR(utb[1].get(), l({{2.0, 4.0}, {-2.0, 3.0}}),
                        r<TypeParam>::value);
}


TYPED_TEST(BatchDense, NonSquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    auto trans = this->mtx_5->transpose();
    auto trans_as_batch_dense = static_cast<Mtx *>(trans.get());

    auto utb = trans_as_batch_dense->unbatch();
    GKO_ASSERT_MTX_NEAR(utb[0].get(), l({{1.0, 6.0}, {1.5, 1.0}, {3.0, 5.0}}),
                        r<TypeParam>::value);
    GKO_ASSERT_MTX_NEAR(utb[1].get(), l({{2.0, 1.0, 4.0}, {-2.0, 3.0, 3.0}}),
                        r<TypeParam>::value);
}


}  // namespace
