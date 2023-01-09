/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>
#include <ginkgo/core/matrix/batch_identity.hpp>
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
              {{I<T>({1.0, -1.0, 1.5}), I<T>({-2.0, 2.0, 3.0})},
               {{1.0, -2.0, -0.5}, {1.0, -2.5, 4.0}}},
              exec)),
          mtx_00(gko::initialize<DenseMtx>(
              {I<T>({1.0, -1.0, 1.5}), I<T>({-2.0, 2.0, 3.0})}, exec)),
          mtx_01(gko::initialize<DenseMtx>(
              {I<T>({1.0, -2.0, -0.5}), I<T>({1.0, -2.5, 4.0})}, exec)),
          mtx_1(
              gko::batch_initialize<Mtx>(std::vector<size_type>{4, 4},
                                         {{{1.0, -1.0, 2.2}, {-2.0, 2.0, -0.5}},
                                          {{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}}},
                                         exec)),
          mtx_10(gko::initialize<DenseMtx>(
              {I<T>({1.0, -1.0, 2.2}), I<T>({-2.0, 2.0, -0.5})}, exec)),
          mtx_11(gko::initialize<DenseMtx>(
              4, {{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}}, exec)),
          mtx_2(gko::batch_initialize<Mtx>(
              std::vector<size_type>{2, 2},
              {{{1.0, 1.5}, {6.0, 1.0}, {-0.25, 1.0}},
               {I<T>({2.0, -2.0}), I<T>({1.0, 3.0}), I<T>({4.0, 3.0})}},
              exec)),
          mtx_20(gko::initialize<DenseMtx>(
              4, {I<T>({1.0, 1.5}), I<T>({6.0, 1.0}), I<T>({-0.25, 1.0})},
              exec)),
          mtx_21(gko::initialize<DenseMtx>(
              {I<T>({2.0, -2.0}), I<T>({1.0, 3.0}), I<T>({4.0, 3.0})}, exec)),
          mtx_3(gko::batch_initialize<Mtx>(
              std::vector<size_type>{4, 4},
              {{I<T>({1.0, 1.5}), I<T>({6.0, 1.0})}, {{2.0, -2.0}, {1.0, 3.0}}},
              exec)),
          mtx_30(gko::initialize<DenseMtx>({I<T>({1.0, 1.5}), I<T>({6.0, 1.0})},
                                           exec)),
          mtx_31(gko::initialize<DenseMtx>(
              {I<T>({2.0, -2.0}), I<T>({1.0, 3.0})}, exec)),
          mtx_4(gko::batch_initialize<Mtx>(
              {{{1.0, 1.5, 3.0}, {6.0, 1.0, 5.0}, {6.0, 1.0, 5.5}},
               {{2.0, -2.0, 1.5}, {4.0, 3.0, 2.2}, {-1.25, 3.0, 0.5}}},
              exec)),
          mtx_5(gko::batch_initialize<Mtx>(
              {{{1.0, 1.5}, {6.0, 1.0}, {7.0, -4.5}},
               {I<T>({2.0, -2.0}), I<T>({1.0, 3.0}), I<T>({4.0, 3.0})}},
              exec)),
          mtx_6(gko::batch_initialize<Mtx>(
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

    std::ranlux48 rand_engine;
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


TYPED_TEST(BatchDense, ApplyFailsForNonUniformBatches)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto mat1 = gko::batch_initialize<Mtx>(
        std::vector<gko::size_type>{4, 4},
        {{I<T>({1.0, -1.0}), I<T>({1.0, -1.0}), I<T>({2.0, -0.5})},
         {{1.0, 2.5, 3.0}, {1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}}},
        this->exec);
    auto mat2 = gko::batch_initialize<Mtx>(
        std::vector<gko::size_type>{4, 4},
        {{{1.0, -1.0, 2.2}, {-2.0, 2.0, -0.5}},
         {{1.0, 2.5, -3.0}, {1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}}},
        this->exec);
    auto res = Mtx::create(
        this->exec, std::vector<gko::dim<2>>{gko::dim<2>{2}, gko::dim<2>{3}});

    ASSERT_THROW(mat2->apply(mat1.get(), res.get()), gko::NotImplemented);
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
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{2, 1}, gko::dim<2>{2, 1}},
        std::vector<gko::size_type>{3, 3});


    ASSERT_THROW(this->mtx_1->apply(this->mtx_2.get(), res.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(BatchDense, ScalesData)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch_initialize<Mtx>(
        std::vector<gko::size_type>{3, 3},
        {{{2.0, -2.0, 1.5}}, {{3.0, -1.0, 0.25}}}, this->exec);

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
    auto alpha = gko::batch_initialize<Mtx>(
        {{{2.0, -2.0, -1.5}}, {{2.0, -2.0, 3.0}}}, this->exec);

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
    auto alpha = gko::batch_initialize<Mtx>(
        {{{2.0, -2.0, 1.5}}, {{2.0, -2.0, 3.0}}}, this->exec);

    auto ualpha = alpha->unbatch();

    this->mtx_1->add_scaled(alpha.get(), this->mtx_0.get());
    this->mtx_10->add_scaled(ualpha[0].get(), this->mtx_00.get());
    this->mtx_11->add_scaled(ualpha[1].get(), this->mtx_01.get());

    auto res = this->mtx_1->unbatch();
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->mtx_10.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->mtx_11.get(), 0.);
}


TYPED_TEST(BatchDense, AddsScale)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch_initialize<Mtx>(
        {{{2.0, -2.0, 1.5}}, {{2.0, -2.0, 3.0}}}, this->exec);
    auto beta = gko::batch_initialize<Mtx>(
        {{{-1.0, 3.0, 0.5}}, {{1.5, 0.5, -4.0}}}, this->exec);

    auto ualpha = alpha->unbatch();
    auto ubeta = beta->unbatch();

    this->mtx_1->add_scale(alpha.get(), this->mtx_0.get(), beta.get());
    this->mtx_10->add_scale(ualpha[0].get(), this->mtx_00.get(),
                            ubeta[0].get());
    this->mtx_11->add_scale(ualpha[1].get(), this->mtx_01.get(),
                            ubeta[1].get());

    auto res = this->mtx_1->unbatch();
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->mtx_10.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->mtx_11.get(), 0.);
}


TYPED_TEST(BatchDense, ConvergenceAddScaled)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch_initialize<Mtx>(
        {{{2.0, -2.0, 1.5}}, {{2.0, -2.0, 3.0}}}, this->exec);

    auto ualpha = alpha->unbatch();


    const int num_rhs = 3;
    const gko::uint32 converged = 0xfffffffd | (0 - (1 << num_rhs));

    gko::kernels::reference::batch_dense::convergence_add_scaled(
        this->exec, alpha.get(), this->mtx_0.get(), this->mtx_1.get(),
        converged);

    auto mtx_10_clone = gko::clone(this->mtx_10);
    auto mtx_11_clone = gko::clone(this->mtx_11);

    this->mtx_10->add_scaled(ualpha[0].get(), this->mtx_00.get());
    this->mtx_11->add_scaled(ualpha[1].get(), this->mtx_01.get());

    auto res = this->mtx_1->unbatch();

    EXPECT_EQ(res[0]->at(0, 0), mtx_10_clone->at(0, 0));
    EXPECT_EQ(res[0]->at(1, 0), mtx_10_clone->at(1, 0));
    EXPECT_EQ(res[0]->at(0, 1), this->mtx_10->at(0, 1));
    EXPECT_EQ(res[0]->at(1, 1), this->mtx_10->at(1, 1));
    EXPECT_EQ(res[0]->at(0, 2), mtx_10_clone->at(0, 2));
    EXPECT_EQ(res[0]->at(1, 2), mtx_10_clone->at(1, 2));

    EXPECT_EQ(res[1]->at(0, 0), mtx_11_clone->at(0, 0));
    EXPECT_EQ(res[1]->at(1, 0), mtx_11_clone->at(1, 0));
    EXPECT_EQ(res[1]->at(0, 1), this->mtx_11->at(0, 1));
    EXPECT_EQ(res[1]->at(1, 1), this->mtx_11->at(1, 1));
    EXPECT_EQ(res[1]->at(0, 2), mtx_11_clone->at(0, 2));
    EXPECT_EQ(res[1]->at(1, 2), mtx_11_clone->at(1, 2));
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

    auto res = this->mtx_1->unbatch();
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->mtx_10.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->mtx_11.get(), 0.);
}


TYPED_TEST(BatchDense, AddsScaleWithScalar)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch_initialize<Mtx>({{2.0}, {-2.0}}, this->exec);
    auto beta = gko::batch_initialize<Mtx>({{-0.5}, {3.0}}, this->exec);

    auto ualpha = alpha->unbatch();
    auto ubeta = beta->unbatch();

    this->mtx_1->add_scale(alpha.get(), this->mtx_0.get(), beta.get());
    this->mtx_10->add_scale(ualpha[0].get(), this->mtx_00.get(),
                            ubeta[0].get());
    this->mtx_11->add_scale(ualpha[1].get(), this->mtx_01.get(),
                            ubeta[1].get());

    auto res = this->mtx_1->unbatch();
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->mtx_10.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->mtx_11.get(), 0.);
}


TYPED_TEST(BatchDense, AddScaleWithScalarViaApply)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch_initialize<Mtx>({{2.0}, {-2.0}}, this->exec);
    auto beta = gko::batch_initialize<Mtx>({{-0.5}, {3.0}}, this->exec);
    auto id = gko::matrix::BatchIdentity<T>::create(
        this->exec, gko::batch_dim<2>(2, gko::dim<2>(3, 3)));
    auto ualpha = alpha->unbatch();
    auto ubeta = beta->unbatch();

    this->mtx_0->apply(alpha.get(), id.get(), beta.get(), this->mtx_1.get());
    this->mtx_10->add_scale(ualpha[0].get(), this->mtx_00.get(),
                            ubeta[0].get());
    this->mtx_11->add_scale(ualpha[1].get(), this->mtx_01.get(),
                            ubeta[1].get());

    auto res = this->mtx_1->unbatch();
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->mtx_10.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->mtx_11.get(), 0.);
}


TYPED_TEST(BatchDense, ConvergenceAddScaledWithScalar)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch_initialize<Mtx>({{2.0}, {-2.0}}, this->exec);

    auto ualpha = alpha->unbatch();


    const int num_rhs = 3;
    const gko::uint32 converged = 0xfffffffd | (0 - (1 << num_rhs));

    gko::kernels::reference::batch_dense::convergence_add_scaled(
        this->exec, alpha.get(), this->mtx_0.get(), this->mtx_1.get(),
        converged);

    auto mtx_10_clone = gko::clone(this->mtx_10);
    auto mtx_11_clone = gko::clone(this->mtx_11);

    this->mtx_10->add_scaled(ualpha[0].get(), this->mtx_00.get());
    this->mtx_11->add_scaled(ualpha[1].get(), this->mtx_01.get());

    auto res = this->mtx_1->unbatch();

    EXPECT_EQ(res[0]->at(0, 0), mtx_10_clone->at(0, 0));
    EXPECT_EQ(res[0]->at(1, 0), mtx_10_clone->at(1, 0));
    EXPECT_EQ(res[0]->at(0, 1), this->mtx_10->at(0, 1));
    EXPECT_EQ(res[0]->at(1, 1), this->mtx_10->at(1, 1));
    EXPECT_EQ(res[0]->at(0, 2), mtx_10_clone->at(0, 2));
    EXPECT_EQ(res[0]->at(1, 2), mtx_10_clone->at(1, 2));

    EXPECT_EQ(res[1]->at(0, 0), mtx_11_clone->at(0, 0));
    EXPECT_EQ(res[1]->at(1, 0), mtx_11_clone->at(1, 0));
    EXPECT_EQ(res[1]->at(0, 1), this->mtx_11->at(0, 1));
    EXPECT_EQ(res[1]->at(1, 1), this->mtx_11->at(1, 1));
    EXPECT_EQ(res[1]->at(0, 2), mtx_11_clone->at(0, 2));
    EXPECT_EQ(res[1]->at(1, 2), mtx_11_clone->at(1, 2));
}


TYPED_TEST(BatchDense, AddScaledFailsOnWrongSizes)
{
    using Mtx = typename TestFixture::Mtx;
    auto alpha =
        gko::batch_initialize<Mtx>({{2.0, 3.0, 4.0, 5.0}, {-2.0}}, this->exec);

    ASSERT_THROW(this->mtx_1->add_scaled(alpha.get(), this->mtx_2.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(BatchDense, AddScaleFailsOnWrongSizes)
{
    using Mtx = typename TestFixture::Mtx;
    auto alpha = gko::batch_initialize<Mtx>({{2.0}, {-2.0}}, this->exec);
    auto beta = gko::batch_initialize<Mtx>({{2.0}, {3.0}}, this->exec);

    ASSERT_THROW(
        this->mtx_1->add_scale(alpha.get(), this->mtx_2.get(), beta.get()),
        gko::DimensionMismatch);
}


TYPED_TEST(BatchDense, AddScaleFailsOnWrongScalarSizes)
{
    using Mtx = typename TestFixture::Mtx;
    auto alpha = gko::batch_initialize<Mtx>(
        {{{2.0, -2.0, 1.5}}, {{2.0, -2.0, 3.0}}}, this->exec);
    auto beta = gko::batch_initialize<Mtx>({{3.0}, {1.5}}, this->exec);

    ASSERT_THROW(
        this->mtx_1->add_scale(alpha.get(), this->mtx_0.get(), beta.get()),
        gko::DimensionMismatch);
}


TYPED_TEST(BatchDense, ComputesDot)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto result =
        Mtx::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>{1, 3}));

    auto ures = result->unbatch();

    this->mtx_0->compute_dot(this->mtx_1.get(), result.get());
    this->mtx_00->compute_dot(this->mtx_10.get(), ures[0].get());
    this->mtx_01->compute_dot(this->mtx_11.get(), ures[1].get());

    auto res = result->unbatch();
    GKO_ASSERT_MTX_NEAR(res[0].get(), ures[0].get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), ures[1].get(), 0.);
}


TYPED_TEST(BatchDense, ConvergenceComputeDot)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto result =
        Mtx::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>{1, 3}));

    for (int ibatch = 0; ibatch < result->get_size().get_batch_sizes().size();
         ibatch++) {
        for (int icol = 0; icol < result->get_size().at()[1]; icol++) {
            result->at(ibatch, 0, icol) = gko::zero<T>();
        }
    }

    auto ures = result->unbatch();

    const int num_rhs = 3;
    const gko::uint32 converged = 0xfffffffd | (0 - (1 << num_rhs));

    gko::kernels::reference::batch_dense::convergence_compute_dot(
        this->exec, this->mtx_0.get(), this->mtx_1.get(), result.get(),
        converged);

    auto ures_00_clone = gko::clone(ures[0]);
    auto ures_01_clone = gko::clone(ures[1]);

    this->mtx_00->compute_dot(this->mtx_10.get(), ures[0].get());
    this->mtx_01->compute_dot(this->mtx_11.get(), ures[1].get());

    auto res = result->unbatch();

    EXPECT_EQ(res[0]->at(0, 0), ures_00_clone->at(0, 0));
    EXPECT_EQ(res[0]->at(0, 1), ures[0]->at(0, 1));
    EXPECT_EQ(res[0]->at(0, 2), ures_00_clone->at(0, 2));

    EXPECT_EQ(res[1]->at(0, 0), ures_01_clone->at(0, 0));
    EXPECT_EQ(res[1]->at(0, 1), ures[1]->at(0, 1));
    EXPECT_EQ(res[1]->at(0, 2), ures_01_clone->at(0, 2));
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
    auto batch_size = gko::batch_dim<2>(
        std::vector<gko::dim<2>>{gko::dim<2>{1, 2}, gko::dim<2>{1, 2}});
    auto result =
        NormVector::create(this->exec, batch_size, gko::batch_stride(2, 2));

    mtx->compute_norm2(result.get());

    EXPECT_EQ(result->at(0, 0, 0), T_nc{3.0});
    EXPECT_EQ(result->at(0, 0, 1), T_nc{5.0});
    EXPECT_EQ(result->at(1, 0, 0), T_nc{5.0});
    EXPECT_EQ(result->at(1, 0, 1), T_nc{3.0});
}


TYPED_TEST(BatchDense, ConvergenceComputeNorm2)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using T_nc = gko::remove_complex<T>;
    using NormVector = gko::matrix::BatchDense<T_nc>;
    auto mtx(gko::batch_initialize<Mtx>(
        {{I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}},
         {I<T>{-4.0, 2.0}, I<T>{-3.0, -2.0}, I<T>{0.0, 1.0}}},
        this->exec));
    auto batch_size = gko::batch_dim<2>(
        std::vector<gko::dim<2>>{gko::dim<2>{1, 2}, gko::dim<2>{1, 2}});
    auto result =
        NormVector::create(this->exec, batch_size, gko::batch_stride(2, 2));

    for (int ibatch = 0; ibatch < result->get_size().get_batch_sizes().size();
         ibatch++) {
        for (int icol = 0; icol < result->get_size().at()[1]; icol++) {
            result->at(ibatch, 0, icol) = gko::zero<T_nc>();
        }
    }

    auto result_clone = gko::clone(result);

    const int num_rhs = 2;
    const gko::uint32 converged = 0xfffffffd | (0 - (1 << num_rhs));

    gko::kernels::reference::batch_dense::convergence_compute_norm2(
        this->exec, mtx.get(), result.get(), converged);

    EXPECT_EQ(result->at(0, 0, 0), result_clone->at(0, 0, 0));
    EXPECT_EQ(result->at(0, 0, 1), T_nc{5.0});

    EXPECT_EQ(result->at(1, 0, 0), result_clone->at(1, 0, 0));
    EXPECT_EQ(result->at(1, 0, 1), T_nc{3.0});
}


TYPED_TEST(BatchDense, ComputDotFailsOnWrongInputSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result =
        Mtx::create(this->exec, gko::batch_dim<2>(std::vector<gko::dim<2>>{
                                    gko::dim<2>{1, 2}, gko::dim<2>{1, 3}}));

    ASSERT_THROW(this->mtx_1->compute_dot(this->mtx_2.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(BatchDense, ComputDotFailsOnWrongResultSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result =
        Mtx::create(this->exec, gko::batch_dim<2>(std::vector<gko::dim<2>>{
                                    gko::dim<2>{1, 2}, gko::dim<2>{1, 2}}));
    auto result2 =
        Mtx::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>{1, 2}));

    ASSERT_THROW(this->mtx_0->compute_dot(this->mtx_1.get(), result.get()),
                 gko::DimensionMismatch);
    ASSERT_THROW(this->mtx_0->compute_dot(this->mtx_1.get(), result2.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(BatchDense, CopiesData)
{
    gko::kernels::reference::batch_dense::copy(this->exec, this->mtx_0.get(),
                                               this->mtx_1.get());

    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx_1.get(), this->mtx_0.get(), 0.);
}


TYPED_TEST(BatchDense, ConvergenceCopyData)
{
    auto umtx_0 = this->mtx_0->unbatch();

    const int num_rhs = 3;
    const gko::uint32 converged = 0xfffffffd | (0 - (1 << num_rhs));
    gko::kernels::reference::batch_dense::convergence_copy(
        this->exec, this->mtx_0.get(), this->mtx_1.get(), converged);

    auto mtx_10_clone = gko::clone(this->mtx_10);
    auto mtx_11_clone = gko::clone(this->mtx_11);

    auto res = this->mtx_1->unbatch();

    EXPECT_EQ(res[0]->at(0, 0), mtx_10_clone->at(0, 0));
    EXPECT_EQ(res[0]->at(1, 0), mtx_10_clone->at(1, 0));
    EXPECT_EQ(res[0]->at(0, 1), this->mtx_0->at(0, 0, 1));
    EXPECT_EQ(res[0]->at(1, 1), this->mtx_0->at(0, 1, 1));
    EXPECT_EQ(res[0]->at(0, 2), mtx_10_clone->at(0, 2));
    EXPECT_EQ(res[0]->at(1, 2), mtx_10_clone->at(1, 2));

    EXPECT_EQ(res[1]->at(0, 0), mtx_11_clone->at(0, 0));
    EXPECT_EQ(res[1]->at(1, 0), mtx_11_clone->at(1, 0));
    EXPECT_EQ(res[1]->at(0, 1), this->mtx_0->at(1, 0, 1));
    EXPECT_EQ(res[1]->at(1, 1), this->mtx_0->at(1, 1, 1));
    EXPECT_EQ(res[1]->at(0, 2), mtx_11_clone->at(0, 2));
    EXPECT_EQ(res[1]->at(1, 2), mtx_11_clone->at(1, 2));
}


TYPED_TEST(BatchDense, BatchScale)
{
    using T = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using BDiag = gko::matrix::BatchDiagonal<T>;

    auto mtx(gko::batch_initialize<Mtx>(
        {{I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}},
         {I<T>{-4.0, 2.0}, I<T>{-3.0, -2.0}, I<T>{0.0, 1.0}}},
        this->exec));

    auto left(gko::batch_diagonal_initialize(
        I<I<T>>{I<T>{1.0, 2.0, 3.0}, I<T>{-1.0, -2.0, -3.0}}, this->exec));
    auto rght(gko::batch_diagonal_initialize(
        I<I<T>>{I<T>{-0.5, -2.0}, I<T>{2.0, 0.25}}, this->exec));

    gko::kernels::reference::batch_dense::batch_scale(this->exec, left.get(),
                                                      rght.get(), mtx.get());

    EXPECT_EQ(mtx->at(0, 0, 0), T{-0.5});
    EXPECT_EQ(mtx->at(0, 1, 0), T{-2.0});
    EXPECT_EQ(mtx->at(0, 2, 0), T{-3.0});
    EXPECT_EQ(mtx->at(0, 0, 1), T{0.0});
    EXPECT_EQ(mtx->at(0, 1, 1), T{-12.0});
    EXPECT_EQ(mtx->at(0, 2, 1), T{-24.0});

    EXPECT_EQ(mtx->at(1, 0, 0), T{8.0});
    EXPECT_EQ(mtx->at(1, 1, 0), T{12.0});
    EXPECT_EQ(mtx->at(1, 2, 0), T{0.0});
    EXPECT_EQ(mtx->at(1, 0, 1), T{-0.5});
    EXPECT_EQ(mtx->at(1, 1, 1), T{1.0});
    EXPECT_EQ(mtx->at(1, 2, 1), T{-0.75});
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


TYPED_TEST(BatchDense, ConvertsToCsr32)
{
    using T = typename TestFixture::value_type;
    using BatchCsr = typename gko::matrix::BatchCsr<T, gko::int32>;
    auto batch_csr_mtx = BatchCsr::create(this->mtx_6->get_executor());

    this->mtx_6->convert_to(batch_csr_mtx.get());

    auto v = batch_csr_mtx->get_const_values();
    auto c = batch_csr_mtx->get_const_col_idxs();
    auto r = batch_csr_mtx->get_const_row_ptrs();
    ASSERT_EQ(batch_csr_mtx->get_num_batch_entries(), 2);
    ASSERT_EQ(batch_csr_mtx->get_size().at(0), gko::dim<2>(3, 3));
    ASSERT_EQ(batch_csr_mtx->get_size().at(1), gko::dim<2>(3, 3));
    ASSERT_EQ(batch_csr_mtx->get_num_stored_elements(), 10);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 2);
    EXPECT_EQ(r[2], 3);
    EXPECT_EQ(r[3], 5);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 2);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], 2);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{3.0});
    EXPECT_EQ(v[3], T{1.0});
    EXPECT_EQ(v[4], T{5.0});
    EXPECT_EQ(v[5], T{2.0});
    EXPECT_EQ(v[6], T{5.0});
    EXPECT_EQ(v[7], T{1.0});
    EXPECT_EQ(v[8], T{-1.0});
    EXPECT_EQ(v[9], T{8.0});
}


TYPED_TEST(BatchDense, MovesToCsr32)
{
    using T = typename TestFixture::value_type;
    using BatchCsr = typename gko::matrix::BatchCsr<T, gko::int32>;
    auto batch_csr_mtx = BatchCsr::create(this->mtx_6->get_executor());

    this->mtx_6->move_to(batch_csr_mtx.get());

    auto v = batch_csr_mtx->get_const_values();
    auto c = batch_csr_mtx->get_const_col_idxs();
    auto r = batch_csr_mtx->get_const_row_ptrs();
    ASSERT_EQ(batch_csr_mtx->get_num_batch_entries(), 2);
    ASSERT_EQ(batch_csr_mtx->get_size().at(0), gko::dim<2>(3, 3));
    ASSERT_EQ(batch_csr_mtx->get_size().at(1), gko::dim<2>(3, 3));
    ASSERT_EQ(batch_csr_mtx->get_num_stored_elements(), 10);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 2);
    EXPECT_EQ(r[2], 3);
    EXPECT_EQ(r[3], 5);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 2);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], 2);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{3.0});
    EXPECT_EQ(v[3], T{1.0});
    EXPECT_EQ(v[4], T{5.0});
    EXPECT_EQ(v[5], T{2.0});
    EXPECT_EQ(v[6], T{5.0});
    EXPECT_EQ(v[7], T{1.0});
    EXPECT_EQ(v[8], T{-1.0});
    EXPECT_EQ(v[9], T{8.0});
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

    ASSERT_FALSE(res->get_num_batch_entries());
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

    ASSERT_FALSE(res->get_num_batch_entries());
}


TYPED_TEST(BatchDense, ConvertsEmptyMatrixToCsr)
{
    using BatchDense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using BatchCsr = typename gko::matrix::BatchCsr<T, gko::int32>;
    auto empty = BatchDense::create(this->exec);
    auto res = BatchCsr::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_num_batch_entries());
}


TYPED_TEST(BatchDense, MovesEmptyMatrixToCsr)
{
    using BatchDense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using BatchCsr = typename gko::matrix::BatchCsr<T, gko::int32>;
    auto empty = BatchDense::create(this->exec);
    auto res = BatchCsr::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_num_batch_entries());
}


TYPED_TEST(BatchDense, ConvertsToBatchDiagonal)
{
    using BDense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using BDiag = gko::matrix::BatchDiagonal<T>;
    auto vec = gko::batch_initialize<BDense>(
        {I<T>({2.0, 3.0, -1.0}), I<T>({1.0, -2.0, 8.0})}, this->exec);
    auto diag = BDiag::create(this->exec);

    vec->convert_to(diag.get());

    auto check_sz = gko::batch_dim<2>{2, gko::dim<2>{3}};
    ASSERT_EQ(diag->get_size(), check_sz);
    auto diag_vals = diag->get_const_values();
    ASSERT_EQ(diag_vals[0], T{2.0});
    ASSERT_EQ(diag_vals[1], T{3.0});
    ASSERT_EQ(diag_vals[2], T{-1.0});
    ASSERT_EQ(diag_vals[3], T{1.0});
    ASSERT_EQ(diag_vals[4], T{-2.0});
    ASSERT_EQ(diag_vals[5], T{8.0});
}


TYPED_TEST(BatchDense, MovesToBatchDiagonal)
{
    using BDense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using BDiag = gko::matrix::BatchDiagonal<T>;
    auto vec = gko::batch_initialize<BDense>(
        {I<T>({2.0, 3.0, -1.0}), I<T>({1.0, -2.0, 8.0})}, this->exec);
    auto vec_ptr = vec->get_const_values();
    auto diag = BDiag::create(this->exec);

    vec->move_to(diag.get());

    auto check_sz = gko::batch_dim<2>{2, gko::dim<2>{3}};
    ASSERT_EQ(diag->get_size(), check_sz);
    auto diag_vals = diag->get_const_values();
    ASSERT_EQ(diag_vals, vec_ptr);
    ASSERT_NE(diag_vals, vec->get_const_values());
    ASSERT_EQ(vec->get_num_batch_entries(), 0);
}


TYPED_TEST(BatchDense, SquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    auto trans = this->mtx_4->transpose();
    auto trans_as_batch_dense = static_cast<Mtx*>(trans.get());

    auto utb = trans_as_batch_dense->unbatch();
    GKO_ASSERT_MTX_NEAR(utb[0].get(),
                        l({{1.0, 6.0, 6.0}, {1.5, 1.0, 1.0}, {3.0, 5.0, 5.5}}),
                        r<TypeParam>::value);
    GKO_ASSERT_MTX_NEAR(
        utb[1].get(), l({{2.0, 4.0, -1.25}, {-2.0, 3.0, 3.0}, {1.5, 2.2, 0.5}}),
        r<TypeParam>::value);
}


TYPED_TEST(BatchDense, NonSquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    auto trans = this->mtx_5->transpose();
    auto trans_as_batch_dense = static_cast<Mtx*>(trans.get());

    auto utb = trans_as_batch_dense->unbatch();
    GKO_ASSERT_MTX_NEAR(utb[0].get(), l({{1.0, 6.0, 7.0}, {1.5, 1.0, -4.5}}),
                        r<TypeParam>::value);
    GKO_ASSERT_MTX_NEAR(utb[1].get(), l({{2.0, 1.0, 4.0}, {-2.0, 3.0, 3.0}}),
                        r<TypeParam>::value);
}


TYPED_TEST(BatchDense, SquareMatrixAddScaledIdentity)
{
    using T = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto mtx = gko::batch_initialize<Mtx>(
        {{I<T>({1.0, -1.0, 1.5}), I<T>({-2.0, 0.0, 3.0}),
          I<T>({1.2, -0.5, 1.0})},
         {{1.0, -2.0, -0.5}, {1.0, -2.5, 4.0}, {3.0, 0.0, -1.5}}},
        this->exec);
    auto alpha = gko::batch_initialize<Mtx>({{2.0}, {-2.0}}, this->exec);
    auto beta = gko::batch_initialize<Mtx>({{3.0}, {-1.0}}, this->exec);
    auto sol_mtx = gko::batch_initialize<Mtx>(
        {{I<T>({5.0, -3.0, 4.5}), I<T>({-6.0, 2.0, 9.0}),
          I<T>({3.6, -1.5, 5.0})},
         {{-3.0, 2.0, 0.5}, {-1.0, 0.5, -4.0}, {-3.0, 0.0, -0.5}}},
        this->exec);

    mtx->add_scaled_identity(alpha.get(), beta.get());

    GKO_ASSERT_BATCH_MTX_NEAR(mtx, sol_mtx, r<T>::value);
}


}  // namespace
