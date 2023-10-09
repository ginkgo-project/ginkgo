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


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/batch_dense_kernels.hpp"
#include "core/test/utils.hpp"


template <typename T>
class Dense : public ::testing::Test {
protected:
    using value_type = T;
    using size_type = gko::size_type;
    using BMtx = gko::batch::matrix::Dense<value_type>;
    using BMVec = gko::batch::MultiVector<value_type>;
    using DenseMtx = gko::matrix::Dense<value_type>;
    Dense()
        : exec(gko::ReferenceExecutor::create()),
          mtx_0(gko::batch::initialize<BMtx>(
              {{I<T>({1.0, -1.0, 1.5}), I<T>({-2.0, 2.0, 3.0})},
               {{1.0, -2.0, -0.5}, {1.0, -2.5, 4.0}}},
              exec)),
          mtx_00(gko::initialize<DenseMtx>(
              {I<T>({1.0, -1.0, 1.5}), I<T>({-2.0, 2.0, 3.0})}, exec)),
          mtx_01(gko::initialize<DenseMtx>(
              {I<T>({1.0, -2.0, -0.5}), I<T>({1.0, -2.5, 4.0})}, exec)),
          b_0(gko::batch::initialize<BMVec>(
              {{I<T>({1.0, 0.0, 1.0}), I<T>({2.0, 0.0, 1.0}),
                I<T>({1.0, 0.0, 2.0})},
               {I<T>({-1.0, 1.0, 1.0}), I<T>({1.0, -1.0, 1.0}),
                I<T>({1.0, 0.0, 2.0})}},
              exec)),
          b_00(gko::initialize<DenseMtx>(
              {I<T>({1.0, 0.0, 1.0}), I<T>({2.0, 0.0, 1.0}),
               I<T>({1.0, 0.0, 2.0})},
              exec)),
          b_01(gko::initialize<DenseMtx>(
              {I<T>({-1.0, 1.0, 1.0}), I<T>({1.0, -1.0, 1.0}),
               I<T>({1.0, 0.0, 2.0})},
              exec)),
          x_0(gko::batch::initialize<BMVec>(
              {{I<T>({2.0, 0.0, 1.0}), I<T>({2.0, 0.0, 2.0})},
               {I<T>({-2.0, 1.0, 1.0}), I<T>({1.0, -1.0, -1.0})}},
              exec)),
          x_00(gko::initialize<DenseMtx>(
              {I<T>({2.0, 0.0, 1.0}), I<T>({2.0, 0.0, 2.0})}, exec)),
          x_01(gko::initialize<DenseMtx>(
              {I<T>({-2.0, 1.0, 1.0}), I<T>({1.0, -1.0, -1.0})}, exec))
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<BMtx> mtx_0;
    std::unique_ptr<DenseMtx> mtx_00;
    std::unique_ptr<DenseMtx> mtx_01;
    std::unique_ptr<BMVec> b_0;
    std::unique_ptr<DenseMtx> b_00;
    std::unique_ptr<DenseMtx> b_01;
    std::unique_ptr<BMVec> x_0;
    std::unique_ptr<DenseMtx> x_00;
    std::unique_ptr<DenseMtx> x_01;

    std::default_random_engine rand_engine;
};


TYPED_TEST_SUITE(Dense, gko::test::ValueTypes);


TYPED_TEST(Dense, AppliesToBatchMultiVector)
{
    using T = typename TestFixture::value_type;

    this->mtx_0->apply(this->b_0.get(), this->x_0.get());

    this->mtx_00->apply(this->b_00.get(), this->x_00.get());
    this->mtx_01->apply(this->b_01.get(), this->x_01.get());
    auto res = gko::batch::unbatch<gko::batch::MultiVector<T>>(this->x_0.get());
    GKO_ASSERT_MTX_NEAR(res[0].get(), this->x_00.get(), 0.);
    GKO_ASSERT_MTX_NEAR(res[1].get(), this->x_01.get(), 0.);
}


TYPED_TEST(Dense, AppliesLinearCombinationWithSameAlphaToBatchMultiVector)
{
    using BMtx = typename TestFixture::BMtx;
    using BMVec = typename TestFixture::BMVec;
    using DenseMtx = typename TestFixture::DenseMtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch::initialize<BMVec>(2, {1.5}, this->exec);
    auto beta = gko::batch::initialize<BMVec>(2, {-4.0}, this->exec);
    auto alpha0 = gko::initialize<DenseMtx>({1.5}, this->exec);
    auto alpha1 = gko::initialize<DenseMtx>({1.5}, this->exec);
    auto beta0 = gko::initialize<DenseMtx>({-4.0}, this->exec);
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


TYPED_TEST(Dense, AppliesLinearCombinationToBatchMultiVector)
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


TYPED_TEST(Dense, ApplyFailsOnWrongNumberOfResultCols)
{
    using BMVec = typename TestFixture::BMVec;

    auto res = BMVec::create(this->exec, gko::batch_dim<2>{2, gko::dim<2>{2}});

    ASSERT_THROW(this->mtx_0->apply(this->b_0.get(), res.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ApplyFailsOnWrongNumberOfResultRows)
{
    using BMVec = typename TestFixture::BMVec;

    auto res = BMVec::create(this->exec, gko::batch_dim<2>{2, gko::dim<2>{3}});

    ASSERT_THROW(this->mtx_0->apply(this->b_0.get(), res.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ApplyFailsOnWrongInnerDimension)
{
    using BMVec = typename TestFixture::BMVec;

    auto res =
        BMVec::create(this->exec, gko::batch_dim<2>{2, gko::dim<2>{2, 3}});

    ASSERT_THROW(this->mtx_0->apply(res.get(), this->x_0.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, AdvancedApplyFailsOnWrongInnerDimension)
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


TYPED_TEST(Dense, AdvancedApplyFailsOnWrongAlphaDimension)
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
