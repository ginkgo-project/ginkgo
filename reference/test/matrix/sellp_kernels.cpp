/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/matrix/sellp.hpp>


#include <gtest/gtest.h>


#include <core/test/utils/assertions.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace {


class Sellp : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Sellp<>;
    using Vec = gko::matrix::Dense<>;

    Sellp()
        : exec(gko::ReferenceExecutor::create()),
          mtx1(Mtx::create(exec)),
          mtx2(Mtx::create(exec))
    {
        // clang-format off
        mtx1 = gko::initialize<Mtx>({{1.0, 3.0, 2.0},
                                     {0.0, 5.0, 0.0}}, exec);
        mtx2 = gko::initialize<Mtx>({{1.0, 3.0, 2.0},
                                     {0.0, 5.0, 0.0}}, exec,
                                     gko::dim<2>{}, 2, 2, 0);
        // clang-format on
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx1;
    std::unique_ptr<Mtx> mtx2;
};


TEST_F(Sellp, AppliesToDenseVector)
{
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = Vec::create(exec, gko::dim<2>{2, 1});

    mtx1->apply(x.get(), y.get());

    ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TEST_F(Sellp, AppliesToDenseMatrix)
{
    // clang-format off
    auto x = gko::initialize<Vec>(
        {{2.0, 3.0},
         {1.0, -1.5},
         {4.0, 2.5}}, exec);
    // clang-format on
    auto y = Vec::create(exec, gko::dim<2>{2});

    mtx1->apply(x.get(), y.get());

    // clang-format off
    ASSERT_MTX_NEAR(y,
                    l({{13.0,  3.5},
                       { 5.0, -7.5}}), 0.0);
    // clang-format on
}


TEST_F(Sellp, AppliesLinearCombinationToDenseVector)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, exec);

    mtx1->apply(alpha.get(), x.get(), beta.get(), y.get());

    ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TEST_F(Sellp, AppliesLinearCombinationToDenseMatrix)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    // clang-format off
    auto x = gko::initialize<Vec>(
        {{2.0, 3.0},
         {1.0, -1.5},
         {4.0, 2.5}}, exec);
    auto y = gko::initialize<Vec>(
        {{1.0, 0.5},
         {2.0, -1.5}}, exec);
    // clang-format on

    mtx1->apply(alpha.get(), x.get(), beta.get(), y.get());

    // clang-format off
    ASSERT_MTX_NEAR(y,
                    l({{-11.0, -2.5},
                       { -1.0,  4.5}}), 0.0);
    // clang-format on
}


TEST_F(Sellp, ApplyFailsOnWrongInnerDimension)
{
    auto x = Vec::create(exec, gko::dim<2>{2});
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx1->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Sellp, ApplyFailsOnWrongNumberOfRows)
{
    auto x = Vec::create(exec, gko::dim<2>{3, 2});
    auto y = Vec::create(exec, gko::dim<2>{3, 2});

    ASSERT_THROW(mtx1->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Sellp, ApplyFailsOnWrongNumberOfCols)
{
    auto x = Vec::create(exec, gko::dim<2>{3}, 2);
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx1->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Sellp, ConvertsToDense)
{
    auto dense_mtx = gko::matrix::Dense<>::create(mtx1->get_executor());

    mtx1->convert_to(dense_mtx.get());

    // clang-format off
    ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TEST_F(Sellp, MovesToDense)
{
    auto dense_mtx = gko::matrix::Dense<>::create(mtx1->get_executor());

    mtx1->move_to(dense_mtx.get());

    // clang-format off
    ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TEST_F(Sellp, AppliesWithSliceSizeAndStrideFactorToDenseVector)
{
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = Vec::create(exec, gko::dim<2>{2, 1});

    mtx2->apply(x.get(), y.get());

    ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TEST_F(Sellp, AppliesWithSliceSizeAndStrideFactorToDenseMatrix)
{
    // clang-format off
    auto x = gko::initialize<Vec>(
        {{2.0, 3.0},
         {1.0, -1.5},
         {4.0, 2.5}}, exec);
    // clang-format on
    auto y = Vec::create(exec, gko::dim<2>{2});

    mtx2->apply(x.get(), y.get());

    // clang-format off
    ASSERT_MTX_NEAR(y,
                    l({{13.0, 3.5},
                       {5.0, -7.5}}), 0.0);
    // clang-format on
}


TEST_F(Sellp, AppliesWithSliceSizeAndStrideFactorLinearCombinationToDenseVector)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, exec);

    mtx2->apply(alpha.get(), x.get(), beta.get(), y.get());

    ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TEST_F(Sellp, AppliesWithSliceSizeAndStrideFactorLinearCombinationToDenseMatrix)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    // clang-format off
    auto x = gko::initialize<Vec>(
        {{2.0, 3.0},
         {1.0, -1.5},
         {4.0, 2.5}}, exec);
    auto y = gko::initialize<Vec>(
        {{1.0, 0.5},
         {2.0, -1.5}}, exec);
    // clang-format on

    mtx2->apply(alpha.get(), x.get(), beta.get(), y.get());

    // clang-format off
    ASSERT_MTX_NEAR(y,
                    l({{-11.0, -2.5},
                       {-1.0, 4.5}}), 0.0);
    // clang-format on
}


TEST_F(Sellp, ApplyWithSliceSizeAndStrideFactorFailsOnWrongInnerDimension)
{
    auto x = Vec::create(exec, gko::dim<2>{2});
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx2->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Sellp, ApplyWithSliceSizeAndStrideFactorFailsOnWrongNumberOfRows)
{
    auto x = Vec::create(exec, gko::dim<2>{3, 2});
    auto y = Vec::create(exec, gko::dim<2>{3, 2});

    ASSERT_THROW(mtx2->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Sellp, ApplyWithSliceSizeAndStrideFactorFailsOnWrongNumberOfCols)
{
    auto x = Vec::create(exec, gko::dim<2>{3}, 2);
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx2->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Sellp, ConvertsWithSliceSizeAndStrideFactorToDense)
{
    auto dense_mtx = gko::matrix::Dense<>::create(mtx2->get_executor());
    // clang-format off
    auto dense_other = gko::initialize<gko::matrix::Dense<>>(
        4, {{1.0, 3.0, 2.0},
            {0.0, 5.0, 0.0}}, exec);
    // clang-format on

    mtx2->convert_to(dense_mtx.get());

    // clang-format off
    ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TEST_F(Sellp, MovesWithSliceSizeAndStrideFactorToDense)
{
    auto dense_mtx = gko::matrix::Dense<>::create(mtx2->get_executor());

    mtx2->move_to(dense_mtx.get());

    // clang-format off
    ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


}  // namespace
