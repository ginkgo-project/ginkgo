// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/conv_kernels.hpp"

#include <algorithm>
#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/conv.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>

#include "core/test/utils.hpp"


template <typename T>
class Conv : public ::testing::Test {
protected:
    using value_type = T;


    using Mtx = gko::matrix::Conv<value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using Array = gko::array<value_type>;

    Conv() : exec{gko::ReferenceExecutor::create()}, kernel_array{exec}
    {
        kernel_vals = {1.0, 2.0, 3.0};
        kernel_array = Array{exec, kernel_vals.begin(), kernel_vals.end()};

        mtx = gko::matrix::Conv<value_type>::create(exec, kernel_array);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;
    Array kernel_array;
    std::vector<value_type> kernel_vals;
};

TYPED_TEST_SUITE(Conv, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Conv, AppliesToDenseVector)
{
    using Vec = typename TestFixture::Vec;

    // generate input and check the result
    auto x = gko::initialize<Vec>({4.0, 5.0, 6.0, 7.0}, this->exec);
    // Allocate output Dense vector: floor((N + 2*padding - K) / stride) + 1
    const gko::size_type output_length = (x->get_size()[0] + 2 * 0 - 3) / 1 + 1;
    auto y = Vec::create(this->exec, gko::dim<2>{output_length, 1});

    this->mtx->apply(x, y);

    GKO_ASSERT_MTX_NEAR(y, l({32.0, 38.0}), 0.0);
}


TYPED_TEST(Conv, ApplyToStridedVectorKeepsPadding)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    // generate input and check the result (similar but vector contains stride)
    auto x = gko::initialize<Vec>({4.0, 5.0, 6.0, 7.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1}, 2);
    y->get_values()[1] = 12345;

    this->mtx->apply(x, y);
    GKO_ASSERT_MTX_NEAR(y, l({32.0, 38.0}), 0.0);
    ASSERT_EQ(y->get_values()[1], T{12345});
}


// TYPED_TEST(Conv, AppliesToDenseMatrix)
//{   // Testing applies to dense matrix
// using Vec = typename TestFixture::Vec;
// using T = typename TestFixture::value_type;
//  // clang-format off

// using Vec = typename TestFixture::Vec;
// using T = typename TestFixture::value_type;
// // clang-format off
// auto x = gko::initialize<Vec>(
//     {I<T>{2.0, 3.0},
//      I<T>{1.0, -1.5},
//      I<T>{4.0, 2.5}}, this->exec);
// // clang-format on
// auto y = Vec::create(this->exec, gko::dim<2>{2, 2});

// this->mtx->apply(x, y);

// // clang-format off
// GKO_ASSERT_MTX_NEAR(y,
//                     l({{13.0,  3.5},
//                        { 5.0, -7.5}}), 0.0);
// // clang-format on
//}


// TYPED_TEST(Conv, AppliesLinearCombinationToDenseVector)
// {
//     using Vec = typename TestFixture::Vec;
//     auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
//     auto beta = gko::initialize<Vec>({2.0}, this->exec);
//     auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
//     auto y = gko::initialize<Vec>({1.0, 2.0}, this->exec);

//     this->mtx->apply(alpha, x, beta, y);

//     GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
// }


// TYPED_TEST(Conv, ApplyLinearCombinationToStridedVectorKeepsPadding)
// {
//     using Vec = typename TestFixture::Vec;
//     using T = typename TestFixture::value_type;
//     auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
//     auto beta = gko::initialize<Vec>({2.0}, this->exec);
//     auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
//     auto y = Vec::create(this->exec, gko::dim<2>{2, 1}, 2);
//     y->get_values()[1] = 1234;
//     y->at(0, 0) = 1.0;
//     y->at(1, 0) = 2.0;

//     this->mtx->apply(alpha, x, beta, y);

//     GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
//     ASSERT_EQ(y->get_values()[1], T{1234});
// }


// TYPED_TEST(Conv, AppliesLinearCombinationToDenseMatrix)
// {
//     using Vec = typename TestFixture::Vec;
//     using T = typename TestFixture::value_type;
//     auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
//     auto beta = gko::initialize<Vec>({2.0}, this->exec);
//     // clang-format off
//     auto x = gko::initialize<Vec>(
//         {I<T>{2.0, 3.0},
//          I<T>{1.0, -1.5},
//          I<T>{4.0, 2.5}}, this->exec);
//     auto y = gko::initialize<Vec>(
//         {I<T>{1.0, 0.5},
//          I<T>{2.0, -1.5}}, this->exec);
//     // clang-format on

//     this->mtx->apply(alpha, x, beta, y);

//     // clang-format off
//     GKO_ASSERT_MTX_NEAR(y,
//                         l({{-11.0, -2.5},
//                            { -1.0,  4.5}}), 0.0);
//     // clang-format on
// }


// TYPED_TEST(Conv, ApplyFailsOnWrongInnerDimension)
// {
//     using Vec = typename TestFixture::Vec;
//     auto x = Vec::create(this->exec, gko::dim<2>{2});
//     auto y = Vec::create(this->exec, gko::dim<2>{2});

//     ASSERT_THROW(this->mtx->apply(x, y), gko::DimensionMismatch);
// }


TYPED_TEST(Conv, ApplyFailsOnWrongNumberOfRows)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{4, 1});
    auto y = Vec::create(this->exec, gko::dim<2>{4, 1});

    ASSERT_THROW(this->mtx->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Conv, ApplyFailsOnWrongNumberOfCols)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 1});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx->apply(x, y), gko::DimensionMismatch);
}
