// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <complex>

#include <gtest/gtest.h>

#include <ginkgo/core/base/math.hpp>

#include "core/test/utils.hpp"


TEST(HighestPrecision, SamePrecisionsGivesSame)
{
    ::testing::StaticAssertTypeEq<gko::highest_precision<gko::half, gko::half>,
                                  gko::half>();
    ::testing::StaticAssertTypeEq<
        gko::highest_precision<gko::bfloat16, gko::bfloat16>, gko::bfloat16>();
    ::testing::StaticAssertTypeEq<gko::highest_precision<float, float>,
                                  float>();
    ::testing::StaticAssertTypeEq<gko::highest_precision<double, double>,
                                  double>();
}

TEST(HighestPrecision, Different16bitGivesFloat)
{
    ::testing::StaticAssertTypeEq<
        gko::highest_precision<gko::half, gko::bfloat16>, float>();
    ::testing::StaticAssertTypeEq<
        gko::highest_precision<gko::bfloat16, gko::half>, float>();
}

TEST(HighestPrecision, HalfCombinations)
{
    // two same precisions give the same precision
    ::testing::StaticAssertTypeEq<gko::highest_precision<gko::half, gko::half>,
                                  gko::half>();
    // different 16 bit precisions give float
    ::testing::StaticAssertTypeEq<
        gko::highest_precision<gko::half, gko::bfloat16>, float>();
    ::testing::StaticAssertTypeEq<gko::highest_precision<gko::half, float>,
                                  float>();
    ::testing::StaticAssertTypeEq<gko::highest_precision<gko::half, double>,
                                  double>();
}


TEST(HighestPrecision, Bfloat16Combinations)
{
    // different 16 bit precisions give float
    ::testing::StaticAssertTypeEq<
        gko::highest_precision<gko::bfloat16, gko::half>, float>();
    // two same precisions give the same precision
    ::testing::StaticAssertTypeEq<
        gko::highest_precision<gko::bfloat16, gko::bfloat16>, gko::bfloat16>();
    ::testing::StaticAssertTypeEq<gko::highest_precision<gko::bfloat16, float>,
                                  float>();
    ::testing::StaticAssertTypeEq<gko::highest_precision<gko::bfloat16, double>,
                                  double>();
}


TEST(HighestPrecision, FloatCombinations)
{
    ::testing::StaticAssertTypeEq<gko::highest_precision<float, gko::half>,
                                  float>();
    ::testing::StaticAssertTypeEq<gko::highest_precision<float, gko::bfloat16>,
                                  float>();
    // two same precisions give the same precision
    ::testing::StaticAssertTypeEq<gko::highest_precision<float, float>,
                                  float>();
    ::testing::StaticAssertTypeEq<gko::highest_precision<float, double>,
                                  double>();
}


TEST(HighestPrecision, DoubleCombinations)
{
    ::testing::StaticAssertTypeEq<gko::highest_precision<double, gko::half>,
                                  double>();
    ::testing::StaticAssertTypeEq<gko::highest_precision<double, gko::bfloat16>,
                                  double>();
    ::testing::StaticAssertTypeEq<gko::highest_precision<double, float>,
                                  double>();
    // two same precisions give the same precision
    ::testing::StaticAssertTypeEq<gko::highest_precision<double, double>,
                                  double>();
}


template <typename TwoRealType>
class ComplexHigestPrecision : public ::testing::Test {
public:
    using first_type =
        typename std::tuple_element<0, decltype(TwoRealType())>::type;
    using second_type =
        typename std::tuple_element<0, decltype(TwoRealType())>::type;
};

using TwoRealType =
    gko::test::cartesian_type_product_t<gko::test::RealValueTypes,
                                        gko::test::RealValueTypes>;

TYPED_TEST_SUITE(ComplexHigestPrecision, TwoRealType,
                 PairTypenameNameGenerator);


TYPED_TEST(ComplexHigestPrecision, ComplexBasedOnReal)
{
    using first_type = typename TestFixture::first_type;
    using second_type = typename TestFixture::second_type;
    ::testing::StaticAssertTypeEq<
        gko::highest_precision<std::complex<first_type>,
                               std::complex<second_type>>,
        std::complex<gko::highest_precision<first_type, second_type>>>();
}
