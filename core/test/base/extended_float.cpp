// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/extended_float.hpp"

#include <bitset>
#include <string>

#include <gtest/gtest.h>

#include "core/test/base/floating_bit_helper.hpp"


using namespace floating_bit_helper;

using half = gko::half;

template <typename T, std::size_t NumComponents, std::size_t ComponentId>
using truncated = gko::truncated<T, NumComponents, ComponentId>;


// clang-format does terrible formatting of string literal concatenation
// clang-format off


TEST(TruncatedDouble, SplitsDoubleToHalves)
{
    double x = create_from_bits<double>("1" "11110100100" "1111" "1000110110110101"
                                "1100101011010101" "1001011101110111");

    auto p1 = static_cast<truncated<double, 2, 0>>(x);
    auto p2 = static_cast<truncated<double, 2, 1>>(x);

    EXPECT_EQ(
        get_bits(p1), get_bits("1" "11110100100" "1111" "1000110110110101"));
    ASSERT_EQ(get_bits(p2), get_bits("1100101011010101" "1001011101110111"));
}


TEST(TruncatedDouble, AssemblesDoubleFromHalves)
{
    double x = create_from_bits<double>("1" "11110100100" "1111" "1000110110110101"
                                "1100101011010101" "1001011101110111");
    auto p1 = static_cast<truncated<double, 2, 0>>(x);
    auto p2 = static_cast<truncated<double, 2, 1>>(x);

    auto d1 = static_cast<double>(p1);
    auto d2 = static_cast<double>(p2);

    EXPECT_EQ(get_bits(d1),
              get_bits("1" "11110100100" "1111" "1000110110110101"
                       "0000000000000000" "0000000000000000"));
    ASSERT_EQ(get_bits(d2),
              get_bits("0" "00000000000" "0000" "0000000000000000"
                       "1100101011010101" "1001011101110111"));
}


TEST(TruncatedDouble, SplitsDoubleToQuarters)
{
    double x = create_from_bits<double>("1" "11110100100" "1111" "1000110110110101"
                                "1100101011010101" "1001011101110111");

    auto p1 = static_cast<truncated<double, 4, 0>>(x);
    auto p2 = static_cast<truncated<double, 4, 1>>(x);
    auto p3 = static_cast<truncated<double, 4, 2>>(x);
    auto p4 = static_cast<truncated<double, 4, 3>>(x);

    EXPECT_EQ(get_bits(p1), get_bits("1" "11110100100" "1111"));
    EXPECT_EQ(get_bits(p2), get_bits("1000110110110101"));
    EXPECT_EQ(get_bits(p3), get_bits("1100101011010101"));
    ASSERT_EQ(get_bits(p4), get_bits("1001011101110111"));
}


TEST(TruncatedDouble, AssemblesDoubleFromQuarters)
{
    double x = create_from_bits<double>("1" "11110100100" "1111" "1000110110110101"
                                "1100101011010101" "1001011101110111");
    auto p1 = static_cast<truncated<double, 4, 0>>(x);
    auto p2 = static_cast<truncated<double, 4, 1>>(x);
    auto p3 = static_cast<truncated<double, 4, 2>>(x);
    auto p4 = static_cast<truncated<double, 4, 3>>(x);

    auto d1 = static_cast<double>(p1);
    auto d2 = static_cast<double>(p2);
    auto d3 = static_cast<double>(p3);
    auto d4 = static_cast<double>(p4);

    ASSERT_EQ(get_bits(d1),
              get_bits("1" "11110100100" "1111" "0000000000000000"
                       "0000000000000000" "0000000000000000"));
    ASSERT_EQ(get_bits(d2),
              get_bits("0" "00000000000" "0000" "1000110110110101"
                       "0000000000000000" "0000000000000000"));
    ASSERT_EQ(get_bits(d3),
              get_bits("0" "00000000000" "0000" "0000000000000000"
                       "1100101011010101" "0000000000000000"));
    ASSERT_EQ(get_bits(d4),
              get_bits("0" "00000000000" "0000" "0000000000000000"
                       "0000000000000000" "1001011101110111"));
}


TEST(TruncatedFloat, SplitsFloatToHalves)
{
    float x = create_from_bits<float>("1" "11110100" "1001111" "1000110110110101");

    auto p1 = static_cast<truncated<float, 2, 0>>(x);
    auto p2 = static_cast<truncated<float, 2, 1>>(x);

    EXPECT_EQ(get_bits(p1), get_bits("1" "11110100" "1001111"));
    ASSERT_EQ(get_bits(p2), get_bits("1000110110110101"));
}


TEST(TruncatedFloat, AssemblesFloatFromHalves)
{
    float x = create_from_bits<float>("1" "11110100" "1001111" "1000110110110101");
    auto p1 = static_cast<truncated<float, 2, 0>>(x);
    auto p2 = static_cast<truncated<float, 2, 1>>(x);

    auto d1 = static_cast<float>(p1);
    auto d2 = static_cast<float>(p2);

    EXPECT_EQ(
        get_bits(d1), get_bits("1" "11110100" "1001111" "0000000000000000"));
    ASSERT_EQ(
        get_bits(d2), get_bits("0" "00000000" "0000000" "1000110110110101"));
}


// clang-format on
