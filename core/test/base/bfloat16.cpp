// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <bitset>
#include <complex>
#include <string>

#include <gtest/gtest.h>

#include <ginkgo/core/base/bfloat16.hpp>

#include "core/test/base/floating_bit_helper.hpp"


using bfloat16 = gko::bfloat16;
using namespace floating_bit_helper;


TEST(Bfloat16, SizeAndAlign)
{
    ASSERT_EQ(sizeof(bfloat16), sizeof(std::uint16_t));
    ASSERT_EQ(alignof(bfloat16), alignof(std::uint16_t));
}


// clang-format does terrible formatting of string literal concatenation
// clang-format off


TEST(FloatToBFloat16, ConvertsOne)
{
    bfloat16 x = create_from_bits<float>("0" "01111111" "00000000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "01111111" "0000000"));
}


TEST(FloatToBFloat16, ConvertsZero)
{
    bfloat16 x = create_from_bits<float>("0" "00000000" "00000000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "00000000" "0000000"));
}


TEST(FloatToBFloat16, ConvertsInf)
{
    bfloat16 x = create_from_bits<float>("0" "11111111" "00000000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111111" "0000000"));
}


TEST(FloatToBFloat16, ConvertsNegInf)
{
    bfloat16 x = create_from_bits<float>("1" "11111111" "00000000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111111" "0000000"));
}


TEST(FloatToBFloat16, ConvertsNan)
{
    bfloat16 x = create_from_bits<float>("0" "11111111" "00000000000000000000001");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111111" "1111111"));
}


TEST(FloatToBFloat16, ConvertsNegNan)
{
    bfloat16 x = create_from_bits<float>("1" "11111111" "00010000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111111" "1111111"));
}


TEST(FloatToBFloat16, FlushesToZero)
{
    bfloat16 x = create_from_bits<float>("0" "00000000" "00000000000100000001000");

    ASSERT_EQ(get_bits(x), get_bits("0" "00000000" "0000000"));
}


TEST(FloatToBFloat16, FlushesToNegZero)
{
    bfloat16 x = create_from_bits<float>("1" "00000000" "00000000000100000001000");

    ASSERT_EQ(get_bits(x), get_bits("1" "00000000" "0000000"));
}


TEST(FloatToBFloat16, FlushesToInf)
{
    bfloat16 x = create_from_bits<float>("0" "11111110" "11111111111111111111111");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111111" "0000000"));
}


TEST(FloatToBFloat16, FlushesToNegInf)
{
    bfloat16 x = create_from_bits<float>("1" "11111110" "11111111111111111111111");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111111" "0000000"));
}


TEST(FloatToBFloat16, TruncatesSmallNumber)
{
    bfloat16 x = create_from_bits<float>("0" "01110001" "10010000000000010000100");

    ASSERT_EQ(get_bits(x), get_bits("0" "01110001" "1001000"));
}


TEST(FloatToBFloat16, TruncatesLargeNumberRoundToEven)
{
    bfloat16 neg_x = create_from_bits<float>("1" "10001110" "10010111111000010000100");
    bfloat16 neg_x2 = create_from_bits<float>("1" "10001110" "10010101111000010000100");
    bfloat16 x = create_from_bits<float>("0" "10001110"  "10010111111000010000100");
    bfloat16 x2 = create_from_bits<float>("0" "10001110" "10010101111000010000100");
    bfloat16 x3 = create_from_bits<float>("0" "10001110" "10010101000000000000000");
    bfloat16 x4 = create_from_bits<float>("0" "10001110" "10010111000000000000000");

    EXPECT_EQ(get_bits(x), get_bits("0" "10001110" "1001100"));
    EXPECT_EQ(get_bits(x2), get_bits("0" "10001110" "1001011"));
    EXPECT_EQ(get_bits(x3), get_bits("0" "10001110" "1001010"));
    EXPECT_EQ(get_bits(x4), get_bits("0" "10001110" "1001100"));
    EXPECT_EQ(get_bits(neg_x), get_bits("1" "10001110" "1001100"));
    EXPECT_EQ(get_bits(neg_x2), get_bits("1" "10001110" "1001011"));
}


TEST(Bfloat16ToFloat, ConvertsOne)
{
    float x = create_from_bits<bfloat16>("0" "01111111" "0000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "01111111" "00000000000000000000000"));
}


TEST(Bfloat16ToFloat, ConvertsZero)
{
    float x = create_from_bits<bfloat16>("0" "00000000" "0000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "00000000" "00000000000000000000000"));
}


TEST(Bfloat16ToFloat, ConvertsInf)
{
    float x = create_from_bits<bfloat16>("0" "11111111" "0000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111111" "00000000000000000000000"));
}


TEST(Bfloat16ToFloat, ConvertsNegInf)
{
    float x = create_from_bits<bfloat16>("1" "11111111" "0000000");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111111" "00000000000000000000000"));
}


TEST(Bfloat16ToFloat, ConvertsNan)
{
    float x = create_from_bits<bfloat16>("0" "11111111" "0001001");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111111" "11111111111111111111111"));
}


TEST(Bfloat16ToFloat, ConvertsNegNan)
{
    float x = create_from_bits<bfloat16>("1" "11111111" "0000001");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111111" "11111111111111111111111"));
}


TEST(Bfloat16ToFloat, ExtendsSmallNumber)
{
    float x = create_from_bits<bfloat16>("0" "01110001" "1000010");

    ASSERT_EQ(get_bits(x), get_bits("0" "01110001" "10000100000000000000000"));
}


TEST(Bfloat16ToFloat, ExtendsLargeNumber)
{
    float x = create_from_bits<bfloat16>("1" "10001110" "1001001");

    ASSERT_EQ(get_bits(x), get_bits("1" "10001110" "10010010000000000000000"));
}


// clang-format on
