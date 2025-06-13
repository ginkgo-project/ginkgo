// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>

#include <gtest/gtest.h>

#include <ginkgo/core/base/half.hpp>

#include "core/test/base/floating_bit_helper.hpp"


using half = gko::half;
using namespace floating_bit_helper;


TEST(Half, SizeAndAlign)
{
    ASSERT_EQ(sizeof(half), sizeof(std::uint16_t));
    ASSERT_EQ(alignof(half), alignof(std::uint16_t));
}

// clang-format does terrible formatting of string literal concatenation
// clang-format off


TEST(FloatToHalf, ConvertsOne)
{
    half x = create_from_bits<float>("0" "01111111" "00000000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "01111" "0000000000"));
}


TEST(FloatToHalf, ConvertsZero)
{
    half x = create_from_bits<float>("0" "00000000" "00000000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "00000" "0000000000"));
}


TEST(FloatToHalf, ConvertsInf)
{
    half x = create_from_bits<float>("0" "11111111" "00000000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111" "0000000000"));
}


TEST(FloatToHalf, ConvertsNegInf)
{
    half x = create_from_bits<float>("1" "11111111" "00000000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111" "0000000000"));
}


TEST(FloatToHalf, ConvertsNan)
{
    half x = create_from_bits<float>("0" "11111111" "00000000000000000000001");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111" "1111111111"));
}


TEST(FloatToHalf, ConvertsNegNan)
{
    half x = create_from_bits<float>("1" "11111111" "00010000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111" "1111111111"));
}


TEST(FloatToHalf, FlushesToZero)
{
    half x = create_from_bits<float>("0" "00000111" "00010001000100000001000");

    ASSERT_EQ(get_bits(x), get_bits("0" "00000" "0000000000"));
}


TEST(FloatToHalf, FlushesToNegZero)
{
    half x = create_from_bits<float>("1" "00000010" "00010001000100000001000");

    ASSERT_EQ(get_bits(x), get_bits("1" "00000" "0000000000"));
}


TEST(FloatToHalf, FlushesToInf)
{
    half x = create_from_bits<float>("0" "10100000" "10010000000000010000100");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111" "0000000000"));
}


TEST(FloatToHalf, FlushesToNegInf)
{
    half x = create_from_bits<float>("1" "11000000" "10010000000000010000100");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111" "0000000000"));
}


TEST(FloatToHalf, TruncatesSmallNumber)
{
    half x = create_from_bits<float>("0" "01110001" "10010000000000010000100");

    ASSERT_EQ(get_bits(x), get_bits("0" "00001" "1001000000"));
}


TEST(FloatToHalf, TruncatesLargeNumberRoundToEven)
{
    half neg_x = create_from_bits<float>("1" "10001110" "10010011111000010000100");
    half neg_x2 = create_from_bits<float>("1" "10001110" "10010011101000010000100");
    half x = create_from_bits<float>("0" "10001110" "10010011111000010000100");
    half x2 = create_from_bits<float>("0" "10001110" "10010011101000010000100");
    half x3 = create_from_bits<float>("0" "10001110" "10010011101000000000000");
    half x4 = create_from_bits<float>("0" "10001110" "10010011111000000000000");

    EXPECT_EQ(get_bits(x), get_bits("0" "11110" "1001010000"));
    EXPECT_EQ(get_bits(x2), get_bits("0" "11110" "1001001111"));
    EXPECT_EQ(get_bits(x3), get_bits("0" "11110" "1001001110"));
    EXPECT_EQ(get_bits(x4), get_bits("0" "11110" "1001010000"));
    EXPECT_EQ(get_bits(neg_x), get_bits("1" "11110" "1001010000"));
    EXPECT_EQ(get_bits(neg_x2), get_bits("1" "11110" "1001001111"));
}


TEST(HalfToFloat, ConvertsOne)
{
    float x = create_from_bits<half>("0" "01111" "0000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "01111111" "00000000000000000000000"));
}


TEST(HalfToFloat, ConvertsZero)
{
    float x = create_from_bits<half>("0" "00000" "0000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "00000000" "00000000000000000000000"));
}


TEST(HalfToFloat, ConvertsInf)
{
    float x = create_from_bits<half>("0" "11111" "0000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111111" "00000000000000000000000"));
}


TEST(HalfToFloat, ConvertsNegInf)
{
    float x = create_from_bits<half>("1" "11111" "0000000000");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111111" "00000000000000000000000"));
}


TEST(HalfToFloat, ConvertsNan)
{
    float x = create_from_bits<half>("0" "11111" "0001001000");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111111" "11111111111111111111111"));
}


TEST(HalfToFloat, ConvertsNegNan)
{
    float x = create_from_bits<half>("1" "11111" "0000000001");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111111" "11111111111111111111111"));
}


TEST(HalfToFloat, ExtendsSmallNumber)
{
    float x = create_from_bits<half>("0" "00001" "1000010001");

    ASSERT_EQ(get_bits(x), get_bits("0" "01110001" "10000100010000000000000"));
}


TEST(HalfToFloat, ExtendsLargeNumber)
{
    float x = create_from_bits<half>("1" "11110" "1001001111");

    ASSERT_EQ(get_bits(x), get_bits("1" "10001110" "10010011110000000000000"));
}


// clang-format on
