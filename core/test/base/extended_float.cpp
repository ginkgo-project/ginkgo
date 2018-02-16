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

#include <core/base/extended_float.hpp>


#include <gtest/gtest.h>


#include <bitset>
#include <string>


namespace {


template <std::size_t N>
struct floating_impl;

template <>
struct floating_impl<16> {
    using type = gko::half;
};

template <>
struct floating_impl<32> {
    using type = float;
};


template <std::size_t N>
using floating = typename floating_impl<N>::type;


class HalfTestBase : public ::testing::Test {
protected:
    using half = gko::half;

    static constexpr std::size_t byte_size = 8;


    template <std::size_t N>
    static floating<N - 1> create_from_bits(const char (&s)[N])
    {
        auto bits = std::bitset<N - 1>(s).to_ullong();
        return reinterpret_cast<floating<N - 1> &>(bits);
    }

    template <typename T>
    static std::bitset<sizeof(T) * byte_size> get_bits(T val)
    {
        auto bits = reinterpret_cast<
            typename gko::detail::basic_float_traits<T>::bits_type &>(val);
        return std::bitset<sizeof(T) * byte_size>(bits);
    }

    template <std::size_t N>
    static std::bitset<N - 1> get_bits(const char (&s)[N])
    {
        return std::bitset<N - 1>(s);
    }
};


class FloatToHalf : public HalfTestBase {
};


// clang-format does terrible formatting of string literal concatenation
// clang-format off


TEST_F(FloatToHalf, ConvertsOne)
{
    half x = create_from_bits("0" "01111111" "00000000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "01111" "0000000000"));
}


TEST_F(FloatToHalf, ConvertsZero)
{
    half x = create_from_bits("0" "00000000" "00000000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "00000" "0000000000"));
}


TEST_F(FloatToHalf, ConvertsInf)
{
    half x = create_from_bits("0" "11111111" "00000000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111" "0000000000"));
}


TEST_F(FloatToHalf, ConvertsNegInf)
{
    half x = create_from_bits("1" "11111111" "00000000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111" "0000000000"));
}


TEST_F(FloatToHalf, ConvertsNan)
{
    half x = create_from_bits("0" "11111111" "00000000000000000000001");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111" "1111111111"));
}


TEST_F(FloatToHalf, ConvertsNegNan)
{
    half x = create_from_bits("1" "11111111" "00010000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111" "1111111111"));
}


TEST_F(FloatToHalf, FlushesToZero)
{
    half x = create_from_bits("0" "00000111" "00010001000100000001000");

    ASSERT_EQ(get_bits(x), get_bits("0" "00000" "0000000000"));
}


TEST_F(FloatToHalf, FlushesToNegZero)
{
    half x = create_from_bits("1" "00000010" "00010001000100000001000");

    ASSERT_EQ(get_bits(x), get_bits("1" "00000" "0000000000"));
}


TEST_F(FloatToHalf, FlushesToInf)
{
    half x = create_from_bits("0" "10100000" "10010000000000010000100");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111" "0000000000"));
}


TEST_F(FloatToHalf, FlushesToNegInf)
{
    half x = create_from_bits("1" "11000000" "10010000000000010000100");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111" "0000000000"));
}


TEST_F(FloatToHalf, TruncatesSmallNumber)
{
    half x = create_from_bits("0" "01110001" "10010000000000010000100");

    ASSERT_EQ(get_bits(x), get_bits("0" "00001" "1001000000"));
}


TEST_F(FloatToHalf, TruncatesLargeNumber)
{
    half x = create_from_bits("1" "10001110" "10010011111000010000100");

    ASSERT_EQ(get_bits(x), get_bits("1" "11110" "1001001111"));

}


// clang-format on


class HalfToFloat : public HalfTestBase {
};


// clang-format off


TEST_F(HalfToFloat, ConvertsOne)
{
    float x = create_from_bits("0" "01111" "0000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "01111111" "00000000000000000000000"));
}


TEST_F(HalfToFloat, ConvertsZero)
{
    float x = create_from_bits("0" "00000" "0000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "00000000" "00000000000000000000000"));
}


TEST_F(HalfToFloat, ConvertsInf)
{
    float x = create_from_bits("0" "11111" "0000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111111" "00000000000000000000000"));
}


TEST_F(HalfToFloat, ConvertsNegInf)
{
    float x = create_from_bits("1" "11111" "0000000000");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111111" "00000000000000000000000"));
}


TEST_F(HalfToFloat, ConvertsNan)
{
    float x = create_from_bits("0" "11111" "0001001000");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111111" "11111111111111111111111"));
}


TEST_F(HalfToFloat, ConvertsNegNan)
{
    float x = create_from_bits("1" "11111" "0000000001");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111111" "11111111111111111111111"));
}


TEST_F(HalfToFloat, ExtendsSmallNumber)
{
    float x = create_from_bits("0" "00001" "1000010001");

    ASSERT_EQ(get_bits(x), get_bits("0" "01110001" "10000100010000000000000"));
}


TEST_F(HalfToFloat, ExtendsLargeNumber)
{
    float x = create_from_bits("1" "11110" "1001001111");

    ASSERT_EQ(get_bits(x), get_bits("1" "10001110" "10010011110000000000000"));
}


// clang-format on


}  // namespace
