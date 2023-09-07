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

#include "core/base/extended_float.hpp"


#include <bitset>
#include <complex>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/math.hpp>

namespace {


template <std::size_t N>
struct floating_impl;

template <>
struct floating_impl<16> {
    using type = gko::bfloat16;
};

template <>
struct floating_impl<32> {
    using type = float;
};

template <>
struct floating_impl<64> {
    using type = double;
};

template <std::size_t N>
using floating = typename floating_impl<N>::type;


class ExtendedFloatTestBase : public ::testing::Test {
protected:
    using bfloat16 = gko::bfloat16;
    template <typename T, std::size_t NumComponents, std::size_t ComponentId>
    using truncated = gko::truncated<T, NumComponents, ComponentId>;

    static constexpr auto byte_size = gko::byte_size;

    template <std::size_t N>
    static floating<N - 1> create_from_bits(const char (&s)[N])
    {
        auto bits = std::bitset<N - 1>(s).to_ullong();
        return reinterpret_cast<floating<N - 1>&>(bits);
    }

    template <typename T>
    static std::bitset<sizeof(T) * byte_size> get_bits(T val)
    {
        auto bits =
            reinterpret_cast<typename gko::detail::float_traits<T>::bits_type&>(
                val);
        return std::bitset<sizeof(T) * byte_size>(bits);
    }

    template <std::size_t N>
    static std::bitset<N - 1> get_bits(const char (&s)[N])
    {
        return std::bitset<N - 1>(s);
    }
};


class FloatToBFloat16 : public ExtendedFloatTestBase {};


// clang-format does terrible formatting of string literal concatenation
// clang-format off


TEST_F(FloatToBFloat16, ConvertsOne)
{
    bfloat16 x = create_from_bits("0" "01111111" "00000000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "01111111" "0000000"));
}


TEST_F(FloatToBFloat16, ConvertsZero)
{
    bfloat16 x = create_from_bits("0" "00000000" "00000000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "00000000" "0000000"));
}


TEST_F(FloatToBFloat16, ConvertsInf)
{
    bfloat16 x = create_from_bits("0" "11111111" "00000000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111111" "0000000"));
}


TEST_F(FloatToBFloat16, ConvertsNegInf)
{
    bfloat16 x = create_from_bits("1" "11111111" "00000000000000000000000");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111111" "0000000"));
}


TEST_F(FloatToBFloat16, ConvertsNan)
{
    bfloat16 x = create_from_bits("0" "11111111" "00000000000000000000001");

    #if defined(SYCL_LANGUAGE_VERSION) && \
    (__LIBSYCL_MAJOR_VERSION > 5 || (__LIBSYCL_MAJOR_VERSION == 5 && __LIBSYCL_MINOR_VERSION >= 7))
    // Sycl put the 1000000000, but ours put mask
    ASSERT_EQ(get_bits(x), get_bits("0" "11111111" "1000000"));
    #else
    ASSERT_EQ(get_bits(x), get_bits("0" "11111111" "1111111"));
    #endif
}


TEST_F(FloatToBFloat16, ConvertsNegNan)
{
    bfloat16 x = create_from_bits("1" "11111111" "00010000000000000000000");

    #if defined(SYCL_LANGUAGE_VERSION) && \
    (__LIBSYCL_MAJOR_VERSION > 5 || (__LIBSYCL_MAJOR_VERSION == 5 && __LIBSYCL_MINOR_VERSION >= 7))
    // Sycl put the 1000000000, but ours put mask
    ASSERT_EQ(get_bits(x), get_bits("1" "11111111" "1000000"));
    #else
    ASSERT_EQ(get_bits(x), get_bits("1" "11111111" "1111111"));
    #endif
}


TEST_F(FloatToBFloat16, FlushesToZero)
{
    bfloat16 x = create_from_bits("0" "00000000" "00000000000100000001000");

    ASSERT_EQ(get_bits(x), get_bits("0" "00000000" "0000000"));
}


TEST_F(FloatToBFloat16, FlushesToNegZero)
{
    bfloat16 x = create_from_bits("1" "00000000" "00000000000100000001000");

    ASSERT_EQ(get_bits(x), get_bits("1" "00000000" "0000000"));
}


TEST_F(FloatToBFloat16, FlushesToInf)
{
    bfloat16 x = create_from_bits("0" "11111110" "11111111111111111111111");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111111" "0000000"));
}


TEST_F(FloatToBFloat16, FlushesToNegInf)
{
    bfloat16 x = create_from_bits("1" "11111110" "11111111111111111111111");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111111" "0000000"));
}


TEST_F(FloatToBFloat16, TruncatesSmallNumber)
{
    bfloat16 x = create_from_bits("0" "01110001" "10010000000000010000100");

    ASSERT_EQ(get_bits(x), get_bits("0" "01110001" "1001000"));
}


TEST_F(FloatToBFloat16, TruncatesLargeNumberRoundToEven)
{
    bfloat16 neg_x = create_from_bits("1" "10001110" "10010111111000010000100");
    bfloat16 neg_x2 = create_from_bits("1" "10001110" "10010101111000010000100");
    bfloat16 x = create_from_bits("0" "10001110"  "10010111111000010000100");
    bfloat16 x2 = create_from_bits("0" "10001110" "10010101111000010000100");
    bfloat16 x3 = create_from_bits("0" "10001110" "10010101000000000000000");
    bfloat16 x4 = create_from_bits("0" "10001110" "10010111000000000000000");

    EXPECT_EQ(get_bits(x), get_bits("0" "10001110" "1001100"));
    EXPECT_EQ(get_bits(x2), get_bits("0" "10001110" "1001011"));
    EXPECT_EQ(get_bits(x3), get_bits("0" "10001110" "1001010"));
    EXPECT_EQ(get_bits(x4), get_bits("0" "10001110" "1001100"));
    EXPECT_EQ(get_bits(neg_x), get_bits("1" "10001110" "1001100"));
    EXPECT_EQ(get_bits(neg_x2), get_bits("1" "10001110" "1001011"));
}


TEST_F(FloatToBFloat16, Convert)
{
    float rho = 86.25;
    float beta = 1110;
    auto float_res = rho/beta;
    gko::bfloat16 rho_h = rho;
    gko::bfloat16 beta_h = beta;
    auto bfloat16_res = rho_h/beta_h;
    std::cout << float_res << std::endl;
    std::cout << float(bfloat16_res) << std::endl;

    std::complex<gko::bfloat16> cpx{100.0, 0.0};
    std::cout << float(gko::squared_norm(cpx)) << std::endl;
}

// clang-format on


class bfloat16ToFloat : public ExtendedFloatTestBase {};


// clang-format off


TEST_F(bfloat16ToFloat, ConvertsOne)
{
    float x = create_from_bits("0" "01111111" "0000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "01111111" "00000000000000000000000"));
}


TEST_F(bfloat16ToFloat, ConvertsZero)
{
    float x = create_from_bits("0" "00000000" "0000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "00000000" "00000000000000000000000"));
}


TEST_F(bfloat16ToFloat, ConvertsInf)
{
    float x = create_from_bits("0" "11111111" "0000000");

    ASSERT_EQ(get_bits(x), get_bits("0" "11111111" "00000000000000000000000"));
}


TEST_F(bfloat16ToFloat, ConvertsNegInf)
{
    float x = create_from_bits("1" "11111111" "0000000");

    ASSERT_EQ(get_bits(x), get_bits("1" "11111111" "00000000000000000000000"));
}


TEST_F(bfloat16ToFloat, ConvertsNan)
{
    float x = create_from_bits("0" "11111111" "0001001");

    #if defined(SYCL_LANGUAGE_VERSION) && \
    (__LIBSYCL_MAJOR_VERSION > 5 || (__LIBSYCL_MAJOR_VERSION == 5 && __LIBSYCL_MINOR_VERSION >= 7))
    // sycl keeps significand
    ASSERT_EQ(get_bits(x), get_bits("0" "11111111" "00010010000000000000000"));
    #else
    ASSERT_EQ(get_bits(x), get_bits("0" "11111111" "11111111111111111111111"));
    #endif
}


TEST_F(bfloat16ToFloat, ConvertsNegNan)
{
    float x = create_from_bits("1" "11111111" "0000001");

    #if defined(SYCL_LANGUAGE_VERSION) && \
    (__LIBSYCL_MAJOR_VERSION > 5 || (__LIBSYCL_MAJOR_VERSION == 5 && __LIBSYCL_MINOR_VERSION >= 7))
    // sycl keeps significand
    ASSERT_EQ(get_bits(x), get_bits("1" "11111111" "00000010000000000000000"));
    #else
    ASSERT_EQ(get_bits(x), get_bits("1" "11111111" "11111111111111111111111"));
    #endif
}


TEST_F(bfloat16ToFloat, ExtendsSmallNumber)
{
    float x = create_from_bits("0" "01110001" "1000010");

    ASSERT_EQ(get_bits(x), get_bits("0" "01110001" "10000100000000000000000"));
}


TEST_F(bfloat16ToFloat, ExtendsLargeNumber)
{
    float x = create_from_bits("1" "10001110" "1001001");

    ASSERT_EQ(get_bits(x), get_bits("1" "10001110" "10010010000000000000000"));
}


// clang-format on


}  // namespace
