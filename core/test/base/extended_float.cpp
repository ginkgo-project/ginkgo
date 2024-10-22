// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/extended_float.hpp"

#include <bitset>
#include <string>

#include <gtest/gtest.h>

#include <ginkgo/core/base/half.hpp>


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

template <>
struct floating_impl<64> {
    using type = double;
};

template <std::size_t N>
using floating = typename floating_impl<N>::type;


class ExtendedFloatTestBase : public ::testing::Test {
protected:
    using half = gko::half;
    template <typename T, std::size_t NumComponents, std::size_t ComponentId>
    using truncated = gko::truncated<T, NumComponents, ComponentId>;

    static constexpr auto byte_size = gko::byte_size;

    template <std::size_t N>
    static floating<N - 1> create_from_bits(const char (&s)[N])
    {
        auto bits = std::bitset<N - 1>(s).to_ullong();
        // We cast to the same size of integer type first.
        // Otherwise, the first memory chunk is different when we use
        // reinterpret_cast or memcpy to get the smaller type out of unsigned
        // long long.
        using bits_type =
            typename gko::detail::float_traits<floating<N - 1>>::bits_type;
        auto bits_val = static_cast<bits_type>(bits);
        floating<N - 1> result;
        static_assert(sizeof(floating<N - 1>) == sizeof(bits_type),
                      "the type should have the same size as its bits_type");
        std::memcpy(&result, &bits_val, sizeof(bits_type));
        return result;
    }

    template <typename T>
    static std::bitset<sizeof(T) * byte_size> get_bits(T val)
    {
        using bits_type = typename gko::detail::float_traits<T>::bits_type;
        bits_type bits;
        static_assert(sizeof(T) == sizeof(bits_type),
                      "the type should have the same size as its bits_type");
        std::memcpy(&bits, &val, sizeof(T));
        return std::bitset<sizeof(T) * byte_size>(bits);
    }

    template <std::size_t N>
    static std::bitset<N - 1> get_bits(const char (&s)[N])
    {
        return std::bitset<N - 1>(s);
    }
};


class TruncatedDouble : public ExtendedFloatTestBase {};

// clang-format does terrible formatting of string literal concatenation
// clang-format off


TEST_F(TruncatedDouble, SplitsDoubleToHalves)
{
    double x = create_from_bits("1" "11110100100" "1111" "1000110110110101"
                                "1100101011010101" "1001011101110111");

    auto p1 = static_cast<truncated<double, 2, 0>>(x);
    auto p2 = static_cast<truncated<double, 2, 1>>(x);

    EXPECT_EQ(
        get_bits(p1), get_bits("1" "11110100100" "1111" "1000110110110101"));
    ASSERT_EQ(get_bits(p2), get_bits("1100101011010101" "1001011101110111"));
}


TEST_F(TruncatedDouble, AssemblesDoubleFromHalves)
{
    double x = create_from_bits("1" "11110100100" "1111" "1000110110110101"
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


TEST_F(TruncatedDouble, SplitsDoubleToQuarters)
{
    double x = create_from_bits("1" "11110100100" "1111" "1000110110110101"
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


TEST_F(TruncatedDouble, AssemblesDoubleFromQuarters)
{
    double x = create_from_bits("1" "11110100100" "1111" "1000110110110101"
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


// clang-format on


class TruncatedFloat : public ExtendedFloatTestBase {};


// clang-format off


TEST_F(TruncatedFloat, SplitsFloatToHalves)
{
    float x = create_from_bits("1" "11110100" "1001111" "1000110110110101");

    auto p1 = static_cast<truncated<float, 2, 0>>(x);
    auto p2 = static_cast<truncated<float, 2, 1>>(x);

    EXPECT_EQ(get_bits(p1), get_bits("1" "11110100" "1001111"));
    ASSERT_EQ(get_bits(p2), get_bits("1000110110110101"));
}


TEST_F(TruncatedFloat, AssemblesFloatFromHalves)
{
    float x = create_from_bits("1" "11110100" "1001111" "1000110110110101");
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


}  // namespace
