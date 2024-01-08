// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/types.hpp>


#include <array>
#include <cstdint>
#include <stdexcept>
#include <type_traits>


#include <gtest/gtest.h>


#include "core/base/types.hpp"


namespace {


TEST(PrecisionReduction, CreatesDefaultEncoding)
{
    auto e = gko::precision_reduction();

    ASSERT_EQ(e.get_preserving(), 0);
    ASSERT_EQ(e.get_nonpreserving(), 0);
}


TEST(PrecisionReduction, CreatesCustomEncoding)
{
    auto e = gko::precision_reduction(2, 4);

    ASSERT_EQ(e.get_preserving(), 2);
    ASSERT_EQ(e.get_nonpreserving(), 4);
}


TEST(PrecisionReduction, ComparesEncodings)
{
    auto x = gko::precision_reduction(1, 2);
    auto y = gko::precision_reduction(1, 2);
    auto z = gko::precision_reduction(3, 1);

    ASSERT_TRUE(x == y);
    ASSERT_TRUE(!(x == z));
    ASSERT_TRUE(!(y == z));
    ASSERT_TRUE(!(x != y));
    ASSERT_TRUE(x != z);
    ASSERT_TRUE(y != z);
}


TEST(PrecisionReduction, CreatesAutodetectEncoding)
{
    auto ad = gko::precision_reduction::autodetect();

    ASSERT_NE(ad, gko::precision_reduction());
    ASSERT_NE(ad, gko::precision_reduction(1, 2));
}


TEST(PrecisionReduction, ConvertsToStorageType)
{
    auto st = static_cast<gko::precision_reduction::storage_type>(
        gko::precision_reduction{});

    ASSERT_EQ(st, 0);
}


TEST(PrecisionReduction, ComputesCommonEncoding)
{
    auto e1 = gko::precision_reduction(2, 3);
    auto e2 = gko::precision_reduction(3, 1);

    ASSERT_EQ(gko::precision_reduction::common(e1, e2),
              gko::precision_reduction(2, 1));
}


TEST(ConfigSet, MaskCorrectly)
{
    constexpr auto mask3_u = gko::detail::mask<3>();
    constexpr auto fullmask_u = gko::detail::mask<32>();
    constexpr auto mask3_u64 = gko::detail::mask<3, std::uint64_t>();
    constexpr auto fullmask_u64 = gko::detail::mask<64, std::uint64_t>();

    ASSERT_EQ(mask3_u, 7u);
    ASSERT_EQ(fullmask_u, 0xffffffffu);
    ASSERT_TRUE((std::is_same<decltype(mask3_u), const std::uint32_t>::value));
    ASSERT_TRUE(
        (std::is_same<decltype(fullmask_u), const std::uint32_t>::value));
    ASSERT_EQ(mask3_u64, 7ull);
    ASSERT_EQ(fullmask_u64, 0xffffffffffffffffull);
    ASSERT_TRUE(
        (std::is_same<decltype(mask3_u64), const std::uint64_t>::value));
    ASSERT_TRUE(
        (std::is_same<decltype(fullmask_u64), const std::uint64_t>::value));
}


TEST(ConfigSet, ShiftCorrectly)
{
    constexpr std::array<unsigned char, 3> bits{3, 5, 7};


    constexpr auto shift0 = gko::detail::shift<0, 3>(bits);
    constexpr auto shift1 = gko::detail::shift<1, 3>(bits);
    constexpr auto shift2 = gko::detail::shift<2, 3>(bits);

    ASSERT_EQ(shift0, 12);
    ASSERT_EQ(shift1, 7);
    ASSERT_EQ(shift2, 0);
}


TEST(ConfigSet, ConfigSet1Correctly)
{
    using Cfg = gko::ConfigSet<3>;

    constexpr auto encoded = Cfg::encode(2);
    constexpr auto decoded = Cfg::decode<0>(encoded);

    ASSERT_EQ(encoded, 2);
    ASSERT_EQ(decoded, 2);
}


TEST(ConfigSet, ConfigSet1FullCorrectly)
{
    using Cfg = gko::ConfigSet<32>;

    constexpr auto encoded = Cfg::encode(0xffffffff);
    constexpr auto decoded = Cfg::decode<0>(encoded);

    ASSERT_EQ(encoded, 0xffffffff);
    ASSERT_EQ(decoded, 0xffffffff);
}


TEST(ConfigSet, ConfigSet2FullCorrectly)
{
    using Cfg = gko::ConfigSet<1, 31>;

    constexpr auto encoded = Cfg::encode(1, 33);

    ASSERT_EQ(encoded, (1u << 31) + 33);
}


TEST(ConfigSet, ConfigSetSomeCorrectly)
{
    using Cfg = gko::ConfigSet<3, 5, 7>;

    constexpr auto encoded = Cfg::encode(2, 11, 13);
    constexpr auto decoded_0 = Cfg::decode<0>(encoded);
    constexpr auto decoded_1 = Cfg::decode<1>(encoded);
    constexpr auto decoded_2 = Cfg::decode<2>(encoded);

    ASSERT_EQ(encoded, (2 << 12) + (11 << 7) + 13);
    ASSERT_EQ(decoded_0, 2);
    ASSERT_EQ(decoded_1, 11);
    ASSERT_EQ(decoded_2, 13);
}


TEST(ConfigSet, ConfigSetSomeFullCorrectly)
{
    using Cfg = gko::ConfigSet<2, 6, 7, 17>;

    constexpr auto encoded = Cfg::encode(2, 11, 13, 19);
    constexpr auto decoded_0 = Cfg::decode<0>(encoded);
    constexpr auto decoded_1 = Cfg::decode<1>(encoded);
    constexpr auto decoded_2 = Cfg::decode<2>(encoded);
    constexpr auto decoded_3 = Cfg::decode<3>(encoded);

    ASSERT_EQ(encoded, (2 << 30) + (11 << 24) + (13 << 17) + 19);
    ASSERT_EQ(decoded_0, 2);
    ASSERT_EQ(decoded_1, 11);
    ASSERT_EQ(decoded_2, 13);
    ASSERT_EQ(decoded_3, 19);
}


}  // namespace
