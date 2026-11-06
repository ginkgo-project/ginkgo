// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/emulation/multi_word.hpp"

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "core/emulation/double_word.hpp"
#include "core/test/emulation/float_float_helper.hpp"
#include "core/test/utils.hpp"


namespace {


using namespace float_float_helper;


template <typename MW, mw::special Special>
struct with_special_impl;

template <typename T, int N, mw::normalization Normalization,
          mw::algorithm Algorithm, bool Fma, mw::special Old,
          mw::special Special>
struct with_special_impl<
    gko::multi_word<T, N, mw::policy<Normalization, Algorithm, Fma, Old>>,
    Special> {
    using type =
        gko::multi_word<T, N,
                        mw::policy<Normalization, Algorithm, Fma, Special>>;
};

// the same multi_word with special::handle
template <typename MW>
using handled = typename with_special_impl<MW, mw::special::handle>::type;


template <typename X, typename Y, typename = void>
struct can_add : std::false_type {};

template <typename X, typename Y>
struct can_add<X, Y,
               std::void_t<decltype(std::declval<X>() + std::declval<Y>())>>
    : std::true_type {};


// the default policy: eager normalization, accurate algorithms, FMA, special
// values ignored
static_assert(std::is_same_v<gko::multi_word<float, 2>, accurate_fma>);
static_assert(sizeof(accurate_fma) == 2 * sizeof(float));
static_assert(std::is_trivially_copyable_v<accurate_fma>);
// the words of multi_word x; are uninitialized, e.g. for __shared__ memory
static_assert(std::is_trivial_v<accurate_fma>);
static_assert(std::is_convertible_v<float, accurate_fma>);
static_assert(std::is_convertible_v<int, accurate_fma>);
// a double in general does not fit into two floats, so only explicitly
static_assert(std::is_constructible_v<accurate_fma, double>);
static_assert(!std::is_convertible_v<double, accurate_fma>);
// the conversion to floating-point types rounds in general, so only explicitly
static_assert(std::is_constructible_v<double, accurate_fma>);
static_assert(!std::is_convertible_v<accurate_fma, double>);
static_assert(std::is_constructible_v<float, accurate_fma>);
static_assert(static_cast<double>(accurate_fma::from_words(1.0f, 0x1p-30f)) ==
              1 + 0x1p-30);
static_assert(can_add<accurate_fma, float>::value);
static_assert(can_add<float, accurate_fma>::value);
static_assert(can_add<accurate_fma, int>::value);
// would silently round the double to float
static_assert(!can_add<accurate_fma, double>::value);
static_assert(!can_add<double, accurate_fma>::value);
// Ginkgo's zero<T>() is T{} and one<T>() is T(1), also in constant expressions
static_assert(gko::zero<accurate_fma>().word(0) == 0.0f &&
              gko::zero<accurate_fma>().word(1) == 0.0f);
static_assert(gko::one<accurate_fma>().word(0) == 1.0f &&
              gko::one<accurate_fma>().word(1) == 0.0f);


template <typename MW>
class MultiWord : public fixture<MW> {};

TYPED_TEST_SUITE(MultiWord, types, TypenameNameGenerator);


TYPED_TEST(MultiWord, DefaultConstructsZero)
{
    using MW = TypeParam;

    ASSERT_TRUE(has_words(MW{}, 0.0f, 0.0f));
}


TYPED_TEST(MultiWord, ConstructsFromSingleWordOrInteger)
{
    using MW = TypeParam;

    ASSERT_TRUE(has_words(MW{1.5f}, 1.5f, 0.0f));
    ASSERT_TRUE(has_words(MW{3}, 3.0f, 0.0f));
}


TYPED_TEST(MultiWord, KeepsGivenWords)
{
    using MW = TypeParam;

    const auto x = MW::from_words(1.0f, 0x1p-27f);

    ASSERT_EQ(x.word(0), 1.0f);
    ASSERT_EQ(x.word(1), 0x1p-27f);
}


TYPED_TEST(MultiWord, NegatesAllWords)
{
    using MW = TypeParam;

    ASSERT_TRUE(has_words(-MW::from_words(1.5f, 0x1p-26f), -1.5f, -0x1p-26f));
}


TYPED_TEST(MultiWord, EvaluatesInHigherPrecision)
{
    using MW = TypeParam;
    const auto x = MW::from_words(1.0f, 0x1p-30f);

    ASSERT_EQ(mw::eval<double>(x), 1 + 0x1p-30);
    ASSERT_EQ(mw::eval<float>(x), 1.0f);
}


TYPED_TEST(MultiWord, RoundsDoubleToTwoWords)
{
    using MW = TypeParam;
    constexpr auto u = TestFixture::u;
    std::uniform_real_distribution<double> significand(1, 2);

    ASSERT_TRUE(has_words(MW{1 + 0x1p-30}, 1.0f, 0x1p-30f));
    // rounding each word gives (1 + 2^-23, -2^-24), whose sum is a tie that
    // rounds to 1
    ASSERT_TRUE(has_words(MW{1 + 0x1p-24 + 0x1p-50}, 1.0f, 0x1p-24f));
    for (int i = 0; i < TestFixture::num_samples; ++i) {
        // any 53 significand bits
        const auto value = std::ldexp(significand(this->engine), i % 60 - 30);

        const auto x = MW{value};

        ASSERT_LE(std::abs(mw::eval<double>(x) - value),
                  u * u * (1 + u) * value)
            << std::hexfloat << value;
        ASSERT_TRUE(is_normalized(x));
    }
}


TYPED_TEST(MultiWord, CompoundAssignmentMatchesBinaryOperators)
{
    for (int i = 0; i < 100; ++i) {
        const auto x = this->random();
        const auto y = this->random();
        const auto a = this->random_word();
        auto z = x;

        ASSERT_TRUE(has_words(z += y, (x + y).word(0), (x + y).word(1)));
        ASSERT_TRUE(
            has_words(z -= a, (x + y - a).word(0), (x + y - a).word(1)));
        z = x;
        ASSERT_TRUE(has_words(z *= a, (x * a).word(0), (x * a).word(1)));
        ASSERT_TRUE(
            has_words(z /= a, (x * a / a).word(0), (x * a / a).word(1)));
        if constexpr (TestFixture::has_product) {
            z = x;
            ASSERT_TRUE(has_words(z *= y, (x * y).word(0), (x * y).word(1)));
            ASSERT_TRUE(
                has_words(z /= y, (x * y / y).word(0), (x * y / y).word(1)));
        }
    }
}


TYPED_TEST(MultiWord, ComparesLikeTheValues)
{
    for (int i = 0; i < TestFixture::num_samples; ++i) {
        const auto x = this->random();
        // equal, close to x and often with the same leading word, or random
        const auto y = i % 4 == 0 ? x : -this->random_partner(x);
        const auto a = i % 2 == 0 ? x.word(0) : this->random_word();
        const auto xd = mw::eval<double>(x);
        const auto yd = mw::eval<double>(y);
        const auto ad = static_cast<double>(a);

        ASSERT_EQ(x == y, xd == yd) << to_string(x) << ", " << to_string(y);
        ASSERT_EQ(x != y, xd != yd) << to_string(x) << ", " << to_string(y);
        ASSERT_EQ(x < y, xd < yd) << to_string(x) << ", " << to_string(y);
        ASSERT_EQ(x <= y, xd <= yd) << to_string(x) << ", " << to_string(y);
        ASSERT_EQ(x > y, xd > yd) << to_string(x) << ", " << to_string(y);
        ASSERT_EQ(x >= y, xd >= yd) << to_string(x) << ", " << to_string(y);
        ASSERT_EQ(x == a, xd == ad) << to_string(x) << ", " << to_string(a);
        ASSERT_EQ(x < a, xd < ad) << to_string(x) << ", " << to_string(a);
        ASSERT_EQ(a < x, ad < xd) << to_string(x) << ", " << to_string(a);
    }
}


TYPED_TEST(MultiWord, ComparesUnnormalizedPairs)
{
    using MW = TypeParam;
    if constexpr (TestFixture::pair) {
        // 2.25 * 2^-24, although the leading word is smaller than 2^-23
        const auto x = MW::from_words(0x1p-24f, 0x1.4p-24f);

        ASSERT_TRUE(x > MW{0x1p-23f});
        ASSERT_TRUE(MW{0x1p-23f} < x);
        ASSERT_TRUE(x == MW{0x1.2p-23f});
    } else {
        GTEST_SKIP() << "the comparison expects normalized words";
    }
}


TYPED_TEST(MultiWord, NanComparesUnequal)
{
    using MW = TypeParam;
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    const auto one = MW{1.0f};

    for (const auto& x : {MW{nan}, MW::from_words(1.0f, nan)}) {
        ASSERT_FALSE(x == x);
        ASSERT_TRUE(x != x);
        ASSERT_FALSE(x < one);
        ASSERT_FALSE(x <= one);
        ASSERT_FALSE(x > one);
        ASSERT_FALSE(x >= one);
        ASSERT_FALSE(one < x);
        ASSERT_FALSE(one <= x);
        ASSERT_FALSE(one > x);
        ASSERT_FALSE(one >= x);
    }
}


TYPED_TEST(MultiWord, AbsNegatesNegativeValues)
{
    using MW = TypeParam;

    ASSERT_TRUE(
        has_words(abs(MW::from_words(-1.0f, 0x1p-30f)), 1.0f, -0x1p-30f));
    ASSERT_TRUE(
        has_words(abs(MW::from_words(1.0f, -0x1p-30f)), 1.0f, -0x1p-30f));
    if constexpr (TestFixture::pair) {
        // negative, although the leading word is positive
        ASSERT_TRUE(has_words(abs(MW::from_words(0x1p-30f, -0x1p-29f)),
                              -0x1p-30f, 0x1p-29f));
    }
}


TYPED_TEST(MultiWord, ConvertsToFloatingPointTypesWithOneRounding)
{
    using MW = TypeParam;

    ASSERT_EQ(static_cast<double>(MW::from_words(1.0f, 0x1p-30f)), 1 + 0x1p-30);
    // 1 + 2^-53 is a tie, which rounds to even, 1 + 1.5 * 2^-53 rounds up
    ASSERT_EQ(static_cast<double>(MW::from_words(1.0f, 0x1p-53f)), 1.0);
    ASSERT_EQ(static_cast<double>(MW::from_words(1.0f, 0x1.8p-53f)),
              1 + 0x1p-52);
    ASSERT_EQ(static_cast<double>(MW::from_words(1.0f, -0x1p-100f)), 1.0);
    ASSERT_EQ(static_cast<float>(MW::from_words(1.0f, 0x1p-30f)), 1.0f);
    // unnormalized words, as pair arithmetic leaves them
    ASSERT_EQ(static_cast<double>(MW::from_words(0x1p-30f, 1.0f)), 1 + 0x1p-30);
    for (int i = 0; i < TestFixture::num_samples; ++i) {
        // the words of random() sum up to a double exactly
        const auto x = this->random();

        const auto value = static_cast<double>(x);

        ASSERT_TRUE(has_words(MW{value}, x.word(0), x.word(1)))
            << std::hexfloat << value;
    }
}


TYPED_TEST(MultiWord, ConvertsSpecialValuesToFloatingPointTypes)
{
    using MW = TypeParam;
    const auto inf = std::numeric_limits<float>::infinity();

    ASSERT_EQ(static_cast<double>(MW{inf}),
              std::numeric_limits<double>::infinity());
    ASSERT_EQ(static_cast<double>(MW{-inf}),
              -std::numeric_limits<double>::infinity());
    ASSERT_TRUE(std::isnan(
        static_cast<double>(MW{std::numeric_limits<float>::quiet_NaN()})));
}


TYPED_TEST(MultiWord, HandledInfinityStaysInfinite)
{
    using MW = handled<TypeParam>;
    const auto inf = std::numeric_limits<float>::infinity();
    const auto one = MW{1.0f};
    const auto zero = MW{0.0f};

    ASSERT_TRUE(has_words(MW{inf} + one, inf, 0.0f));
    ASSERT_TRUE(has_words(one + MW{inf}, inf, 0.0f));
    ASSERT_TRUE(has_words(MW{-inf} + one, -inf, 0.0f));
    ASSERT_TRUE(has_words(one - MW{inf}, -inf, 0.0f));
    ASSERT_TRUE(has_words(MW{inf} + 1.0f, inf, 0.0f));
    ASSERT_TRUE(has_words(MW{inf} * 2.0f, inf, 0.0f));
    ASSERT_TRUE(has_words(MW{inf} / 2.0f, inf, 0.0f));
    ASSERT_TRUE(has_words(one / 0.0f, inf, 0.0f));
    ASSERT_TRUE(has_words(normalize(MW{inf}), inf, 0.0f));
    ASSERT_TRUE(
        has_words(MW{std::numeric_limits<double>::infinity()}, inf, 0.0f));
    ASSERT_TRUE(MW{inf} == MW{inf});
    if constexpr (TestFixture::has_product) {
        ASSERT_TRUE(has_words(MW{inf} * MW{2.0f}, inf, 0.0f));
        ASSERT_TRUE(has_words(MW{inf} / MW{2.0f}, inf, 0.0f));
        ASSERT_TRUE(has_words(one / zero, inf, 0.0f));
        ASSERT_TRUE(has_words(-one / zero, -inf, 0.0f));
        ASSERT_TRUE(has_words(1.0f / zero, inf, 0.0f));
    }
}


TYPED_TEST(MultiWord, HandledNanResultsHaveZeroSecondWord)
{
    using MW = handled<TypeParam>;
    const auto inf = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<MW> results{MW{inf} - MW{inf}, MW{nan} + MW{1.0f},
                            MW{1.0f} + MW{nan}, MW{0.0f} * inf,
                            MW{0.0f} / 0.0f};
    if constexpr (TestFixture::has_product) {
        results.push_back(MW{0.0f} * MW{inf});
        results.push_back(MW{0.0f} / MW{0.0f});
    }

    for (const auto& result : results) {
        ASSERT_TRUE(std::isnan(result.word(0))) << to_string(result);
        ASSERT_EQ(result.word(1), 0.0f) << to_string(result);
    }
}


TYPED_TEST(MultiWord, HandledOverflowGivesInfinity)
{
    using MW = handled<TypeParam>;
    const auto inf = std::numeric_limits<float>::infinity();
    const auto max = MW{std::numeric_limits<float>::max()};

    ASSERT_TRUE(has_words(max + max, inf, 0.0f));
    ASSERT_TRUE(has_words(-max - max, -inf, 0.0f));
    ASSERT_TRUE(has_words(max * 2.0f, inf, 0.0f));
    ASSERT_TRUE(has_words(max / 0.5f, inf, 0.0f));
    ASSERT_TRUE(has_words(MW{1e300}, inf, 0.0f));
    if constexpr (TestFixture::has_product) {
        ASSERT_TRUE(has_words(max * MW{2.0f}, inf, 0.0f));
        ASSERT_TRUE(has_words(max / MW{0.5f}, inf, 0.0f));
    }
}


TYPED_TEST(MultiWord, HandledOverflowWhileNormalizingGivesInfinity)
{
    using MW = handled<TypeParam>;
    if constexpr (TestFixture::pair) {
        GTEST_SKIP() << "pair arithmetic does not normalize";
    } else {
        const auto inf = std::numeric_limits<float>::infinity();
        const auto max = std::numeric_limits<float>::max();
        // a quarter of an ulp of max: max + quarter still rounds to max, but
        // max + 2 * quarter is halfway to the next power of two and rounds to
        // inf
        const auto quarter = 0x1p102f;
        const auto x = MW::from_words(max, quarter);

        ASSERT_TRUE(has_words(x + MW{quarter}, inf, 0.0f));
        ASSERT_TRUE(has_words(x + quarter, inf, 0.0f));
        ASSERT_TRUE(has_words(-x - MW{quarter}, -inf, 0.0f));
    }
}


TYPED_TEST(MultiWord, HandledFailingIntermediateStepFallsBackToLeadingWords)
{
    using MW = handled<TypeParam>;
    const auto inf = std::numeric_limits<float>::infinity();
    // without FMA, the splitting in TwoProd overflows
    const auto large = MW{0x1p120f};

    // the exact division residual is fma(-inf, 0, 1)
    ASSERT_TRUE(has_words(MW{1.0f} / inf, 0.0f, 0.0f));
    ASSERT_TRUE(has_words(large * 0x1p-120f, 1.0f, 0.0f));
    ASSERT_TRUE(has_words(large * 0.0f, 0.0f, 0.0f));
    if constexpr (TestFixture::has_product) {
        ASSERT_TRUE(has_words(MW{1.0f} / MW{inf}, 0.0f, 0.0f));
        ASSERT_TRUE(has_words(large * MW{0x1p-120f}, 1.0f, 0.0f));
        // DWDivDW3 computes 1 / y, which overflows
        ASSERT_TRUE(has_words(
            MW{0x1p-100f} / MW{std::numeric_limits<float>::denorm_min()},
            0x1p49f, 0.0f));
    }
}


TYPED_TEST(MultiWord, IgnoredInfinityPlusOneGivesNan)
{
    using MW = TypeParam;
    const auto inf = std::numeric_limits<float>::infinity();

    const auto result = MW{inf} + MW{1.0f};

    // the leading word in double-word arithmetic, the second word in pair
    // arithmetic
    ASSERT_TRUE(std::isnan(result.word(0)) || std::isnan(result.word(1)))
        << to_string(result);
}


TEST(MultiWordThreeWords, NegatesComparesAndEvaluates)
{
    using MW = gko::multi_word<float, 3>;
    const auto x = MW::from_words(1.0f, 0x1p-30f, 0x1p-50f);
    const auto y = MW::from_words(1.0f, 0x1p-30f, 0x1p-51f);

    ASSERT_EQ((-x).word(2), -0x1p-50f);
    ASSERT_TRUE(y < x);
    ASSERT_TRUE(x == x);
    ASSERT_TRUE(-x < -y);
    ASSERT_TRUE(x > MW{1.0f});
    ASSERT_EQ(mw::eval<double>(x), 1 + 0x1p-30 + 0x1p-50);
    ASSERT_EQ(static_cast<double>(x), 1 + 0x1p-30 + 0x1p-50);
}


}  // namespace
