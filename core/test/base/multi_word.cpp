// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/multi_word.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "core/test/utils.hpp"


namespace {


namespace mw = gko::mw;

constexpr auto eager = mw::normalization::eager;
constexpr auto none = mw::normalization::none;
constexpr auto accurate = mw::algorithm::accurate;
constexpr auto fast = mw::algorithm::fast;


// two float words emulating a double
template <mw::normalization Normalization, mw::algorithm Algorithm, bool Fma,
          mw::special Special = mw::special::ignore>
using float_float =
    gko::multi_word<float, 2,
                    mw::policy<Normalization, Algorithm, Fma, Special>>;

using accurate_fma = float_float<eager, accurate, true>;
using accurate_no_fma = float_float<eager, accurate, false>;
using fast_fma = float_float<eager, fast, true>;
using fast_no_fma = float_float<eager, fast, false>;
using pair_fma = float_float<none, accurate, true>;
using pair_no_fma = float_float<none, accurate, false>;


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
static_assert(std::is_convertible_v<float, accurate_fma>);
static_assert(std::is_convertible_v<int, accurate_fma>);
// a double in general does not fit into two floats, so only explicitly
static_assert(std::is_constructible_v<accurate_fma, double>);
static_assert(!std::is_convertible_v<double, accurate_fma>);
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


std::string to_string(float value)
{
    std::ostringstream stream;
    stream << std::hexfloat << value;
    return stream.str();
}


template <typename T, int N, typename Policy>
std::string to_string(const gko::multi_word<T, N, Policy>& value)
{
    std::string result = "(";
    for (int i = 0; i < N; ++i) {
        result += (i > 0 ? ", " : "") + to_string(value.word(i));
    }
    return result + ")";
}


template <typename MW>
::testing::AssertionResult has_words(const MW& value, float word0, float word1)
{
    if (value.word(0) == word0 && value.word(1) == word1) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
           << to_string(value)
           << " != " << to_string(MW::from_words(word0, word1));
}


// the leading word is the sum of the words, rounded to nearest
template <typename MW>
::testing::AssertionResult is_normalized(const MW& value)
{
    if (value.word(0) + value.word(1) == value.word(0)) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
           << to_string(value) << " is not normalized";
}


template <typename MW>
class MultiWord : public ::testing::Test {
public:
    using policy = typename MW::policy_type;
    static constexpr bool pair = policy::norm == mw::normalization::none;
    static constexpr bool fast = policy::alg == mw::algorithm::fast;
    static constexpr bool fma = policy::fma;
    // the accurate product and quotient of two multi_words need FMA
    static constexpr bool has_product = pair || fast || fma;
    static constexpr int num_samples = 20000;
    // the unit roundoff of float
    static constexpr double u = 0x1p-24;

    MultiWord() : engine(42) {}

    /*
     * The relative error bounds in units of u^2, see multi_word. Fast and
     * pair addition only have a bound for operands with equal signs. Pair
     * arithmetic gives the value of the fast algorithms, except for the
     * division, which has no bound of its own, so the test uses the one of
     * DWDivDW2.
     */

    static double add_bound() { return 3 / (1 - 4 * u); }

    static double add_word_bound() { return 2; }

    static double mul_bound() { return pair || !fma ? 7 : 5; }

    static double mul_word_bound()
    {
        if (pair || (fast && !fma)) {
            return 3;
        }
        return fma ? 2 : 1.5 + 4 * u;
    }

    static double div_bound() { return pair || fast ? 15 + 56 * u : 9.8; }

    static double div_word_bound() { return 3; }

    /**
     * Compares with the double result of the operation on the exact double
     * values of the operands. The double reference and the evaluation of the
     * result in double each round once, which adds up to 2^-52 = u^2 / 16.
     */
    template <typename X, typename Y>
    static ::testing::AssertionResult is_within_bound(
        const X& x, const char* operation, const Y& y, const MW& result,
        double expected, double bound)
    {
        const auto computed = mw::eval<double>(result);
        const auto error = std::abs(computed - expected) / (u * u);
        const auto magnitude = std::abs(expected);
        if (expected == 0 ? computed == 0
                          : error <= (bound + 0.125) * magnitude) {
            return ::testing::AssertionSuccess();
        }
        return ::testing::AssertionFailure()
               << to_string(x) << ' ' << operation << ' ' << to_string(y)
               << " = " << to_string(result) << " has a relative error of "
               << error / magnitude << "u^2 > " << bound << "u^2";
    }

    // a random double with an exponent in [-30, 30]
    double random_double()
    {
        std::uniform_real_distribution<double> significand(1, 2);
        std::uniform_int_distribution<int> exponent(-30, 30);
        std::uniform_int_distribution<int> kind(0, 9);
        std::uniform_int_distribution<int> gap(1, 27);
        auto value = std::ldexp(significand(engine), exponent(engine));
        const auto k = kind(engine);
        if (k == 0) {
            // a zero second word
            value = static_cast<float>(value);
        } else if (k == 1) {
            // a second word with a single bit far below the first word
            int exp{};
            std::frexp(value, &exp);
            value = static_cast<float>(value) +
                    std::ldexp(random_sign(), exp - 25 - gap(engine));
        }
        return random_sign() * value;
    }

    // a random value, whose words sum up to a double exactly
    MW random() { return MW{random_double()}; }

    float random_word() { return static_cast<float>(random_double()); }

    // a random second operand for x, such that x + y often cancels
    MW random_partner(const MW& x)
    {
        std::uniform_int_distribution<int> kind(0, 2);
        std::uniform_real_distribution<double> perturbation(-1, 1);
        std::uniform_int_distribution<int> equal_bits(1, 50);
        switch (kind(engine)) {
        case 0:
            return -x;
        case 1:
            // the leading words cancel exactly or partially
            return MW{
                -mw::eval<double>(x) *
                (1 + std::ldexp(perturbation(engine), -equal_bits(engine)))};
        default:
            return random();
        }
    }

    // a random single word for x, such that x + a often cancels
    float random_word_partner(const MW& x)
    {
        std::uniform_int_distribution<int> kind(0, 2);
        switch (kind(engine)) {
        case 0:
            return -x.word(0);
        case 1:
            return -std::nextafter(x.word(0), 0.0f);
        default:
            return random_word();
        }
    }

    MW random_with_sign_of(const MW& x)
    {
        const auto y = random_partner(x);
        return (y.word(0) < 0) == (x.word(0) < 0) ? y : -y;
    }

    double random_sign() { return negative(engine) ? -1.0 : 1.0; }

    std::default_random_engine engine;
    std::bernoulli_distribution negative;
};

using FloatFloatTypes =
    ::testing::Types<accurate_fma, accurate_no_fma, fast_fma, fast_no_fma,
                     pair_fma, pair_no_fma>;

TYPED_TEST_SUITE(MultiWord, FloatFloatTypes, TypenameNameGenerator);


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


TYPED_TEST(MultiWord, ZeroAndOneAreTheIdentities)
{
    using MW = TypeParam;
    const auto zero = gko::zero<MW>();
    const auto one = gko::one<MW>();

    ASSERT_TRUE(has_words(zero, 0.0f, 0.0f));
    ASSERT_TRUE(has_words(one, 1.0f, 0.0f));
    ASSERT_TRUE(gko::is_zero(zero));
    ASSERT_TRUE(gko::is_nonzero(one));
    for (int i = 0; i < 1000; ++i) {
        const auto x = this->random();
        ASSERT_TRUE(has_words(gko::zero(x), 0.0f, 0.0f));
        ASSERT_TRUE(has_words(gko::one(x), 1.0f, 0.0f));
        ASSERT_TRUE(has_words(x + zero, x.word(0), x.word(1)));
        ASSERT_TRUE(has_words(x - zero, x.word(0), x.word(1)));
        if constexpr (TestFixture::has_product) {
            ASSERT_TRUE(has_words(x * one, x.word(0), x.word(1)));
            ASSERT_TRUE(has_words(x / one, x.word(0), x.word(1)));
        }
    }
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


TYPED_TEST(MultiWord, NormalizeMovesExcessIntoLeadingWord)
{
    using MW = TypeParam;

    ASSERT_TRUE(has_words(normalize(MW::from_words(1.0f, 1.0f)), 2.0f, 0.0f));
    // a second word larger than the first, but with the same exponent, which
    // Fast2Sum still handles
    ASSERT_TRUE(has_words(normalize(MW::from_words(0x1p-24f, 0x1.4p-24f)),
                          0x1.2p-23f, 0.0f));
    ASSERT_TRUE(
        has_words(normalize(MW::from_words(1.0f, 0x1p-30f)), 1.0f, 0x1p-30f));
    if constexpr (TestFixture::pair) {
        // a second word with a larger exponent, as pair arithmetic can leave
        // it, needs TwoSum: Fast2Sum would lose the first word
        ASSERT_TRUE(has_words(normalize(MW::from_words(0x1p-30f, 1.0f)), 1.0f,
                              0x1p-30f));
    }
}


TYPED_TEST(MultiWord, OperatesExactlyOnSmallIntegers)
{
    using MW = TypeParam;
    const auto x = MW{6};
    const auto y = MW{3};

    ASSERT_TRUE(has_words(x + y, 9.0f, 0.0f));
    ASSERT_TRUE(has_words(x - y, 3.0f, 0.0f));
    ASSERT_TRUE(has_words(x + 3.0f, 9.0f, 0.0f));
    ASSERT_TRUE(has_words(3.0f - x, -3.0f, 0.0f));
    ASSERT_TRUE(has_words(x * 3.0f, 18.0f, 0.0f));
    ASSERT_TRUE(has_words(x / 3.0f, 2.0f, 0.0f));
    if constexpr (TestFixture::has_product) {
        ASSERT_TRUE(has_words(x * y, 18.0f, 0.0f));
        // DWDivDW3 multiplies by the reciprocal, which is only exact for
        // powers of two
        ASSERT_TRUE(has_words(x / MW{2}, 3.0f, 0.0f));
        ASSERT_TRUE(has_words(3.0f / MW{4}, 0.75f, 0.0f));
    }
}


TYPED_TEST(MultiWord, KeepsRoundingErrorInSecondWord)
{
    using MW = TypeParam;
    const auto small = 0x1p-30f;
    // (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24, the last bit is a tie that float
    // rounds to even
    const auto factor = 1 + 0x1p-12f;

    ASSERT_TRUE(has_words(MW{1.0f} + MW{small}, 1.0f, small));
    ASSERT_TRUE(has_words(MW{small} + MW{1.0f}, 1.0f, small));
    ASSERT_TRUE(has_words(MW{1.0f} - MW{small}, 1.0f, -small));
    ASSERT_TRUE(has_words(MW{1.0f} + small, 1.0f, small));
    ASSERT_TRUE(has_words(small + MW{1.0f}, 1.0f, small));
    ASSERT_TRUE(has_words(1.0f - MW{small}, 1.0f, -small));
    ASSERT_TRUE(has_words(MW{factor} * factor, 1 + 0x1p-11f, 0x1p-24f));
    if constexpr (TestFixture::has_product) {
        ASSERT_TRUE(has_words(MW{factor} * MW{factor}, 1 + 0x1p-11f, 0x1p-24f));
    }
}


TYPED_TEST(MultiWord, AddWhenLeadingWordsCancel)
{
    using MW = TypeParam;
    const auto x1 = 0x1p-25f;
    const auto y1 = -0x1p-60f;
    const auto x = MW::from_words(1.0f, x1);
    const auto y = MW::from_words(-1.0f, y1);

    const auto sum = x + y;

    // the exact sum is x1 + y1
    if constexpr (TestFixture::pair) {
        // rounds x1 + y1 to the second word, and does not normalize
        ASSERT_TRUE(has_words(sum, 0.0f, x1));
    } else if constexpr (TestFixture::fast) {
        // rounds x1 + y1 to a single word: the relative error is about
        // 2^-35, far above 3u^2 = 3 * 2^-48
        ASSERT_TRUE(has_words(sum, x1, 0.0f));
    } else {
        ASSERT_TRUE(has_words(sum, x1, y1));
    }
}


TYPED_TEST(MultiWord, SubtractingAValueFromItselfGivesZero)
{
    for (int i = 0; i < 100; ++i) {
        const auto x = this->random();

        ASSERT_TRUE(has_words(x - x, 0.0f, 0.0f)) << to_string(x);
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


TYPED_TEST(MultiWord, AddIsWithinErrorBound)
{
    const auto bound = TestFixture::add_bound();
    for (int i = 0; i < TestFixture::num_samples; ++i) {
        const auto x = this->random();
        // only the accurate algorithm has a bound under cancellation
        const auto y = TestFixture::pair || TestFixture::fast
                           ? this->random_with_sign_of(x)
                           : this->random_partner(x);
        const auto expected = mw::eval<double>(x) + mw::eval<double>(y);

        const auto sum = x + y;

        ASSERT_TRUE(
            TestFixture::is_within_bound(x, "+", y, sum, expected, bound));
        ASSERT_TRUE(TestFixture::is_within_bound(x, "-", -y, x - (-y), expected,
                                                 bound));
        if constexpr (!TestFixture::pair) {
            ASSERT_TRUE(is_normalized(sum));
        }
    }
}


TYPED_TEST(MultiWord, AddWordIsWithinErrorBound)
{
    const auto bound = TestFixture::add_word_bound();
    for (int i = 0; i < TestFixture::num_samples; ++i) {
        const auto x = this->random();
        const auto a = this->random_word_partner(x);
        const auto xd = mw::eval<double>(x);
        const auto ad = static_cast<double>(a);

        const auto sum = x + a;

        ASSERT_TRUE(
            TestFixture::is_within_bound(x, "+", a, sum, xd + ad, bound));
        ASSERT_TRUE(
            TestFixture::is_within_bound(a, "+", x, a + x, xd + ad, bound));
        ASSERT_TRUE(
            TestFixture::is_within_bound(x, "-", -a, x - (-a), xd + ad, bound));
        ASSERT_TRUE(
            TestFixture::is_within_bound(-a, "-", x, -a - x, -ad - xd, bound));
        if constexpr (!TestFixture::pair) {
            ASSERT_TRUE(is_normalized(sum));
        }
    }
}


TYPED_TEST(MultiWord, MulIsWithinErrorBound)
{
    if constexpr (TestFixture::has_product) {
        const auto bound = TestFixture::mul_bound();
        for (int i = 0; i < TestFixture::num_samples; ++i) {
            const auto x = this->random();
            const auto y = this->random();
            const auto expected = mw::eval<double>(x) * mw::eval<double>(y);

            const auto product = x * y;

            ASSERT_TRUE(TestFixture::is_within_bound(x, "*", y, product,
                                                     expected, bound));
            if constexpr (!TestFixture::pair) {
                ASSERT_TRUE(is_normalized(product));
            }
        }
    } else {
        GTEST_SKIP() << "the accurate product of two multi_words needs FMA";
    }
}


TYPED_TEST(MultiWord, MulWordIsWithinErrorBound)
{
    const auto bound = TestFixture::mul_word_bound();
    for (int i = 0; i < TestFixture::num_samples; ++i) {
        const auto x = this->random();
        const auto a = this->random_word();
        const auto expected = mw::eval<double>(x) * a;

        const auto product = x * a;

        ASSERT_TRUE(
            TestFixture::is_within_bound(x, "*", a, product, expected, bound));
        ASSERT_TRUE(
            TestFixture::is_within_bound(a, "*", x, a * x, expected, bound));
        if constexpr (!TestFixture::pair) {
            ASSERT_TRUE(is_normalized(product));
        }
    }
}


TYPED_TEST(MultiWord, DivIsWithinErrorBound)
{
    if constexpr (TestFixture::has_product) {
        const auto bound = TestFixture::div_bound();
        for (int i = 0; i < TestFixture::num_samples; ++i) {
            const auto x = this->random();
            const auto y = this->random();
            const auto a = this->random_word();
            const auto xd = mw::eval<double>(x);
            const auto yd = mw::eval<double>(y);

            const auto quotient = x / y;

            ASSERT_TRUE(TestFixture::is_within_bound(x, "/", y, quotient,
                                                     xd / yd, bound));
            ASSERT_TRUE(
                TestFixture::is_within_bound(a, "/", y, a / y, a / yd, bound));
            if constexpr (!TestFixture::pair) {
                ASSERT_TRUE(is_normalized(quotient));
            }
        }
    } else {
        GTEST_SKIP() << "the accurate quotient of two multi_words needs FMA";
    }
}


TYPED_TEST(MultiWord, DivWordIsWithinErrorBound)
{
    const auto bound = TestFixture::div_word_bound();
    for (int i = 0; i < TestFixture::num_samples; ++i) {
        const auto x = this->random();
        const auto a = this->random_word();
        const auto expected = mw::eval<double>(x) / a;

        const auto quotient = x / a;

        ASSERT_TRUE(
            TestFixture::is_within_bound(x, "/", a, quotient, expected, bound));
        if constexpr (!TestFixture::pair) {
            ASSERT_TRUE(is_normalized(quotient));
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
}


}  // namespace
