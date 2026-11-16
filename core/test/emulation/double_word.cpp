// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/emulation/double_word.hpp"

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "core/test/emulation/float_float_helper.hpp"
#include "core/test/utils.hpp"


namespace {


using namespace float_float_helper;


template <typename MW>
class DoubleWord : public fixture<MW> {
public:
    using base = fixture<MW>;
    using base::fast;
    using base::fma;
    using base::pair;
    using base::u;

    /*
     * The relative error bounds in units of u^2, see
     * core/emulation/double_word.hpp. Fast and pair addition only have a bound
     * for operands with equal signs. Pair arithmetic gives the value of the
     * fast algorithms, except for the division, which has no bound of its own,
     * so the test uses the one of DWDivDW2.
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
};

TYPED_TEST_SUITE(DoubleWord, types, TypenameNameGenerator);


TYPED_TEST(DoubleWord, ZeroAndOneAreTheIdentities)
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


TYPED_TEST(DoubleWord, NormalizeMovesExcessIntoLeadingWord)
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


TYPED_TEST(DoubleWord, OperatesExactlyOnSmallIntegers)
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


TYPED_TEST(DoubleWord, KeepsRoundingErrorInSecondWord)
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


TYPED_TEST(DoubleWord, AddWhenLeadingWordsCancel)
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


TYPED_TEST(DoubleWord, SubtractingAValueFromItselfGivesZero)
{
    for (int i = 0; i < 100; ++i) {
        const auto x = this->random();

        ASSERT_TRUE(has_words(x - x, 0.0f, 0.0f)) << to_string(x);
    }
}


TYPED_TEST(DoubleWord, AddIsWithinErrorBound)
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


TYPED_TEST(DoubleWord, AddWordIsWithinErrorBound)
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


TYPED_TEST(DoubleWord, MulIsWithinErrorBound)
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


TYPED_TEST(DoubleWord, MulWordIsWithinErrorBound)
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


TYPED_TEST(DoubleWord, DivIsWithinErrorBound)
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


TYPED_TEST(DoubleWord, DivWordIsWithinErrorBound)
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


TYPED_TEST(DoubleWord, SqrtIsWithinErrorBound)
{
    // SQRTDWtoDW for all policies, Lefevre et al. 2023
    constexpr double bound = 25.0 / 8;
    for (int i = 0; i < TestFixture::num_samples; ++i) {
        const auto x = abs(this->random());
        const auto expected = std::sqrt(mw::eval<double>(x));

        const auto root = sqrt(x);

        ASSERT_TRUE(
            TestFixture::is_within_bound(x, "sqrt", x, root, expected, bound));
        if constexpr (!TestFixture::pair) {
            ASSERT_TRUE(is_normalized(root));
        }
    }
}


TYPED_TEST(DoubleWord, SqrtIsExactForSquares)
{
    using MW = TypeParam;

    ASSERT_TRUE(has_words(sqrt(MW{4.0f}), 2.0f, 0.0f));
    ASSERT_TRUE(has_words(sqrt(MW{0x1p-100f}), 0x1p-50f, 0.0f));
    // (1 + 2^-20)^2 needs both words
    ASSERT_TRUE(
        has_words(sqrt(MW{(1 + 0x1p-20) * (1 + 0x1p-20)}), 1 + 0x1p-20f, 0.0f));
}


TYPED_TEST(DoubleWord, SqrtOfSpecialValuesIsTheSqrtOfTheLeadingWord)
{
    using MW = TypeParam;
    const auto inf = std::numeric_limits<float>::infinity();

    ASSERT_TRUE(has_words(sqrt(MW{}), 0.0f, 0.0f));
    ASSERT_TRUE(has_words(sqrt(MW{inf}), inf, 0.0f));
    ASSERT_TRUE(std::isnan(sqrt(MW{-1.0f}).word(0)));
    ASSERT_TRUE(std::isnan(sqrt(MW{-inf}).word(0)));
    ASSERT_TRUE(
        std::isnan(sqrt(MW{std::numeric_limits<float>::quiet_NaN()}).word(0)));
}


TYPED_TEST(DoubleWord, SqrtOfUnnormalizedPairNormalizesFirst)
{
    using MW = TypeParam;

    if constexpr (TestFixture::pair) {
        // a zero leading word, but a positive value
        ASSERT_TRUE(
            has_words(sqrt(MW::from_words(0.0f, 0x1p-30f)), 0x1p-15f, 0.0f));
        // words that cancel give 0, not the NaN of their negative leading word
        ASSERT_TRUE(has_words(sqrt(MW::from_words(-1.0f, 1.0f)), 0.0f, 0.0f));
        // a negative value behind a zero leading word
        ASSERT_TRUE(std::isnan(sqrt(MW::from_words(0.0f, -1.0f)).word(0)));
    } else {
        GTEST_SKIP() << "only pair arithmetic leaves unnormalized words";
    }
}


}  // namespace
