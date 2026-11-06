// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_EMULATION_FLOAT_FLOAT_HELPER_HPP_
#define GKO_CORE_TEST_EMULATION_FLOAT_FLOAT_HELPER_HPP_


#include <cmath>
#include <random>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "core/emulation/double_word.hpp"


/**
 * The multi_word types of two float words that the tests of multi_word and
 * double_word run with, and helpers to create and check their values.
 */
namespace float_float_helper {


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

using types = ::testing::Types<accurate_fma, accurate_no_fma, fast_fma,
                               fast_no_fma, pair_fma, pair_no_fma>;


inline std::string to_string(float value)
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


/**
 * The policy of MW and random values of MW.
 */
template <typename MW>
class fixture : public ::testing::Test {
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

    fixture() : engine(42) {}

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


}  // namespace float_float_helper


#endif  // GKO_CORE_TEST_EMULATION_FLOAT_FLOAT_HELPER_HPP_
