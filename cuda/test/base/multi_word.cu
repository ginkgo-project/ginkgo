// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/multi_word.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "cuda/test/utils.hpp"


namespace {


namespace mw = gko::mw;

constexpr auto eager = mw::normalization::eager;
constexpr auto none = mw::normalization::none;
constexpr auto accurate = mw::algorithm::accurate;
constexpr auto fast = mw::algorithm::fast;
constexpr auto ignore = mw::special::ignore;
constexpr auto handle = mw::special::handle;


// two float words emulating a double
template <mw::normalization Normalization, mw::algorithm Algorithm, bool Fma,
          mw::special Special>
using float_float =
    gko::multi_word<float, 2,
                    mw::policy<Normalization, Algorithm, Fma, Special>>;


// the accurate product and quotient of two multi_words need FMA
template <typename MW>
constexpr bool has_product =
    MW::policy_type::norm == mw::normalization::none ||
    MW::policy_type::alg == mw::algorithm::fast || MW::policy_type::fma;


constexpr int num_operations = 13;


template <typename MW>
__host__ __device__ void apply_operations(const MW& x, const MW& y, float a,
                                          double d, MW* results)
{
    results[0] = x + y;
    results[1] = x - y;
    results[2] = x + a;
    results[3] = x - a;
    results[4] = a - x;
    results[5] = x * a;
    results[6] = x / a;
    results[7] = normalize(x);
    results[8] = abs(x);
    results[9] = MW{d};
    if constexpr (has_product<MW>) {
        results[10] = x * y;
        results[11] = x / y;
        results[12] = a / x;
    }
}


template <typename MW>
__host__ __device__ int compare(const MW& x, const MW& y)
{
    return (x == y) | (x != y) << 1 | (x < y) << 2 | (x <= y) << 3 |
           (x > y) << 4 | (x >= y) << 5;
}


// the words of x[i] are x[2 * i] and x[2 * i + 1]
template <typename MW>
__global__ void run_operations(int n, const float* x, const float* y,
                               const float* a, const double* d, float* results,
                               int* comparisons, double* evaluations)
{
    const auto i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        const auto first = MW::from_words(x[2 * i], x[2 * i + 1]);
        const auto second = MW::from_words(y[2 * i], y[2 * i + 1]);
        MW result[num_operations];
        apply_operations(first, second, a[i], d[i], result);
        for (int k = 0; k < num_operations; ++k) {
            for (int w = 0; w < 2; ++w) {
                results[(i * num_operations + k) * 2 + w] = result[k].word(w);
            }
        }
        comparisons[i] = compare(first, second);
        evaluations[i] = mw::eval<double>(first);
    }
}


template <typename MW>
class MultiWord : public CudaTestFixture {
protected:
    static constexpr int p = std::numeric_limits<float>::digits;

    MultiWord() : engine(42) {}

    // a random normalized number with any exponent, possibly with a
    // subnormal second word
    MW random()
    {
        std::uniform_real_distribution<float> significand(1, 2);
        // stay a bit away from overflow, but reach subnormal second words
        std::uniform_int_distribution<int> exponent(-126 + p, 123);
        std::uniform_int_distribution<int> gap(0, 2 * p);
        auto leading = std::ldexp(significand(engine), exponent(engine));
        leading = negative(engine) ? -leading : leading;
        int exp{};
        std::frexp(leading, &exp);
        auto second =
            std::ldexp(significand(engine), exp - p - 2 - gap(engine));
        second = negative(engine) ? -second : second;
        const auto n = mw::detail::fast_two_sum(leading, second);
        return MW::from_words(n[0], n[1]);
    }

    // a random second operand for x, such that x + y and x - y often cancel
    MW random_partner(const MW& x)
    {
        std::uniform_int_distribution<int> kind(0, 3);
        const auto y = random();
        switch (kind(engine)) {
        case 0:
            return MW::from_words(-x.word(0), y.word(1));
        case 1:
            return MW::from_words(x.word(0), y.word(1));
        case 2:
            return MW::from_words(std::nextafter(x.word(0), 0.0f), 0.0f);
        default:
            return y;
        }
    }

    // a random double from far below to far above the range of float
    double random_double()
    {
        std::uniform_real_distribution<double> significand(1, 2);
        std::uniform_int_distribution<int> exponent(-160, 140);
        const auto value = std::ldexp(significand(engine), exponent(engine));
        return negative(engine) ? -value : value;
    }

    static std::vector<MW> special_values()
    {
        const auto inf = std::numeric_limits<float>::infinity();
        const auto max = std::numeric_limits<float>::max();
        const auto min = std::numeric_limits<float>::min();
        const auto denorm = std::numeric_limits<float>::denorm_min();
        return {MW{0.0f}, MW{-0.0f}, MW{1.0f}, MW{-1.0f}, MW{inf}, MW{-inf},
                MW{std::numeric_limits<float>::quiet_NaN()}, MW{max}, MW{-max},
                // a quarter of an ulp of max
                MW::from_words(max, 0x1p102f), MW{min}, MW{denorm},
                MW::from_words(min, denorm),
                MW::from_words(1.0f, std::ldexp(1.0f, -p - 1))};
    }

    // NaN results only need to agree on being NaN, the payload may differ
    template <typename T>
    static bool same(T a, T b)
    {
        return (std::isnan(a) && std::isnan(b)) ||
               std::memcmp(&a, &b, sizeof(T)) == 0;
    }

    std::default_random_engine engine;
    std::bernoulli_distribution negative;
};

using FloatFloatTypes =
    ::testing::Types<float_float<eager, accurate, true, ignore>,
                     float_float<eager, accurate, false, ignore>,
                     float_float<eager, fast, true, ignore>,
                     float_float<eager, fast, false, ignore>,
                     float_float<none, accurate, true, ignore>,
                     float_float<none, accurate, false, ignore>,
                     float_float<eager, accurate, true, handle>,
                     float_float<eager, accurate, false, handle>,
                     float_float<eager, fast, true, handle>,
                     float_float<eager, fast, false, handle>,
                     float_float<none, accurate, true, handle>,
                     float_float<none, accurate, false, handle>>;

TYPED_TEST_SUITE(MultiWord, FloatFloatTypes, TypenameNameGenerator);


TYPED_TEST(MultiWord, DeviceResultsAreBitIdenticalToHost)
{
    using MW = TypeParam;
    std::vector<MW> xs;
    std::vector<MW> ys;
    std::vector<float> as;
    std::vector<double> ds;
    for (int i = 0; i < 100000; ++i) {
        const auto x = this->random();
        xs.push_back(x);
        ys.push_back(this->random_partner(x));
        as.push_back(i % 2 == 0 ? -x.word(0) : this->random().word(0));
        ds.push_back(this->random_double());
    }
    const auto specials = TestFixture::special_values();
    for (const auto& x : specials) {
        for (const auto& y : specials) {
            xs.push_back(x);
            ys.push_back(y);
            as.push_back(y.word(0));
            ds.push_back(static_cast<double>(y.word(0)));
        }
    }
    const auto n = static_cast<int>(xs.size());
    gko::array<float> x_words(this->ref, 2 * n);
    gko::array<float> y_words(this->ref, 2 * n);
    for (int i = 0; i < n; ++i) {
        for (int w = 0; w < 2; ++w) {
            x_words.get_data()[2 * i + w] = xs[i].word(w);
            y_words.get_data()[2 * i + w] = ys[i].word(w);
        }
    }
    gko::array<float> d_x(this->exec, x_words);
    gko::array<float> d_y(this->exec, y_words);
    gko::array<float> d_a(this->exec,
                          gko::array<float>(this->ref, as.begin(), as.end()));
    gko::array<double> d_d(this->exec,
                           gko::array<double>(this->ref, ds.begin(), ds.end()));
    gko::array<float> d_results(this->exec, 2 * num_operations * n);
    gko::array<int> d_comparisons(this->exec, n);
    gko::array<double> d_evaluations(this->exec, n);
    constexpr int block_size = 256;

    run_operations<MW><<<(n + block_size - 1) / block_size, block_size, 0,
                         this->exec->get_stream()>>>(
        n, d_x.get_const_data(), d_y.get_const_data(), d_a.get_const_data(),
        d_d.get_const_data(), d_results.get_data(), d_comparisons.get_data(),
        d_evaluations.get_data());

    gko::array<float> results(this->ref, d_results);
    gko::array<int> comparisons(this->ref, d_comparisons);
    gko::array<double> evaluations(this->ref, d_evaluations);
    for (int i = 0; i < n; ++i) {
        MW expected[num_operations];
        apply_operations(xs[i], ys[i], as[i], ds[i], expected);
        for (int k = 0; k < num_operations; ++k) {
            for (int w = 0; w < 2; ++w) {
                ASSERT_TRUE(TestFixture::same(
                    results.get_const_data()[(i * num_operations + k) * 2 + w],
                    expected[k].word(w)))
                    << "input " << i << ", operation " << k << ", word " << w;
            }
        }
        ASSERT_EQ(comparisons.get_const_data()[i], compare(xs[i], ys[i]))
            << "input " << i;
        ASSERT_TRUE(TestFixture::same(evaluations.get_const_data()[i],
                                      mw::eval<double>(xs[i])))
            << "input " << i;
    }
}


}  // namespace
