// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_TEST_EMULATION_FLOAT_FLOAT_HELPER_CUH_
#define GKO_CUDA_TEST_EMULATION_FLOAT_FLOAT_HELPER_CUH_


#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "core/emulation/double_word.hpp"
#include "cuda/test/utils.hpp"


/**
 * Compares operations of multi_words of two float words on the device with
 * the host, for the tests of multi_word and double_word.
 */
namespace float_float_helper {


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

using types = ::testing::Types<float_float<eager, accurate, true, ignore>,
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


// the accurate product and quotient of two multi_words need FMA
template <typename MW>
constexpr bool has_product =
    MW::policy_type::norm == mw::normalization::none ||
    MW::policy_type::alg == mw::algorithm::fast || MW::policy_type::fma;


// the words of x as two results; double holds every float exactly
template <typename MW>
__host__ __device__ void store(const MW& x, double* results)
{
    results[0] = x.word(0);
    results[1] = x.word(1);
}


/**
 * Writes the Operations::num_results results of
 * Operations::apply(x, y, a, d, results) for every input. The words of x[i]
 * are x[2 * i] and x[2 * i + 1]. Results that apply does not write are zero.
 */
template <typename MW, typename Operations>
__global__ void run_operations(int n, const float* x, const float* y,
                               const float* a, const double* d, double* results)
{
    const auto i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        auto own_results = results + i * Operations::num_results;
        for (int k = 0; k < Operations::num_results; ++k) {
            own_results[k] = 0;
        }
        Operations::apply(MW::from_words(x[2 * i], x[2 * i + 1]),
                          MW::from_words(y[2 * i], y[2 * i + 1]), a[i], d[i],
                          own_results);
    }
}


template <typename MW>
class fixture : public CudaTestFixture {
protected:
    static constexpr int p = std::numeric_limits<float>::digits;

    fixture() : engine(42) {}

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
    static bool same(double a, double b)
    {
        return (std::isnan(a) && std::isnan(b)) ||
               std::memcmp(&a, &b, sizeof(double)) == 0;
    }

    /**
     * Checks that the device gives the same results as the host for random
     * inputs, which often cancel, and for all pairs of special values.
     */
    template <typename Operations>
    void check_device_against_host()
    {
        constexpr int num_results = Operations::num_results;
        std::vector<MW> xs;
        std::vector<MW> ys;
        std::vector<float> as;
        std::vector<double> ds;
        for (int i = 0; i < 100000; ++i) {
            const auto x = random();
            xs.push_back(x);
            ys.push_back(random_partner(x));
            as.push_back(i % 2 == 0 ? -x.word(0) : random().word(0));
            ds.push_back(random_double());
        }
        const auto specials = special_values();
        for (const auto& x : specials) {
            for (const auto& y : specials) {
                xs.push_back(x);
                ys.push_back(y);
                as.push_back(y.word(0));
                ds.push_back(static_cast<double>(y.word(0)));
            }
        }
        const auto n = static_cast<int>(xs.size());
        gko::array<float> x_words(ref, 2 * n);
        gko::array<float> y_words(ref, 2 * n);
        for (int i = 0; i < n; ++i) {
            for (int w = 0; w < 2; ++w) {
                x_words.get_data()[2 * i + w] = xs[i].word(w);
                y_words.get_data()[2 * i + w] = ys[i].word(w);
            }
        }
        gko::array<float> d_x(exec, x_words);
        gko::array<float> d_y(exec, y_words);
        gko::array<float> d_a(exec,
                              gko::array<float>(ref, as.begin(), as.end()));
        gko::array<double> d_d(exec,
                               gko::array<double>(ref, ds.begin(), ds.end()));
        gko::array<double> d_results(exec, num_results * n);
        constexpr int block_size = 256;

        run_operations<MW, Operations><<<(n + block_size - 1) / block_size,
                                         block_size, 0, exec->get_stream()>>>(
            n, d_x.get_const_data(), d_y.get_const_data(), d_a.get_const_data(),
            d_d.get_const_data(), d_results.get_data());

        gko::array<double> results(ref, d_results);
        for (int i = 0; i < n; ++i) {
            std::vector<double> expected(num_results, 0.0);
            Operations::apply(xs[i], ys[i], as[i], ds[i], expected.data());
            for (int k = 0; k < num_results; ++k) {
                ASSERT_TRUE(same(results.get_const_data()[i * num_results + k],
                                 expected[k]))
                    << "input " << i << ", result " << k;
            }
        }
    }

    std::default_random_engine engine;
    std::bernoulli_distribution negative;
};


}  // namespace float_float_helper


#endif  // GKO_CUDA_TEST_EMULATION_FLOAT_FLOAT_HELPER_CUH_
