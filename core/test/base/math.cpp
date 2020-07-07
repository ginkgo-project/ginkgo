/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/base/math.hpp>


#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>


#include <gtest/gtest.h>


namespace {


static_assert(
    std::is_same<double, decltype(real(std::complex<double>()))>::value,
    "real must return a real type");
static_assert(
    std::is_same<double, decltype(imag(std::complex<double>()))>::value,
    "imag must return a real type");


template <typename T>
void test_real_is_finite()
{
    using limits = std::numeric_limits<T>;
    constexpr auto inf = limits::infinity();
    // Use volatile to avoid MSVC report divided by zero.
    volatile const T zero{0};
    ASSERT_TRUE(gko::is_finite(T{0}));
    ASSERT_TRUE(gko::is_finite(-T{0}));
    ASSERT_TRUE(gko::is_finite(T{1}));
    ASSERT_FALSE(gko::is_finite(inf));
    ASSERT_FALSE(gko::is_finite(-inf));
    ASSERT_FALSE(gko::is_finite(limits::quiet_NaN()));
    ASSERT_FALSE(gko::is_finite(limits::signaling_NaN()));
    ASSERT_FALSE(gko::is_finite(inf - inf));    // results in nan
    ASSERT_FALSE(gko::is_finite(inf / inf));    // results in nan
    ASSERT_FALSE(gko::is_finite(inf * T{2}));   // results in inf
    ASSERT_FALSE(gko::is_finite(T{1} / zero));  // results in inf
    ASSERT_FALSE(gko::is_finite(T{0} / zero));  // results in nan
}


template <typename ComplexType>
void test_complex_is_finite()
{
    static_assert(gko::is_complex_s<ComplexType>::value,
                  "Template type must be a complex type.");
    using T = gko::remove_complex<ComplexType>;
    using c_type = ComplexType;
    using limits = std::numeric_limits<T>;
    constexpr auto inf = limits::infinity();
    constexpr auto quiet_nan = limits::quiet_NaN();
    constexpr auto signaling_nan = limits::signaling_NaN();

    ASSERT_TRUE(gko::is_finite(c_type{T{0}, T{0}}));
    ASSERT_TRUE(gko::is_finite(c_type{-T{0}, -T{0}}));
    ASSERT_TRUE(gko::is_finite(c_type{T{1}, T{0}}));
    ASSERT_TRUE(gko::is_finite(c_type{T{0}, T{1}}));
    ASSERT_FALSE(gko::is_finite(c_type{inf, T{0}}));
    ASSERT_FALSE(gko::is_finite(c_type{-inf, T{0}}));
    ASSERT_FALSE(gko::is_finite(c_type{quiet_nan, T{0}}));
    ASSERT_FALSE(gko::is_finite(c_type{signaling_nan, T{0}}));
    ASSERT_FALSE(gko::is_finite(c_type{T{0}, inf}));
    ASSERT_FALSE(gko::is_finite(c_type{T{0}, -inf}));
    ASSERT_FALSE(gko::is_finite(c_type{T{0}, quiet_nan}));
    ASSERT_FALSE(gko::is_finite(c_type{T{0}, signaling_nan}));
}


TEST(IsFinite, Float) { test_real_is_finite<float>(); }


TEST(IsFinite, Double) { test_real_is_finite<double>(); }


TEST(IsFinite, FloatComplex) { test_complex_is_finite<std::complex<float>>(); }


TEST(IsFinite, DoubleComplex)
{
    test_complex_is_finite<std::complex<double>>();
}


TEST(Conjugate, FloatComplex)
{
    std::complex<float> a(1, 1);
    std::complex<float> b(1, -1);

    ASSERT_EQ(conj(a), b);
}


TEST(Conjugate, DoubleComplex)
{
    std::complex<double> a(1, 1);
    std::complex<double> b(1, -1);

    ASSERT_EQ(conj(a), b);
}


}  // namespace
