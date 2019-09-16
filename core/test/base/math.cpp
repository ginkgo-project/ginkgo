/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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


#include <gtest/gtest.h>


namespace {


template <typename T>
void test_real_isfinite()
{
    using limits = std::numeric_limits<T>;
    constexpr auto inf = limits::infinity();

    ASSERT_TRUE(gko::isfinite(T{0}));
    ASSERT_TRUE(gko::isfinite(-T{0}));
    ASSERT_TRUE(gko::isfinite(T{1}));
    ASSERT_FALSE(gko::isfinite(inf));
    ASSERT_FALSE(gko::isfinite(-inf));
    ASSERT_FALSE(gko::isfinite(limits::quiet_NaN()));
    ASSERT_FALSE(gko::isfinite(limits::signaling_NaN()));
    ASSERT_FALSE(gko::isfinite(inf - inf));    // results in nan
    ASSERT_FALSE(gko::isfinite(inf / inf));    // results in nan
    ASSERT_FALSE(gko::isfinite(inf * T{2}));   // results in inf
    ASSERT_FALSE(gko::isfinite(T{1} / T{0}));  // results in inf
    ASSERT_FALSE(gko::isfinite(T{0} / T{0}));  // results in nan
}


template <typename ComplexType>
void test_complex_isfinite()
{
    static_assert(gko::is_complex_s<ComplexType>::value,
                  "Template type must be a complex type.");
    using T = gko::remove_complex<ComplexType>;
    using c_type = ComplexType;
    using limits = std::numeric_limits<T>;
    constexpr auto inf = limits::infinity();
    constexpr auto quiet_nan = limits::quiet_NaN();
    constexpr auto signaling_nan = limits::signaling_NaN();

    ASSERT_TRUE(gko::isfinite(c_type{T{0}, T{0}}));
    ASSERT_TRUE(gko::isfinite(c_type{-T{0}, -T{0}}));
    ASSERT_TRUE(gko::isfinite(c_type{T{1}, T{0}}));
    ASSERT_TRUE(gko::isfinite(c_type{T{0}, T{1}}));
    ASSERT_FALSE(gko::isfinite(c_type{inf, T{0}}));
    ASSERT_FALSE(gko::isfinite(c_type{-inf, T{0}}));
    ASSERT_FALSE(gko::isfinite(c_type{quiet_nan, T{0}}));
    ASSERT_FALSE(gko::isfinite(c_type{signaling_nan, T{0}}));
    ASSERT_FALSE(gko::isfinite(c_type{T{0}, inf}));
    ASSERT_FALSE(gko::isfinite(c_type{T{0}, -inf}));
    ASSERT_FALSE(gko::isfinite(c_type{T{0}, quiet_nan}));
    ASSERT_FALSE(gko::isfinite(c_type{T{0}, signaling_nan}));
}


TEST(IsFinite, Float) { test_real_isfinite<float>(); }


TEST(IsFinite, Double) { test_real_isfinite<double>(); }


TEST(IsFinite, FloatComplex) { test_complex_isfinite<std::complex<float>>(); }


TEST(IsFinite, DoubleComplex) { test_complex_isfinite<std::complex<double>>(); }


}  // namespace
