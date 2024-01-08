// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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


template <typename ValueType, typename IndexType>
class DummyClass {};


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


TEST(RemoveComplexClass, Float)
{
    using origin = DummyClass<float, int>;
    using expect = DummyClass<float, int>;

    bool check = std::is_same<expect, gko::remove_complex<origin>>::value;

    ASSERT_TRUE(check);
}


TEST(RemoveComplexClass, Double)
{
    using origin = DummyClass<double, int>;
    using expect = DummyClass<double, int>;

    bool check = std::is_same<expect, gko::remove_complex<origin>>::value;

    ASSERT_TRUE(check);
}


TEST(RemoveComplexClass, FloatComplex)
{
    using origin = DummyClass<std::complex<float>, int>;
    using expect = DummyClass<float, int>;

    bool check = std::is_same<expect, gko::remove_complex<origin>>::value;

    ASSERT_TRUE(check);
}


TEST(RemoveComplexClass, DoubleComplex)
{
    using origin = DummyClass<std::complex<double>, int>;
    using expect = DummyClass<double, int>;

    bool check = std::is_same<expect, gko::remove_complex<origin>>::value;

    ASSERT_TRUE(check);
}


}  // namespace
