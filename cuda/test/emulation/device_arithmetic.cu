// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/emulation/device_arithmetic.hpp"

#include <complex>
#include <limits>
#include <type_traits>

#include <gtest/gtest.h>

#include <ginkgo/core/base/half.hpp>

#include "cuda/test/utils.hpp"


namespace {


#if GINKGO_EMULATE_DOUBLE
using double_arithmetic = gko::emulated_double;
#else
using double_arithmetic = double;
#endif


TEST(DeviceArithmetic, ReplacesOnlyDouble)
{
    static_assert(
        std::is_same_v<gko::device_arithmetic_t<double>, double_arithmetic>);
    static_assert(std::is_same_v<gko::device_arithmetic_t<const double>,
                                 double_arithmetic>);
    static_assert(std::is_same_v<gko::device_arithmetic_t<float>, float>);
    static_assert(
        std::is_same_v<gko::device_arithmetic_t<gko::half>, gko::half>);
    static_assert(std::is_same_v<gko::device_arithmetic_t<std::complex<double>>,
                                 std::complex<double>>);
}


TEST(Tolerance, UsesPrecisionOfDeviceArithmetic)
{
#if GINKGO_EMULATE_DOUBLE
    // two float words hold 2 * 24 significand bits
    EXPECT_DOUBLE_EQ(r<double>::value, 10 * 0x1p-47);
#else
    EXPECT_DOUBLE_EQ(r<double>::value,
                     10 * std::numeric_limits<double>::epsilon());
#endif
    EXPECT_FLOAT_EQ(r<float>::value,
                    10 * std::numeric_limits<float>::epsilon());
    // complex<double> is computed in itself
    EXPECT_DOUBLE_EQ(r<std::complex<double>>::value,
                     10 * std::numeric_limits<double>::epsilon() * 1.4142);
}


}  // namespace
