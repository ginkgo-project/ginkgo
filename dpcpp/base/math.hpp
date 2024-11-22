// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_BASE_MATH_HPP_
#define GKO_DPCPP_BASE_MATH_HPP_

#include <climits>
#include <cmath>

#include <sycl/half_type.hpp>

#include <ginkgo/core/base/math.hpp>

#include "dpcpp/base/complex.hpp"
#include "dpcpp/base/dpct.hpp"

namespace gko {
namespace detail {


template <>
struct basic_float_traits<sycl::half> {
    using type = sycl::half;
    static constexpr int sign_bits = 1;
    static constexpr int significand_bits = 10;
    static constexpr int exponent_bits = 5;
    static constexpr bool rounds_to_nearest = true;
};


template <>
struct is_complex_or_scalar_impl<sycl::half> : public std::true_type {};


}  // namespace detail


bool __dpct_inline__ is_nan(const sycl::half& val)
{
    return std::isnan(static_cast<float>(val));
}

bool __dpct_inline__ is_nan(const std::complex<sycl::half>& val)
{
    return is_nan(val.real()) || is_nan(val.imag());
}


sycl::half __dpct_inline__ abs(const sycl::half& val)
{
    return abs(static_cast<float>(val));
}

sycl::half __dpct_inline__ abs(const std::complex<sycl::half>& val)
{
    return abs(static_cast<std::complex<float>>(val));
}

sycl::half __dpct_inline__ sqrt(const sycl::half& val)
{
    return sqrt(static_cast<float>(val));
}

std::complex<sycl::half> __dpct_inline__
sqrt(const std::complex<sycl::half>& val)
{
    return sqrt(static_cast<std::complex<float>>(val));
}


bool __dpct_inline__ is_finite(const sycl::half& value)
{
    return abs(value) < std::numeric_limits<sycl::half>::infinity();
}

bool __dpct_inline__ is_finite(const std::complex<sycl::half>& value)
{
    return is_finite(value.real()) && is_finite(value.imag());
}


}  // namespace gko


#endif  // GKO_DPCPP_BASE_MATH_HPP_
