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

template <typename ValueType>
struct complex_helper {
    using type = std::complex<ValueType>;
};

template <>
struct complex_helper<sycl::half> {
    using type = gko::complex<sycl::half>;
};


template <typename T>
struct type_size_impl<gko::complex<T>> {
    static constexpr auto value = sizeof(T) * byte_size;
};


template <typename T>
struct remove_complex_impl<gko::complex<T>> {
    using type = T;
};


template <typename T>
struct truncate_type_impl<gko::complex<T>> {
    using type =
        typename complex_helper<typename truncate_type_impl<T>::type>::type;
};

template <typename T>
struct is_complex_impl<gko::complex<T>> : public std::true_type {};

template <typename T>
struct is_complex_or_scalar_impl<gko::complex<T>>
    : public is_complex_or_scalar_impl<T> {};

}  // namespace detail


bool __dpct_inline__ is_nan(const sycl::half& val)
{
    return std::isnan(static_cast<float>(val));
}

bool __dpct_inline__ is_nan(const gko::complex<sycl::half>& val)
{
    return is_nan(val.real()) || is_nan(val.imag());
}


sycl::half __dpct_inline__ abs(const sycl::half& val)
{
    return abs(static_cast<float>(val));
}

sycl::half __dpct_inline__ abs(const gko::complex<sycl::half>& val)
{
    return abs(static_cast<std::complex<float>>(val));
}

sycl::half __dpct_inline__ sqrt(const sycl::half& val)
{
    return sqrt(static_cast<float>(val));
}

gko::complex<sycl::half> __dpct_inline__
sqrt(const gko::complex<sycl::half>& val)
{
    return sqrt(static_cast<std::complex<float>>(val));
}


bool __dpct_inline__ is_finite(const sycl::half& value)
{
    return abs(value) < std::numeric_limits<sycl::half>::infinity();
}

bool __dpct_inline__ is_finite(const gko::complex<sycl::half>& value)
{
    return is_finite(value.real()) && is_finite(value.imag());
}


}  // namespace gko


#endif  // GKO_DPCPP_BASE_MATH_HPP_
