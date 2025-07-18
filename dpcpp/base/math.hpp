// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_BASE_MATH_HPP_
#define GKO_DPCPP_BASE_MATH_HPP_

#include <climits>
#include <cmath>

#if GINKGO_DPCPP_MAJOR_VERSION < 7 || \
    (GINKGO_DPCPP_MAJOR_VERSION == 7 && GINKGO_DPCPP_MINOR_VERSION < 1)
// bit_cast.hpp became self-contained during SYCL 7.0
// ref: https://github.com/intel/llvm/pull/9684
#include <sycl/detail/defines_elementary.hpp>
#endif
#include <sycl/bit_cast.hpp>
#include <sycl/half_type.hpp>

#include <ginkgo/core/base/math.hpp>

#include "dpcpp/base/bf16_alias.hpp"
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
struct basic_float_traits<vendor_bf16> {
    using type = vendor_bf16;
    static constexpr int sign_bits = 1;
    static constexpr int significand_bits = 7;
    static constexpr int exponent_bits = 8;
    static constexpr bool rounds_to_nearest = true;
};


template <>
struct is_complex_or_scalar_impl<sycl::half> : public std::true_type {};

template <>
struct is_complex_or_scalar_impl<vendor_bf16> : public std::true_type {};

template <typename ValueType>
struct complex_helper {
    using type = std::complex<ValueType>;
};

template <>
struct complex_helper<sycl::half> {
    using type = gko::complex<sycl::half>;
};

template <>
struct complex_helper<vendor_bf16> {
    using type = gko::complex<vendor_bf16>;
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

template <>
struct highest_precision_impl<sycl::half, vendor_bf16> {
    using type = float;
};

template <>
struct highest_precision_impl<vendor_bf16, sycl::half> {
    using type = float;
};

template <>
struct highest_precision_impl<vendor_bf16, float> {
    using type = float;
};

template <>
struct highest_precision_impl<float, vendor_bf16> {
    using type = float;
};

template <>
struct highest_precision_impl<vendor_bf16, double> {
    using type = double;
};

template <>
struct highest_precision_impl<double, vendor_bf16> {
    using type = double;
};


template <typename T1, typename T2>
struct highest_precision_impl<gko::complex<T1>, gko::complex<T2>> {
    using type = typename complex_helper<
        typename highest_precision_impl<T1, T2>::type>::type;
};

template <typename T1, typename T2>
struct highest_precision_impl<std::complex<T1>, gko::complex<T2>> {
    using type = typename complex_helper<
        typename highest_precision_impl<T1, T2>::type>::type;
};

template <typename T1, typename T2>
struct highest_precision_impl<gko::complex<T1>, std::complex<T2>> {
    using type = typename complex_helper<
        typename highest_precision_impl<T1, T2>::type>::type;
};


}  // namespace detail


// currently, std::numeric_limits<bfloat16> from sycl is wrong or use the
// default implementation.
//
template <typename T>
struct device_numeric_limits {
    static constexpr auto inf() { return std::numeric_limits<T>::infinity(); }
    static constexpr auto max() { return std::numeric_limits<T>::max(); }
    static constexpr auto min() { return std::numeric_limits<T>::min(); }
};

// There is no underlying data public access or storage_type input in
// constructor. we use sycl::bit_cast (not guaranteed be constexpr) to create
// the corresponding bfloat16
template <>
struct device_numeric_limits<vendor_bf16> {
    static GKO_ATTRIBUTES GKO_INLINE auto inf()
    {
        return sycl::bit_cast<vendor_bf16>(
            static_cast<unsigned short>(0b0'11111111'0000000u));
    }

    static GKO_ATTRIBUTES GKO_INLINE auto max()
    {
        return sycl::bit_cast<vendor_bf16>(
            static_cast<unsigned short>(0b0'11111110'1111111u));
    }

    static GKO_ATTRIBUTES GKO_INLINE auto min()
    {
        return sycl::bit_cast<vendor_bf16>(
            static_cast<unsigned short>(0b0'00000001'0000000u));
    }
};


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


bool __dpct_inline__ is_nan(const vendor_bf16& val)
{
    return std::isnan(static_cast<float>(val));
}

bool __dpct_inline__ is_nan(const gko::complex<vendor_bf16>& val)
{
    return is_nan(val.real()) || is_nan(val.imag());
}


vendor_bf16 __dpct_inline__ abs(const vendor_bf16& val)
{
    return abs(static_cast<float>(val));
}

vendor_bf16 __dpct_inline__ abs(const gko::complex<vendor_bf16>& val)
{
    return abs(static_cast<std::complex<float>>(val));
}

vendor_bf16 __dpct_inline__ sqrt(const vendor_bf16& val)
{
    return sqrt(static_cast<float>(val));
}

gko::complex<vendor_bf16> __dpct_inline__
sqrt(const gko::complex<vendor_bf16>& val)
{
    return sqrt(static_cast<std::complex<float>>(val));
}


bool __dpct_inline__ is_finite(const vendor_bf16& value)
{
    return abs(value) < device_numeric_limits<vendor_bf16>::inf();
}

bool __dpct_inline__ is_finite(const gko::complex<vendor_bf16>& value)
{
    return is_finite(value.real()) && is_finite(value.imag());
}


}  // namespace gko


#endif  // GKO_DPCPP_BASE_MATH_HPP_
