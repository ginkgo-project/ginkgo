// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_BASE_MATH_HPP_
#define GKO_DPCPP_BASE_MATH_HPP_

#include <climits>
#include <cmath>

#include <sycl/half_type.hpp>

#include <ginkgo/core/base/math.hpp>

#include "dpcpp/base/dpct.hpp"


namespace std {


template <>
class complex<sycl::half> {
public:
    using value_type = sycl::half;

    complex(const value_type& real = value_type(0.f),
            const value_type& imag = value_type(0.f))
        : real_(real), imag_(imag)
    {}

    template <typename T, typename U,
              typename = std::enable_if_t<std::is_scalar<T>::value &&
                                          std::is_scalar<U>::value>>
    explicit complex(const T& real, const U& imag)
        : real_(static_cast<value_type>(real)),
          imag_(static_cast<value_type>(imag))
    {}

    template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
    complex(const T& real)
        : real_(static_cast<value_type>(real)),
          imag_(static_cast<value_type>(0.f))
    {}

    template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
    complex(const complex<T>& other)
        : real_(static_cast<value_type>(other.real())),
          imag_(static_cast<value_type>(other.imag()))
    {}

    value_type real() const noexcept { return real_; }

    value_type imag() const noexcept { return imag_; }

    operator std::complex<float>() const noexcept
    {
        return std::complex<float>(static_cast<float>(real_),
                                   static_cast<float>(imag_));
    }

    template <typename V>
    complex& operator=(const V& val)
    {
        real_ = val;
        imag_ = value_type();
        return *this;
    }

    template <typename V>
    complex& operator=(const std::complex<V>& val)
    {
        real_ = val.real();
        imag_ = val.imag();
        return *this;
    }

    complex& operator+=(const value_type& real)
    {
        real_ += real;
        return *this;
    }

    complex& operator-=(const value_type& real)
    {
        real_ -= real;
        return *this;
    }

    complex& operator*=(const value_type& real)
    {
        real_ *= real;
        imag_ *= real;
        return *this;
    }

    complex& operator/=(const value_type& real)
    {
        real_ /= real;
        imag_ /= real;
        return *this;
    }

    template <typename T>
    complex& operator+=(const complex<T>& val)
    {
        real_ += val.real();
        imag_ += val.imag();
        return *this;
    }

    template <typename T>
    complex& operator-=(const complex<T>& val)
    {
        real_ -= val.real();
        imag_ -= val.imag();
        return *this;
    }

    template <typename T>
    complex& operator*=(const complex<T>& val)
    {
        auto val_f = static_cast<std::complex<float>>(val);
        auto result_f = static_cast<std::complex<float>>(*this);
        result_f *= val_f;
        real_ = result_f.real();
        imag_ = result_f.imag();
        return *this;
    }

    template <typename T>
    complex& operator/=(const complex<T>& val)
    {
        auto val_f = static_cast<std::complex<float>>(val);
        auto result_f = static_cast<std::complex<float>>(*this);
        result_f /= val_f;
        real_ = result_f.real();
        imag_ = result_f.imag();
        return *this;
    }

// It's for MacOS.
// TODO: check whether mac compiler always use complex version even when real
// half
#define COMPLEX_HALF_OPERATOR(_op, _opeq)                                  \
    friend complex<sycl::half> operator _op(const complex<sycl::half> lhf, \
                                            const complex<sycl::half> rhf) \
    {                                                                      \
        auto a = lhf;                                                      \
        a _opeq rhf;                                                       \
        return a;                                                          \
    }

    COMPLEX_HALF_OPERATOR(+, +=)
    COMPLEX_HALF_OPERATOR(-, -=)
    COMPLEX_HALF_OPERATOR(*, *=)
    COMPLEX_HALF_OPERATOR(/, /=)

#undef COMPLEX_HALF_OPERATOR

private:
    value_type real_;
    value_type imag_;
};

}  // namespace std


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
