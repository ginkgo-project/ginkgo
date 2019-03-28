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

#ifndef GKO_CORE_BASE_MATH_HPP_
#define GKO_CORE_BASE_MATH_HPP_


#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>


#include <cmath>
#include <complex>
#include <cstdlib>


namespace gko {


// type manipulations


namespace detail {


/**
 * Keep the same data type if it is not complex.
 */
template <typename T>
struct remove_complex_impl {
    using type = T;
};

/**
 * Use the underlying real type if it is complex type.
 */
template <typename T>
struct remove_complex_impl<std::complex<T>> {
    using type = T;
};


template <typename T>
struct is_complex_impl : public std::integral_constant<bool, false> {};

template <typename T>
struct is_complex_impl<std::complex<T>>
    : public std::integral_constant<bool, true> {};


}  // namespace detail


/**
 * Obtains a real counterpart of a std::complex type, and leaves the type
 * unchanged if it is not a complex type.
 */
template <typename T>
using remove_complex = typename detail::remove_complex_impl<T>::type;


/**
 * Checks if T is a complex type.
 *
 * @tparam T  type to check
 *
 * @return `true` if T is a complex type, `false` otherwise
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr bool is_complex()
{
    return detail::is_complex_impl<T>::value;
}


namespace detail {


template <typename T>
struct reduce_precision_impl {
    using type = T;
};

template <typename T>
struct reduce_precision_impl<std::complex<T>> {
    using type = std::complex<typename reduce_precision_impl<T>::type>;
};

template <>
struct reduce_precision_impl<double> {
    using type = float;
};

template <>
struct reduce_precision_impl<float> {
    using type = half;
};


template <typename T>
struct increase_precision_impl {
    using type = T;
};

template <typename T>
struct increase_precision_impl<std::complex<T>> {
    using type = std::complex<typename increase_precision_impl<T>::type>;
};

template <>
struct increase_precision_impl<float> {
    using type = double;
};

template <>
struct increase_precision_impl<half> {
    using type = float;
};


}  // namespace detail


/**
 * Obtains the next type in the hierarchy with lower precision than T.
 */
template <typename T>
using reduce_precision = typename detail::reduce_precision_impl<T>::type;


/**
 * Obtains the next type in the hierarchy with higher precision than T.
 */
template <typename T>
using increase_precision = typename detail::increase_precision_impl<T>::type;


/**
 * Reduces the precision of the input parameter.
 *
 * @tparam T  the original precision
 *
 * @param val  the value to round down
 *
 * @return the rounded down value
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr reduce_precision<T> round_down(T val)
{
    return static_cast<reduce_precision<T>>(val);
}


/**
 * Increases the precision of the input parameter.
 *
 * @tparam T  the original precision
 *
 * @param val  the value to round up
 *
 * @return the rounded up value
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr increase_precision<T> round_up(T val)
{
    return static_cast<increase_precision<T>>(val);
}


template <typename FloatType, size_type NumComponents, size_type ComponentId>
class truncated;


namespace detail {


template <typename T>
struct truncate_type_impl {
    using type = truncated<T, 2, 0>;
};

template <typename T, size_type Components>
struct truncate_type_impl<truncated<T, Components, 0>> {
    using type = truncated<T, 2 * Components, 0>;
};

template <typename T>
struct truncate_type_impl<std::complex<T>> {
    using type = std::complex<typename truncate_type_impl<T>::type>;
};


template <typename T>
struct type_size_impl {
    static constexpr auto value = sizeof(T) * byte_size;
};

template <typename T>
struct type_size_impl<std::complex<T>> {
    static constexpr auto value = sizeof(T) * byte_size;
};


}  // namespace detail


/**
 * Truncates the type by half (by dropping bits), but ensures that it is at
 * least `Limit` bits wide.
 */
template <typename T, size_type Limit = sizeof(uint16) * byte_size>
using truncate_type =
    xstd::conditional_t<detail::type_size_impl<T>::value >= 2 * Limit,
                        typename detail::truncate_type_impl<T>::type, T>;


/**
 * Used to convert objects of type `S` to objects of type `R` using static_cast.
 *
 * @tparam S  source type
 * @tparam R  result type
 */
template <typename S, typename R>
struct default_converter {
    /**
     * Converts the object to result type.
     *
     * @param val  the object to convert
     * @return the converted object
     */
    GKO_ATTRIBUTES R operator()(S val) { return static_cast<R>(val); }
};


// mathematical functions


/**
 * Performs integer division with rounding up.
 *
 * @param num  numerator
 * @param den  denominator
 *
 * @return returns the ceiled quotient.
 */
GKO_INLINE GKO_ATTRIBUTES constexpr int64 ceildiv(int64 num, int64 den)
{
    return (num + den - 1) / den;
}


/**
 * Returns the additive identity for T.
 *
 * @return additive identity for T
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr T zero()
{
    return T(0);
}


/**
 * Returns the additive identity for T.
 *
 * @return additive identity for T
 *
 * @note This version takes an unused reference argument to avoid complicated
 *       calls like `zero<decltype(x)>()`. Instead, it allows `zero(x)`.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr T zero(const T &)
{
    return zero<T>();
}


/**
 * Returns the multiplicative identity for T.
 *
 * @return the multiplicative identity for T
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr T one()
{
    return T(1);
}


/**
 * Returns the multiplicative identity for T.
 *
 * @return the multiplicative identity for T
 *
 * @note This version takes an unused reference argument to avoid complicated
 *       calls like `one<decltype(x)>()`. Instead, it allows `one(x)`.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr T one(const T &)
{
    return one<T>();
}


/**
 * Returns the absolute value of the object.
 *
 * @tparam T  the type of the object
 *
 * @param x  the object
 *
 * @return x >= zero<T>() ? x : -x;
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr T abs(const T &x)
{
    return x >= zero<T>() ? x : -x;
}


using std::abs;  // use optimized abs functions for basic types


/**
 * Returns the larger of the arguments.
 *
 * @tparam T  type of the arguments
 *
 * @param x  first argument
 * @param y  second argument
 *
 * @return x >= y ? x : y
 *
 * @note C++11 version of this function is not constexpr, thus we provide our
 *       own implementation.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr T max(const T &x, const T &y)
{
    return x >= y ? x : y;
}


/**
 * Returns the smaller of the arguments.
 *
 * @tparam T  type of the arguments
 *
 * @param x  first argument
 * @param y  second argument
 *
 * @return x <= y ? x : y
 *
 * @note C++11 version of this function is not `constexpr`, thus we provide our
 *       own implementation.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr T min(const T &x, const T &y)
{
    return x <= y ? x : y;
}


/**
 * Returns the real part of the object.
 *
 * @tparam T  type of the object
 *
 * @param x  the object
 *
 * @return real part of the object (by default, the object itself)
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE constexpr T real(const T &x)
{
    return x;
}


/**
 * Returns the imaginary part of the object.
 *
 * @tparam T  type of the object
 *
 * @param x  the object
 *
 * @return imaginary part of the object (by default, zero<T>())
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE constexpr T imag(const T &)
{
    return zero<T>();
}


/**
 * Returns the conjugate of an object.
 *
 * @param x  the number to conjugate
 *
 * @return  conjugate of the object (by default, the object itself)
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE T conj(const T &x)
{
    return x;
}


using std::sqrt;  // use standard sqrt functions for basic types


/**
 * Returns the squared norm of the object.
 *
 * @tparam T type of the object.
 *
 * @return  The squared norm of the object.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr auto squared_norm(const T &x)
    -> decltype(real(conj(x) * x))
{
    return real(conj(x) * x);
}


/**
 * Returns the position of the most significant bit of the number.
 *
 * This is the same as the rounded down base-2 logarithm of the number.
 *
 * @tparam T  a numeric type supporting bit shift and comparison
 *
 * @param n  a number
 * @param hint  a lower bound for the position o the significant bit
 *
 * @return maximum of `hint` and the significant bit position of `n`
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr uint32 get_significant_bit(
    const T &n, uint32 hint = 0u) noexcept
{
    return (T{1} << (hint + 1)) > n ? hint : get_significant_bit(n, hint + 1u);
}


/**
 * Returns the smallest power of `base` not smaller than `limit`.
 *
 * @tparam T  a numeric type supporting multiplication and comparison
 *
 * @param base  the base of the power to be returned
 * @param limit  the lower limit on the size of the power returned
 * @param hint  a lower bound on the result, has to be a power of base
 *
 * @return the smallest power of `base` not smaller than `limit`
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr T get_superior_power(
    const T &base, const T &limit, const T &hint = T{1}) noexcept
{
    return hint >= limit ? hint : get_superior_power(base, limit, hint * base);
}


}  // namespace gko


#endif  // GKO_CORE_BASE_MATH_HPP_
