/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_BASE_MATH_HPP_
#define GKO_CORE_BASE_MATH_HPP_


#include "core/base/types.hpp"


namespace gko {


/**
 * Returns the conjugate of a number.
 *
 * @param x  the number to conjugate
 *
 * @return  conjugate of `x`
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE std::complex<T> conj(const std::complex<T> &x)
{
    return std::conj(x);
}

/**
 * @copydoc conj(std::complex<T>)
 *
 * @note This is the overload for real types, which acts as an identity.
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE T conj(T x)
{
    return x;
}


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
 * Returns the multiplicative identity for T.
 *
 * @return the multiplicative identity for T
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr T one()
{
    return T(1);
}


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
struct is_complex_impl : public std::integral_constant<bool, false> {
};

template <typename T>
struct is_complex_impl<std::complex<T>>
    : public std::integral_constant<bool, true> {
};


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
 * @return  `true` if T is a complex type, `false` otherwise
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


/**
 * Returns the squared norm of the object.
 *
 * @tparam T type of the object.
 *
 * @return  The squared norm of the object.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr remove_complex<T> squared_norm(const T &x)
{
    using std::real;
    return real(gko::conj(x) * x);
}


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
     */
    GKO_ATTRIBUTES R operator()(S val) { return static_cast<R>(val); }
};


}  // namespace gko


#endif  // GKO_CORE_BASE_MATH_HPP_
