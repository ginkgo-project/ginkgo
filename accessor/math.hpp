/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_ACCESSOR_MATH_HPP_
#define GKO_ACCESSOR_MATH_HPP_

#include <type_traits>


#include "utils.hpp"


namespace gko {
namespace acc {
namespace detail {


// Note: All functions have postfix `impl` so they are not considered for
//       overload resolution (in case a class / function also is in the
//       namespace `detail`)
template <typename T>
constexpr GKO_ACC_ATTRIBUTES GKO_ACC_INLINE
    std::enable_if_t<!is_complex<T>::value, T>
    real_impl(const T& x)
{
    return x;
}

template <typename T>
constexpr GKO_ACC_ATTRIBUTES GKO_ACC_INLINE
    std::enable_if_t<is_complex<T>::value, remove_complex_t<T>>
    real_impl(const T& x)
{
    return x.real();
}


template <typename T>
constexpr GKO_ACC_ATTRIBUTES GKO_ACC_INLINE
    std::enable_if_t<!is_complex<T>::value, T>
    imag_impl(const T&)
{
    return T{};
}

template <typename T>
constexpr GKO_ACC_ATTRIBUTES GKO_ACC_INLINE
    std::enable_if_t<is_complex<T>::value, remove_complex_t<T>>
    imag_impl(const T& x)
{
    return x.imag();
}


template <typename T>
constexpr GKO_ACC_ATTRIBUTES GKO_ACC_INLINE
    std::enable_if_t<!is_complex<T>::value, T>
    conj_impl(const T& x)
{
    return x;
}

template <typename T>
constexpr GKO_ACC_ATTRIBUTES GKO_ACC_INLINE
    std::enable_if_t<is_complex<T>::value, T>
    conj_impl(const T& x)
{
    return T{real_impl(x), -imag_impl(x)};
}


}  // namespace detail


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
constexpr GKO_ACC_ATTRIBUTES GKO_ACC_INLINE auto real(const T& x)
{
    return detail::real_impl(detail::to_arithmetic_type(x));
}


/**
 * Returns the imaginary part of the object.
 *
 * @tparam T  type of the object
 *
 * @param x  the object
 *
 * @return imag part of the object (by default, zero)
 */
template <typename T>
constexpr GKO_ACC_ATTRIBUTES GKO_ACC_INLINE auto imag(const T& x)
{
    return detail::imag_impl(detail::to_arithmetic_type(x));
}


/**
 * Returns the conjugate of an object.
 *
 * @param x  the number to conjugate
 *
 * @return  conjugate of the object (by default, the object itself)
 */
template <typename T>
constexpr GKO_ACC_ATTRIBUTES GKO_ACC_INLINE auto conj(const T& x)
{
    return detail::conj_impl(detail::to_arithmetic_type(x));
}


/**
 * Returns the squared norm of the object.
 *
 * @tparam T type of the object.
 *
 * @return  The squared norm of the object.
 */
template <typename T>
constexpr GKO_ACC_ATTRIBUTES GKO_ACC_INLINE auto squared_norm(const T& x)
{
    return real(conj(x) * x);
}


}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_MATH_HPP_
