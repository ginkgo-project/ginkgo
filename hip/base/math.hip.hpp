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

#ifndef GKO_HIP_BASE_MATH_HIP_HPP_
#define GKO_HIP_BASE_MATH_HIP_HPP_


#include <ginkgo/core/base/math.hpp>


#include <thrust/complex.h>


namespace gko {
namespace detail {


template <typename T>
struct remove_complex_impl<thrust::complex<T>> {
    using type = T;
};


template <typename T>
struct is_complex_impl<thrust::complex<T>>
    : public std::integral_constant<bool, true> {};


template <typename T>
struct truncate_type_impl<thrust::complex<T>> {
    using type = thrust::complex<typename truncate_type_impl<T>::type>;
};


}  // namespace detail


template <>
__device__ GKO_INLINE std::complex<float> zero<std::complex<float>>()
{
    thrust::complex<float> z(0);
    return reinterpret_cast<std::complex<float> &>(z);
}

template <>
__device__ GKO_INLINE std::complex<double> zero<std::complex<double>>()
{
    thrust::complex<double> z(0);
    return reinterpret_cast<std::complex<double> &>(z);
}

template <>
__device__ GKO_INLINE std::complex<float> one<std::complex<float>>()
{
    thrust::complex<float> z(1);
    return reinterpret_cast<std::complex<float> &>(z);
}

template <>
__device__ GKO_INLINE std::complex<double> one<std::complex<double>>()
{
    thrust::complex<double> z(1);
    return reinterpret_cast<std::complex<double> &>(z);
}


// This first part is specific for clang and intel in combination with the nvcc
// compiler from the toolkit older than 9.2.
// Both want to use their `__builtin_isfinite` function, which is not present
// as a __device__ function, so it results in a compiler error.
// Here, `isfinite` is written by hand, which might not be as performant as the
// intrinsic function from CUDA, but it compiles and works.
#if defined(__CUDA_ARCH__) &&                                           \
    (defined(_MSC_VER) ||                                               \
     (defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && \
      (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__) < 9002 &&    \
      (defined(__clang__) || defined(__ICC) || defined(__ICL))))


namespace detail {


/**
 * This structure can be used to get the exponent mask of a given floating
 * point type. Uses specialization to implement different types.
 */
template <typename T>
struct mask_creator {};

template <>
struct mask_creator<float> {
    using int_type = int32;
    static constexpr int_type number_exponent_bits = 8;
    static constexpr int_type number_significand_bits = 23;
    // integer representation of a floating point number, where all exponent
    // bits are set
    static constexpr int_type exponent_mask =
        ((int_type{1} << number_exponent_bits) - 1) << number_significand_bits;
    static __device__ int_type reinterpret_int(const float &value)
    {
        return __float_as_int(value);
    }
};

template <>
struct mask_creator<double> {
    using int_type = int64;
    static constexpr int_type number_exponent_bits = 11;
    static constexpr int_type number_significand_bits = 52;
    // integer representation of a floating point number, where all exponent
    // bits are set
    static constexpr int_type exponent_mask =
        ((int_type{1} << number_exponent_bits) - 1) << number_significand_bits;
    static __device__ int_type reinterpret_int(const double &value)
    {
        return __double_as_longlong(value);
    }
};


}  // namespace detail


/**
 * Checks if a given value is finite, meaning it is neither +/- infinity
 * nor NaN.
 *
 * @internal  It checks if all exponent bits are set. If all are set, the
 *            number either represents NaN or +/- infinity, meaning it is a
 *            non-finite number.
 *
 * @param value  value to check
 *
 * returns `true` if the given value is finite, meaning it is neither
 *         +/- infinity nor NaN.
 */
#define GKO_DEFINE_ISFINITE_FOR_TYPE(_type)                               \
    GKO_INLINE __device__ bool isfinite(const _type &value)               \
    {                                                                     \
        constexpr auto mask = detail::mask_creator<_type>::exponent_mask; \
        const auto re_int =                                               \
            detail::mask_creator<_type>::reinterpret_int(value);          \
        return (re_int & mask) != mask;                                   \
    }

GKO_DEFINE_ISFINITE_FOR_TYPE(float)
GKO_DEFINE_ISFINITE_FOR_TYPE(double)
#undef GKO_DEFINE_ISFINITE_FOR_TYPE


/**
 * Checks if all components of a complex value are finite, meaning they are
 * neither +/- infinity nor NaN.
 *
 * @internal required for the clang compiler. This function will be used rather
 *           than the `isfinite` function in the public `math.hpp` because
 *           there is no template parameter, so it is prefered during lookup.
 *
 * @tparam T  complex type of the value to check
 *
 * @param value  complex value to check
 *
 * returns `true` if both components of the given value are finite, meaning
 *         they are neither +/- infinity nor NaN.
 */
#define GKO_DEFINE_ISFINITE_FOR_COMPLEX_TYPE(_type)              \
    GKO_INLINE __device__ bool isfinite(const _type &value)      \
    {                                                            \
        return isfinite(value.real()) && isfinite(value.imag()); \
    }

GKO_DEFINE_ISFINITE_FOR_COMPLEX_TYPE(thrust::complex<float>)
GKO_DEFINE_ISFINITE_FOR_COMPLEX_TYPE(thrust::complex<double>)
#undef GKO_DEFINE_ISFINITE_FOR_COMPLEX_TYPE


#elif __HIP_DEVICE_COMPILE__


// If it is compiled with the HIP compiler, use their `isfinite`
using ::isfinite;


#endif  // __HIP_DEVICE_COMPILE__


}  // namespace gko


#endif  // GKO_HIP_BASE_MATH_HIP_HPP_
