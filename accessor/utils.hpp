/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_ACCESSOR_UTILS_HPP_
#define GKO_ACCESSOR_UTILS_HPP_

#include <cassert>
#include <complex>
#include <cstddef>  // for std::size_t


#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/complex.h>
#endif


#if defined(__CUDACC__) || defined(__HIPCC__)
#define GKO_ACC_ATTRIBUTES __host__ __device__
#define GKO_ACC_INLINE __forceinline__
#define GKO_ACC_RESTRICT __restrict__
#else
#define GKO_ACC_ATTRIBUTES
#define GKO_ACC_INLINE inline
#define GKO_ACC_RESTRICT
#endif  // defined(__CUDACC__) || defined(__HIPCC__)


#if (defined(__CUDA_ARCH__) && defined(__APPLE__)) || \
    defined(__HIP_DEVICE_COMPILE__)

#ifdef NDEBUG
#define GKO_ACC_ASSERT(condition) ((void)0)
#else  // NDEBUG
// Poor man's assertions on GPUs for MACs. They won't terminate the program
// but will at least print something on the screen
#define GKO_ACC_ASSERT(condition)                                           \
    ((condition)                                                            \
         ? ((void)0)                                                        \
         : ((void)printf("%s: %d: %s: Assertion `" #condition "' failed\n", \
                         __FILE__, __LINE__, __func__)))
#endif  // NDEBUG

#else  // (defined(__CUDA_ARCH__) && defined(__APPLE__)) ||
       // defined(__HIP_DEVICE_COMPILE__)

// Handle assertions normally on other systems
#define GKO_ACC_ASSERT(condition) assert(condition)

#endif  // (defined(__CUDA_ARCH__) && defined(__APPLE__)) ||
        // defined(__HIP_DEVICE_COMPILE__)


namespace gko {
namespace acc {

namespace xstd {


template <typename...>
using void_t = void;


}


using size_type = std::size_t;


namespace detail {


template <typename T>
struct remove_complex_impl {
    using type = T;
};


template <typename T>
struct remove_complex_impl<std::complex<T>> {
    using type = T;
};


#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T>
struct remove_complex_impl<thrust::complex<T>> {
    using type = T;
};
#endif


template <typename T>
struct is_complex_impl {
    static constexpr bool value{false};
};


template <typename T>
struct is_complex_impl<std::complex<T>> {
    static constexpr bool value{true};
};


#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T>
struct is_complex_impl<thrust::complex<T>> {
    static constexpr bool value{true};
};
#endif


}  // namespace detail


template <typename T>
using remove_complex_t = typename detail::remove_complex_impl<T>::type;


template <typename T>
using is_complex = typename detail::is_complex_impl<T>;


/**
 * Evaluates if all template arguments Args fulfill std::is_integral. If that is
 * the case, this class inherits from `std::true_type`, otherwise, it inherits
 * from `std::false_type`.
 * If no values are passed in, `std::true_type` is inherited from.
 *
 * @tparam Args...  Arguments to test for std::is_integral
 */
template <typename... Args>
struct are_all_integral : public std::true_type {};

template <typename First, typename... Args>
struct are_all_integral<First, Args...>
    : public std::conditional_t<std::is_integral<std::decay_t<First>>::value,
                                are_all_integral<Args...>, std::false_type> {};


}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_UTILS_HPP_
