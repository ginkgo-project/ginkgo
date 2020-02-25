/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

// We need this struct, because otherwise we would call a __host__ function in a
// __device__ function (even though it is constexpr)
template <typename T>
struct device_numeric_limits {
    static constexpr auto inf = std::numeric_limits<T>::infinity();
    static constexpr auto max = std::numeric_limits<T>::max();
    static constexpr auto min = std::numeric_limits<T>::min();
};


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


/**
 * Checks if a given value is finite, meaning it is neither +/- infinity
 * nor NaN.
 *
 * @internal  Should only be used if the provided one (from CUDA or HIP) can
 *            not be used.
 *            Designed to mirror the math function of CUDA (PTX code was
 *            identical in the testcase).
 *
 * @tparam T  type of the value to check
 *
 * @param value  value to check
 *
 * returns `true` if the given value is finite, meaning it is neither
 *         +/- infinity nor NaN.
 */
template <typename T>
GKO_INLINE __device__ xstd::enable_if_t<!is_complex_s<T>::value, bool>
custom_isfinite(T value)
{
    constexpr T infinity{device_numeric_limits<T>::inf};
    return abs(value) < infinity;
}

template <typename T>
GKO_INLINE __device__ xstd::enable_if_t<is_complex_s<T>::value, bool>
custom_isfinite(T value)
{
    return custom_isfinite(value.real()) && custom_isfinite(value.imag());
}


}  // namespace detail


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


#define GKO_DEFINE_ISFINITE_FOR_TYPE(_type)          \
    GKO_INLINE __device__ bool isfinite(_type value) \
    {                                                \
        return detail::custom_isfinite(value);       \
    }

GKO_DEFINE_ISFINITE_FOR_TYPE(float)
GKO_DEFINE_ISFINITE_FOR_TYPE(double)
#undef GKO_DEFINE_ISFINITE_FOR_TYPE


// For all other compiler in combination with CUDA or HIP, just use the provided
// `isfinite` function
#elif defined(__CUDA_ARCH__) || __HIP_DEVICE_COMPILE__


// If it is compiled with the CUDA or HIP compiler, use their `isfinite`
using ::isfinite;


#endif  // defined(__CUDA_ARCH__) || __HIP_DEVICE_COMPILE__
