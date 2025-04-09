// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_MATH_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_MATH_HPP_


#include <thrust/complex.h>

#include <ginkgo/core/base/math.hpp>


#ifdef GKO_COMPILING_CUDA


#include <cuda_bf16.h>
#include <cuda_fp16.h>

using vendor_bf16 = __nv_bfloat16;


#elif defined(GKO_COMPILING_HIP)


#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

using vendor_bf16 = hip_bfloat16;


#endif


#include "common/cuda_hip/base/thrust_macro.hpp"


namespace gko {


// We need this struct, because otherwise we would call a __host__ function in a
// __device__ function (even though it is constexpr)
template <typename T>
struct device_numeric_limits {
    static constexpr auto inf() { return std::numeric_limits<T>::infinity(); }
    static constexpr auto max() { return std::numeric_limits<T>::max(); }
    static constexpr auto min() { return std::numeric_limits<T>::min(); }
};

template <>
struct device_numeric_limits<__half> {
    // from __half documentation, it accepts unsigned short
    // __half and __half_raw does not have constexpr constructor
    static GKO_ATTRIBUTES GKO_INLINE auto inf()
    {
        __half_raw bits;
        bits.x = static_cast<unsigned short>(0b0111110000000000u);
        return __half{bits};
    }

    static GKO_ATTRIBUTES GKO_INLINE auto max()
    {
        __half_raw bits;
        bits.x = static_cast<unsigned short>(0b0111101111111111u);
        return __half{bits};
    }

    static GKO_ATTRIBUTES GKO_INLINE auto min()
    {
        __half_raw bits;
        bits.x = static_cast<unsigned short>(0b0000010000000000u);
        return __half{bits};
    }
};


template <>
struct device_numeric_limits<__nv_bfloat16> {
    static GKO_ATTRIBUTES GKO_INLINE auto inf()
    {
        __nv_bfloat16_raw bits;
        bits.x = static_cast<unsigned short>(0b0'11111111'0000000u);
        return __nv_bfloat16{bits};
    }

    static GKO_ATTRIBUTES GKO_INLINE auto max()
    {
        __nv_bfloat16_raw bits;
        bits.x = static_cast<unsigned short>(0b0'11111110'1111111u);
        return __nv_bfloat16{bits};
    }

    static GKO_ATTRIBUTES GKO_INLINE auto min()
    {
        __nv_bfloat16_raw bits;
        bits.x = static_cast<unsigned short>(0b0'00000001'0000000u);
        return __nv_bfloat16{bits};
    }
};


namespace detail {


template <typename T>
struct remove_complex_impl<thrust::complex<T>> {
    using type = T;
};


template <typename T>
struct truncate_type_impl<thrust::complex<T>> {
    using type = thrust::complex<typename truncate_type_impl<T>::type>;
};


template <typename T>
struct type_size_impl<thrust::complex<T>> {
    static constexpr auto value = sizeof(T) * byte_size;
};


template <typename T>
struct is_complex_impl<thrust::complex<T>> : public std::true_type {};

template <>
struct is_complex_or_scalar_impl<__half> : public std::true_type {};

template <>
struct is_complex_or_scalar_impl<vendor_bf16> : public std::true_type {};

template <typename T>
struct is_complex_or_scalar_impl<thrust::complex<T>>
    : public is_complex_or_scalar_impl<T> {};


}  // namespace detail
}  // namespace gko


#if GINKGO_ENABLE_HALF


GKO_THRUST_NAEMSPACE_PREFIX
namespace thrust {


template <>
GKO_ATTRIBUTES GKO_INLINE complex<__half> sqrt<__half>(const complex<__half>& a)
{
    return sqrt(static_cast<complex<float>>(a));
}


template <>
GKO_ATTRIBUTES GKO_INLINE __half abs<__half>(const complex<__half>& z)
{
    return abs(static_cast<complex<float>>(z));
}


template <>
GKO_ATTRIBUTES GKO_INLINE complex<vendor_bf16> sqrt<vendor_bf16>(
    const complex<vendor_bf16>& a)
{
    return sqrt(static_cast<complex<float>>(a));
}


template <>
GKO_ATTRIBUTES GKO_INLINE vendor_bf16
abs<vendor_bf16>(const complex<vendor_bf16>& z)
{
    return abs(static_cast<complex<float>>(z));
}


}  // namespace thrust
GKO_THRUST_NAEMSPACE_POSTFIX


namespace gko {


// It is required by NVHPC 23.3, `isnan` is undefined when NVHPC is used as a
// host compiler.
#if defined(__CUDACC__) || defined(GKO_COMPILING_HIP)

__device__ __forceinline__ bool is_nan(const __half& val)
{
    // from the cuda_fp16.hpp
#if GINKGO_HIP_PLATFORM_HCC || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
    return __hisnan(val);
#else
    return isnan(static_cast<float>(val));
#endif
}

__device__ __forceinline__ bool is_nan(const thrust::complex<__half>& val)
{
    return is_nan(val.real()) || is_nan(val.imag());
}


__device__ __forceinline__ __half abs(const __half& val)
{
#if GINKGO_HIP_PLATFORM_HCC || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
    return __habs(val);
#else
    return abs(static_cast<float>(val));
#endif
}

__device__ __forceinline__ __half sqrt(const __half& val)
{
#if GINKGO_HIP_PLATFORM_HCC || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
    return hsqrt(val);
#else
    return sqrt(static_cast<float>(val));
#endif
}


// using overload here. Otherwise, compiler still think the is_finite
// specialization is still __host__ __device__ function.
__device__ __forceinline__ bool is_finite(const __half& value)
{
    return abs(value) < device_numeric_limits<__half>::inf();
}

__device__ __forceinline__ bool is_finite(const thrust::complex<__half>& value)
{
    return is_finite(value.real()) && is_finite(value.imag());
}


__device__ __forceinline__ bool is_nan(const vendor_bf16& val)
{
    // from the cuda_fp16.hpp
#if GINKGO_HIP_PLATFORM_HCC || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
    return __hisnan(val);
#else
    return isnan(static_cast<float>(val));
#endif
}

__device__ __forceinline__ bool is_nan(const thrust::complex<vendor_bf16>& val)
{
    return is_nan(val.real()) || is_nan(val.imag());
}


__device__ __forceinline__ vendor_bf16 abs(const vendor_bf16& val)
{
#if GINKGO_HIP_PLATFORM_HCC || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
    return __habs(val);
#else
    return abs(static_cast<float>(val));
#endif
}

__device__ __forceinline__ vendor_bf16 sqrt(const vendor_bf16& val)
{
#if GINKGO_HIP_PLATFORM_HCC || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
    return hsqrt(val);
#else
    return sqrt(static_cast<float>(val));
#endif
}


// using overload here. Otherwise, compiler still think the is_finite
// specialization is still __host__ __device__ function.
__device__ __forceinline__ bool is_finite(const vendor_bf16& value)
{
    return abs(value) < device_numeric_limits<vendor_bf16>::inf();
}

__device__ __forceinline__ bool is_finite(
    const thrust::complex<vendor_bf16>& value)
{
    return is_finite(value.real()) && is_finite(value.imag());
}

#endif


}  // namespace gko


#endif  // GINKGO_ENABLE_HALF


#endif  // GKO_COMMON_CUDA_HIP_BASE_MATH_HPP_
