// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_MATH_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_MATH_HPP_


#include <thrust/complex.h>

#include <ginkgo/core/base/math.hpp>


#ifdef GKO_COMPILING_CUDA


#include <cuda_fp16.h>


#elif defined(GKO_COMPILING_HIP)


#include <hip/hip_fp16.h>


#endif

#include "common/cuda_hip/base/bf16_alias.hpp"
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
        bits.x = static_cast<uint16>(0b0111110000000000u);
        return __half{bits};
    }

    static GKO_ATTRIBUTES GKO_INLINE auto max()
    {
        __half_raw bits;
        bits.x = static_cast<uint16>(0b0111101111111111u);
        return __half{bits};
    }

    static GKO_ATTRIBUTES GKO_INLINE auto min()
    {
        __half_raw bits;
        bits.x = static_cast<uint16>(0b0000010000000000u);
        return __half{bits};
    }
};


#if defined(GKO_COMPILING_CUDA)


template <>
struct device_numeric_limits<__nv_bfloat16> {
    static GKO_ATTRIBUTES GKO_INLINE auto inf()
    {
        __nv_bfloat16_raw bits;
        bits.x = static_cast<uint16>(0b0'11111111'0000000u);
        return __nv_bfloat16{bits};
    }

    static GKO_ATTRIBUTES GKO_INLINE auto max()
    {
        __nv_bfloat16_raw bits;
        bits.x = static_cast<uint16>(0b0'11111110'1111111u);
        return __nv_bfloat16{bits};
    }

    static GKO_ATTRIBUTES GKO_INLINE auto min()
    {
        __nv_bfloat16_raw bits;
        bits.x = static_cast<uint16>(0b0'00000001'0000000u);
        return __nv_bfloat16{bits};
    }
};


#endif

#ifdef GKO_COMPILING_HIP


#if HIP_VERSION >= 60200000


template <>
struct device_numeric_limits<__hip_bfloat16> {
    static GKO_ATTRIBUTES GKO_INLINE auto inf()
    {
        __hip_bfloat16_raw bits;
        bits.x = static_cast<uint16>(0b0'11111111'0000000u);
        return __hip_bfloat16{bits};
    }

    static GKO_ATTRIBUTES GKO_INLINE auto max()
    {
        __hip_bfloat16_raw bits;
        bits.x = static_cast<uint16>(0b0'11111110'1111111u);
        return __hip_bfloat16{bits};
    }

    static GKO_ATTRIBUTES GKO_INLINE auto min()
    {
        __hip_bfloat16_raw bits;
        bits.x = static_cast<uint16>(0b0'00000001'0000000u);
        return __hip_bfloat16{bits};
    }
};


#else


template <>
struct device_numeric_limits<hip_bfloat16> {
    static GKO_ATTRIBUTES GKO_INLINE auto inf()
    {
        hip_bfloat16 vals;
        vals.data = static_cast<uint16>(0b0'11111111'0000000u);
        return vals;
    }

    static GKO_ATTRIBUTES GKO_INLINE auto max()
    {
        hip_bfloat16 vals;
        vals.data = static_cast<uint16>(0b0'11111110'1111111u);
        return vals;
    }

    static GKO_ATTRIBUTES GKO_INLINE auto min()
    {
        hip_bfloat16 vals;
        vals.data = static_cast<uint16>(0b0'00000001'0000000u);
        return vals;
    }
};


#endif
#endif

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


template <>
struct highest_precision_impl<__half, vendor_bf16> {
    using type = float;
};

template <>
struct highest_precision_impl<vendor_bf16, __half> {
    using type = float;
};

template <>
struct highest_precision_impl<__half, float> {
    using type = float;
};

template <>
struct highest_precision_impl<float, __half> {
    using type = float;
};

template <>
struct highest_precision_impl<__half, double> {
    using type = double;
};

template <>
struct highest_precision_impl<double, __half> {
    using type = double;
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
struct highest_precision_impl<GKO_THRUST_QUALIFIER::complex<T1>,
                              GKO_THRUST_QUALIFIER::complex<T2>> {
    using type = GKO_THRUST_QUALIFIER::complex<
        typename highest_precision_impl<T1, T2>::type>;
};


}  // namespace detail
}  // namespace gko


GKO_THRUST_NAMESPACE_PREFIX
namespace thrust {


#if GINKGO_ENABLE_HALF


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

#endif  // GINKGO_ENABLE_HALF


#if GINKGO_ENABLE_BFLOAT16

template <>
GKO_ATTRIBUTES GKO_INLINE complex<gko::vendor_bf16> sqrt<gko::vendor_bf16>(
    const complex<gko::vendor_bf16>& a)
{
    return static_cast<complex<gko::vendor_bf16>>(
        sqrt(static_cast<complex<float>>(a)));
}


template <>
GKO_ATTRIBUTES GKO_INLINE gko::vendor_bf16 abs<gko::vendor_bf16>(
    const complex<gko::vendor_bf16>& z)
{
    return static_cast<gko::vendor_bf16>(abs(static_cast<complex<float>>(z)));
}


#endif  // GINKGO_ENABLE_BFLOAT16


}  // namespace thrust
GKO_THRUST_NAMESPACE_POSTFIX


namespace gko {


// It is required by NVHPC 23.3, `isnan` is undefined when NVHPC is used as a
// host compiler.
#if defined(__CUDACC__) || defined(GKO_COMPILING_HIP)


#if GINKGO_ENABLE_HALF


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


#endif  // GINKGO_ENABLE_HALF


#if GINKGO_ENABLE_BFLOAT16


__device__ __forceinline__ bool is_nan(const vendor_bf16& val)
{
    // from the cuda_bf16.hpp, amd_hip_bf16.h
#if GINKGO_HIP_PLATFORM_HCC && HIP_VERSION >= 60200000 || \
    (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
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
#if GINKGO_HIP_PLATFORM_HCC && HIP_VERSION >= 60200000 || \
    (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
    return __habs(val);
#else
    return static_cast<vendor_bf16>(abs(static_cast<float>(val)));
#endif
}


__device__ __forceinline__ vendor_bf16 sqrt(const vendor_bf16& val)
{
#if GINKGO_HIP_PLATFORM_HCC && HIP_VERSION >= 60200000 || \
    (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
    return hsqrt(val);
#else
    return static_cast<vendor_bf16>(sqrt(static_cast<float>(val)));
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

#if defined(GKO_COMPILING_HIP) && HIP_VERSION < 60200000


// hip_bfloat16 does not have a constexpr constructor from int
template <>
GKO_INLINE vendor_bf16 one<vendor_bf16>()
{
    vendor_bf16 val;
    val.data = static_cast<uint16>(0b0'01111111'0000000u);
    return val;
}

// hip_bfloat16 does not have an implicit conversion from float
template <>
GKO_INLINE thrust::complex<vendor_bf16> one<thrust::complex<vendor_bf16>>()
{
    thrust::complex<vendor_bf16> val(one<vendor_bf16>());
    return val;
}


#endif
#endif  // GINKGO_ENABLE_BFLOAT16
#endif  // defined(__CUDACC__) || defined(GKO_COMPILING_HIP)


}  // namespace gko


#endif  // GKO_COMMON_CUDA_HIP_BASE_MATH_HPP_
