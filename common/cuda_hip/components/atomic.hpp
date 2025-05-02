// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_COMPONENTS_ATOMIC_HPP_
#define GKO_COMMON_CUDA_HIP_COMPONENTS_ATOMIC_HPP_


#include <type_traits>

#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/types.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace detail {


template <typename ValueType, typename = void>
struct atomic_helper {
    __forceinline__ __device__ static ValueType atomic_add(ValueType*,
                                                           ValueType)
    {
        static_assert(sizeof(ValueType) == 0,
                      "This default function is not implemented, only the "
                      "specializations are.");
        // TODO: add proper implementation of generic atomic add
    }
    __forceinline__ __device__ static ValueType atomic_max(ValueType*,
                                                           ValueType)
    {
        static_assert(sizeof(ValueType) == 0,
                      "This default function is not implemented, only the "
                      "specializations are.");
        // TODO: add proper implementation of generic atomic max
    }
};


// TODO: consider it implemented by memcpy.
template <typename ResultType, typename ValueType>
__forceinline__ __device__ ResultType reinterpret(ValueType val)
{
    static_assert(sizeof(ValueType) == sizeof(ResultType),
                  "The type to reinterpret to must be of the same size as the "
                  "original type.");
    return reinterpret_cast<ResultType&>(val);
}


#define GKO_BIND_ATOMIC_HELPER_STRUCTURE(CONVERTER_TYPE)                     \
    template <typename ValueType>                                            \
    struct atomic_helper<                                                    \
        ValueType,                                                           \
        std::enable_if_t<(sizeof(ValueType) == sizeof(CONVERTER_TYPE))>> {   \
        __forceinline__ __device__ static ValueType atomic_add(              \
            ValueType* __restrict__ addr, ValueType val)                     \
        {                                                                    \
            using c_type = CONVERTER_TYPE;                                   \
            return atomic_wrapper(addr, [&val](c_type& old, c_type assumed,  \
                                               c_type* c_addr) {             \
                old = atomicCAS(c_addr, assumed,                             \
                                reinterpret<c_type>(                         \
                                    val + reinterpret<ValueType>(assumed))); \
            });                                                              \
        }                                                                    \
        __forceinline__ __device__ static ValueType atomic_max(              \
            ValueType* __restrict__ addr, ValueType val)                     \
        {                                                                    \
            using c_type = CONVERTER_TYPE;                                   \
            return atomic_wrapper(                                           \
                addr, [&val](c_type& old, c_type assumed, c_type* c_addr) {  \
                    if (reinterpret<ValueType>(assumed) < val) {             \
                        old = atomicCAS(c_addr, assumed,                     \
                                        reinterpret<c_type>(val));           \
                    }                                                        \
                });                                                          \
        }                                                                    \
                                                                             \
    private:                                                                 \
        template <typename Callable>                                         \
        __forceinline__ __device__ static ValueType atomic_wrapper(          \
            ValueType* __restrict__ addr, Callable set_old)                  \
        {                                                                    \
            CONVERTER_TYPE* address_as_converter =                           \
                reinterpret_cast<CONVERTER_TYPE*>(addr);                     \
            CONVERTER_TYPE old = *address_as_converter;                      \
            CONVERTER_TYPE assumed;                                          \
            do {                                                             \
                assumed = old;                                               \
                set_old(old, assumed, address_as_converter);                 \
            } while (assumed != old);                                        \
            return reinterpret<ValueType>(old);                              \
        }                                                                    \
    };

// Support 64-bit ATOMIC_ADD and ATOMIC_MAX
GKO_BIND_ATOMIC_HELPER_STRUCTURE(unsigned long long int);
// Support 32-bit ATOMIC_ADD and ATOMIC_MAX
GKO_BIND_ATOMIC_HELPER_STRUCTURE(unsigned int);


#if defined(CUDA_VERSION) && (__CUDA_ARCH__ >= 700)
// Support 16-bit atomicCAS, atomicADD, and atomicMAX only on CUDA with CC
// >= 7.0
GKO_BIND_ATOMIC_HELPER_STRUCTURE(unsigned short int);
#endif


#undef GKO_BIND_ATOMIC_HELPER_STRUCTURE


}  // namespace detail


template <typename T>
__forceinline__ __device__ T atomic_add(T* __restrict__ addr, T val)
{
    return detail::atomic_helper<T>::atomic_add(addr, val);
}


#define GKO_BIND_ATOMIC_ADD(ValueType)               \
    __forceinline__ __device__ ValueType atomic_add( \
        ValueType* __restrict__ addr, ValueType val) \
    {                                                \
        return atomicAdd(addr, val);                 \
    }

GKO_BIND_ATOMIC_ADD(int);
GKO_BIND_ATOMIC_ADD(unsigned int);
GKO_BIND_ATOMIC_ADD(unsigned long long int);
GKO_BIND_ATOMIC_ADD(float);

// AMD
#if defined(__HIPCC__) && GINKGO_HIP_PLATFORM_HCC


// the double atomicAdd is added after 4.3
GKO_BIND_ATOMIC_ADD(double);


#else  // NVIDIA


#if !(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600))
// CUDA supports 64-bit double atomicAdd on devices of compute
// capability 6.x and higher
GKO_BIND_ATOMIC_ADD(double);
#endif  // !(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600))

#if !(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700))
// CUDA supports 16-bit __half floating-point atomicAdd on devices
// of compute capability 7.x and higher.
GKO_BIND_ATOMIC_ADD(__half);
#endif  // !(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700))

#if !(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
// CUDA supports 16-bit __nv_bfloat16 floating-point atomicAdd on devices
// of compute capability 8.x and higher.
GKO_BIND_ATOMIC_ADD(__nv_bfloat16);
#endif  // !(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))

#if !(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600))
// CUDA supports 32-bit __half2 floating-point atomicAdd on
// devices of compute capability 6.x and higher. note: The atomicity of the
// __half2 add operation is guaranteed separately for each of the two __half
// elements; the entire __half2 is not guaranteed to be atomic as a single
// 32-bit access.
GKO_BIND_ATOMIC_ADD(__half2);
#endif  // !(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600))

#if !(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)) || \
    !(defined(CUDA_VERSION) && CUDA_VERSION < 12020)
// CUDA supports 32-bit __half2 floating-point atomicAdd natively on
// devices of compute capability 8.x and higher or cuda version is later
// than 12.2. note: The atomicity of the
// __nv_bfloat162 add operation is guaranteed separately for each of the two
// __nv_bfloat16 elements; the entire __nv_bfloat162 is not guaranteed to be
// atomic as a single 32-bit access.
GKO_BIND_ATOMIC_ADD(__nv_bfloat162);
#endif


#endif  // defined(__HIPCC__) && GINKGO_HIP_PLATFORM_HCC


#undef GKO_BIND_ATOMIC_ADD


template <typename T>
__forceinline__ __device__ T atomic_max(T* __restrict__ addr, T val)
{
    return detail::atomic_helper<T>::atomic_max(addr, val);
}


#define GKO_BIND_ATOMIC_MAX(ValueType)               \
    __forceinline__ __device__ ValueType atomic_max( \
        ValueType* __restrict__ addr, ValueType val) \
    {                                                \
        return atomicMax(addr, val);                 \
    }

GKO_BIND_ATOMIC_MAX(int);
GKO_BIND_ATOMIC_MAX(unsigned int);

#if !defined(__HIPCC__) || \
    (defined(__HIP_DEVICE_COMPILE__) && GINKGO_HIP_PLATFORM_NVCC)


#if defined(__CUDA_ARCH__) && (350 <= __CUDA_ARCH__)
// Only Compute Capability 3.5 and higher supports 64-bit atomicMax
GKO_BIND_ATOMIC_MAX(unsigned long long int);
#endif

#else   // Is HIP platform & on AMD hardware
GKO_BIND_ATOMIC_MAX(unsigned long long int);
#endif  // !defined(__HIPCC__) || (defined(__HIP_DEVICE_COMPILE__) &&
        // GINKGO_HIP_PLATFORM_HCC)


#undef GKO_BIND_ATOMIC_MAX


/**
 * @internal
 *
 * @note It is not 'real' complex<float> atomic add operation
 */
__forceinline__ __device__ thrust::complex<float> atomic_add(
    thrust::complex<float>* __restrict__ address, thrust::complex<float> val)
{
    auto addr = reinterpret_cast<float*>(address);
    // Separate to real part and imag part
    auto real = atomic_add(addr, val.real());
    auto imag = atomic_add(addr + 1, val.imag());
    return {real, imag};
}


/**
 * @internal
 *
 * @note It is not 'real' complex<double> atomic add operation
 */
__forceinline__ __device__ thrust::complex<double> atomic_add(
    thrust::complex<double>* __restrict__ address, thrust::complex<double> val)
{
    auto addr = reinterpret_cast<double*>(address);
    // Separate to real part and imag part
    auto real = atomic_add(addr, val.real());
    auto imag = atomic_add(addr + 1, val.imag());
    return {real, imag};
}


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_CUDA_HIP_COMPONENTS_ATOMIC_HPP_
