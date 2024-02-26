// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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


#if !(defined(CUDA_VERSION) && (CUDA_VERSION < 10010))
// CUDA 10.1 starts supporting 16-bit unsigned short int atomicCAS
GKO_BIND_ATOMIC_HELPER_STRUCTURE(unsigned short int);
#endif  // !(defined(CUDA_VERSION) && (CUDA_VERSION < 10010))

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


#if !((defined(CUDA_VERSION) && (CUDA_VERSION < 8000)) || \
      (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)))
// CUDA 8.0 starts suppoting 64-bit double atomicAdd on devices of compute
// capability 6.x and higher
GKO_BIND_ATOMIC_ADD(double);
#endif  // !((defined(CUDA_VERSION) && (CUDA_VERSION < 8000)) ||
        // (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)))

#if !((defined(CUDA_VERSION) && (CUDA_VERSION < 10000)) || \
      (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
// CUDA 10.0 starts supporting 16-bit __half floating-point atomicAdd on devices
// of compute capability 7.x and higher.
GKO_BIND_ATOMIC_ADD(__half);
#endif  // !((defined(CUDA_VERSION) && (CUDA_VERSION < 10000)) ||
        // (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))

#if !((defined(CUDA_VERSION) && (CUDA_VERSION < 10000)) || \
      (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)))
// CUDA 10.0 starts supporting 32-bit __half2 floating-point atomicAdd on
// devices of compute capability 6.x and higher. note: The atomicity of the
// __half2 add operation is guaranteed separately for each of the two __half
// elements; the entire __half2 is not guaranteed to be atomic as a single
// 32-bit access.
GKO_BIND_ATOMIC_ADD(__half2);
#endif  // !((defined(CUDA_VERSION) && (CUDA_VERSION < 10000)) ||
        // (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)))


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
