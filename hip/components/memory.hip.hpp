// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_COMPONENTS_MEMORY_HIP_HPP_
#define GKO_HIP_COMPONENTS_MEMORY_HIP_HPP_


#include <cstring>
#include <type_traits>


#include <ginkgo/core/base/math.hpp>


#include "hip/base/types.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


#if GINKGO_HIP_PLATFORM_NVCC


#include "common/cuda_hip/components/memory.nvidia.hpp.inc"


#else


/**
 * Used to map primitive types to an equivalently-sized/aligned type that can be
 * used in atomic intrinsics.
 */
template <typename T>
struct gcc_atomic_intrinsic_type_map {};


template <>
struct gcc_atomic_intrinsic_type_map<int32> {
    using type = int32;
};


template <>
struct gcc_atomic_intrinsic_type_map<float> {
    using type = int32;
};


template <>
struct gcc_atomic_intrinsic_type_map<int64> {
    using type = int64;
};


template <>
struct gcc_atomic_intrinsic_type_map<double> {
    using type = int64;
};


#if HIP_VERSION >= 50100000
// These intrinsics can be found used in clang/test/SemaCUDA/atomic-ops.cu
// in the LLVM source code

#define HIP_ATOMIC_LOAD(ptr, memorder, scope) \
    __hip_atomic_load(ptr, memorder, scope)
#define HIP_ATOMIC_STORE(ptr, value, memorder, scope) \
    __hip_atomic_store(ptr, value, memorder, scope)
#define HIP_SCOPE_GPU __HIP_MEMORY_SCOPE_AGENT
#define HIP_SCOPE_THREADBLOCK __HIP_MEMORY_SCOPE_WORKGROUP
#else
#define HIP_ATOMIC_LOAD(ptr, memorder, scope) __atomic_load_n(ptr, memorder)
#define HIP_ATOMIC_STORE(ptr, value, memorder, scope) \
    __atomic_store_n(ptr, value, memorder)
#define HIP_SCOPE_GPU -1
#define HIP_SCOPE_THREADBLOCK -1
#endif


/**
 * Loads a value from memory using an atomic operation.
 *
 * @tparam memorder  The GCC memory ordering type
 * (https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html) to use
 * for this atomic operation.
 * @tparam scope  The visibility of this operation, i.e. which threads may have
 * written to this memory location before. HIP_SCOPE_GPU means that we want to
 * observe writes from all threads on this device, HIP_SCOPE_THREADBLOCK means
 * we want to observe only writes from within the same threadblock.
 */
template <int memorder, int scope, typename ValueType>
__device__ __forceinline__ ValueType load_generic(const ValueType* ptr)
{
    using atomic_type = typename gcc_atomic_intrinsic_type_map<ValueType>::type;
    static_assert(sizeof(atomic_type) == sizeof(ValueType), "invalid map");
    static_assert(alignof(atomic_type) == alignof(ValueType), "invalid map");
    auto cast_value = HIP_ATOMIC_LOAD(reinterpret_cast<const atomic_type*>(ptr),
                                      memorder, scope);
    ValueType result{};
    std::memcpy(&result, &cast_value, sizeof(ValueType));
    return result;
}


/**
 * Stores a value to memory using an atomic operation.
 *
 * @tparam memorder  The GCC memory ordering type
 * (https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html) to use
 * for this atomic operation.
 * @tparam scope  The visibility of this operation, i.e. which threads may
 * observe the write to this memory location. HIP_SCOPE_GPU means that we want
 * to all threads on this device to observe it, HIP_SCOPE_THREADBLOCK means we
 * want only threads within the same threadblock to observe it.
 */
template <int memorder, int scope, typename ValueType>
__device__ __forceinline__ void store_generic(ValueType* ptr, ValueType value)
{
    using atomic_type = typename gcc_atomic_intrinsic_type_map<ValueType>::type;
    static_assert(sizeof(atomic_type) == sizeof(ValueType), "invalid map");
    static_assert(alignof(atomic_type) == alignof(ValueType), "invalid map");
    atomic_type cast_value{};
    std::memcpy(&cast_value, &value, sizeof(ValueType));
    HIP_ATOMIC_STORE(reinterpret_cast<atomic_type*>(ptr), cast_value, memorder,
                     scope);
}


template <typename ValueType>
__device__ __forceinline__ ValueType load_relaxed(const ValueType* ptr)
{
    return load_generic<__ATOMIC_RELAXED, HIP_SCOPE_GPU>(ptr);
}


template <typename ValueType>
__device__ __forceinline__ ValueType load_relaxed_shared(const ValueType* ptr)
{
    return load_generic<__ATOMIC_RELAXED, HIP_SCOPE_THREADBLOCK>(ptr);
}


template <typename ValueType>
__device__ __forceinline__ ValueType load_relaxed_local(const ValueType* ptr)
{
    return load_generic<__ATOMIC_RELAXED, HIP_SCOPE_THREADBLOCK>(ptr);
}


template <typename ValueType>
__device__ __forceinline__ ValueType load_acquire(const ValueType* ptr)
{
    return load_generic<__ATOMIC_ACQUIRE, HIP_SCOPE_GPU>(ptr);
}


template <typename ValueType>
__device__ __forceinline__ ValueType load_acquire_shared(const ValueType* ptr)
{
    return load_generic<__ATOMIC_ACQUIRE, HIP_SCOPE_THREADBLOCK>(ptr);
}


template <typename ValueType>
__device__ __forceinline__ ValueType load_acquire_local(const ValueType* ptr)
{
    return load_generic<__ATOMIC_ACQUIRE, HIP_SCOPE_THREADBLOCK>(ptr);
}


template <typename ValueType>
__device__ __forceinline__ void store_relaxed(ValueType* ptr, ValueType value)
{
    store_generic<__ATOMIC_RELAXED, HIP_SCOPE_GPU>(ptr, value);
}


template <typename ValueType>
__device__ __forceinline__ void store_relaxed_shared(ValueType* ptr,
                                                     ValueType value)
{
    store_generic<__ATOMIC_RELAXED, HIP_SCOPE_THREADBLOCK>(ptr, value);
}


template <typename ValueType>
__device__ __forceinline__ void store_relaxed_local(ValueType* ptr,
                                                    ValueType value)
{
    store_generic<__ATOMIC_RELAXED, HIP_SCOPE_THREADBLOCK>(ptr, value);
}


template <typename ValueType>
__device__ __forceinline__ void store_release(ValueType* ptr, ValueType value)
{
    store_generic<__ATOMIC_RELEASE, HIP_SCOPE_GPU>(ptr, value);
}


template <typename ValueType>
__device__ __forceinline__ void store_release_shared(ValueType* ptr,
                                                     ValueType value)
{
    store_generic<__ATOMIC_RELEASE, HIP_SCOPE_THREADBLOCK>(ptr, value);
}


template <typename ValueType>
__device__ __forceinline__ void store_release_local(ValueType* ptr,
                                                    ValueType value)
{
    store_generic<__ATOMIC_RELEASE, HIP_SCOPE_THREADBLOCK>(ptr, value);
}


template <typename ValueType>
__device__ __forceinline__ thrust::complex<ValueType> load_relaxed(
    const thrust::complex<ValueType>* ptr)
{
    auto real_ptr = reinterpret_cast<const ValueType*>(ptr);
    auto real = load_relaxed(real_ptr);
    auto imag = load_relaxed(real_ptr + 1);
    return {real, imag};
}


template <typename ValueType>
__device__ __forceinline__ thrust::complex<ValueType> load_relaxed_shared(
    const thrust::complex<ValueType>* ptr)
{
    auto real_ptr = reinterpret_cast<const ValueType*>(ptr);
    auto real = load_relaxed_shared(real_ptr);
    auto imag = load_relaxed_shared(real_ptr + 1);
    return {real, imag};
}


template <typename ValueType>
__device__ __forceinline__ thrust::complex<ValueType> load_relaxed_local(
    const thrust::complex<ValueType>* ptr)
{
    auto real_ptr = reinterpret_cast<const ValueType*>(ptr);
    auto real = load_relaxed_local(real_ptr);
    auto imag = load_relaxed_local(real_ptr + 1);
    return {real, imag};
}


template <typename ValueType>
__device__ __forceinline__ void store_relaxed(thrust::complex<ValueType>* ptr,
                                              thrust::complex<ValueType> value)
{
    auto real_ptr = reinterpret_cast<ValueType*>(ptr);
    store_relaxed(real_ptr, value.real());
    store_relaxed(real_ptr + 1, value.imag());
}


template <typename ValueType>
__device__ __forceinline__ void store_relaxed_shared(
    thrust::complex<ValueType>* ptr, thrust::complex<ValueType> value)
{
    auto real_ptr = reinterpret_cast<ValueType*>(ptr);
    store_relaxed_shared(real_ptr, value.real());
    store_relaxed_shared(real_ptr + 1, value.imag());
}


template <typename ValueType>
__device__ __forceinline__ void store_relaxed_local(
    thrust::complex<ValueType>* ptr, thrust::complex<ValueType> value)
{
    auto real_ptr = reinterpret_cast<ValueType*>(ptr);
    store_relaxed_local(real_ptr, value.real());
    store_relaxed_local(real_ptr + 1, value.imag());
}


#undef HIP_ATOMIC_LOAD
#undef HIP_ATOMIC_STORE
#undef HIP_SCOPE_GPU
#undef HIP_SCOPE_THREADBLOCK


#endif  // !GINKGO_HIP_PLATFORM_NVCC


}  // namespace hip
}  // namespace kernels
}  // namespace gko

#endif  // GKO_HIP_COMPONENTS_MEMORY_HIP_HPP_
