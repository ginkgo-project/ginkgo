// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_COMPONENTS_MEMORY_HIP_HPP_
#define GKO_HIP_COMPONENTS_MEMORY_HIP_HPP_


#include <type_traits>


#include <ginkgo/core/base/math.hpp>


#include "hip/base/types.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


#include "common/cuda_hip/components/memory.hpp.inc"


template <typename ValueType>
__device__ __forceinline__ ValueType load_relaxed(const ValueType* ptr)
{
    return load(ptr, 0);
}


template <typename ValueType>
__device__ __forceinline__ ValueType load_acquire(const ValueType* ptr)
{
    auto result = load(ptr, 0);
    __threadfence();
    return result;
}

template <typename ValueType>
__device__ __forceinline__ void store_relaxed(ValueType* ptr, ValueType value)
{
    store(ptr, 0, value);
}


template <typename ValueType>
__device__ __forceinline__ void store_release(ValueType* ptr, ValueType value)
{
    __threadfence();
    store(ptr, 0, value);
}


template <typename ValueType>
__device__ __forceinline__ ValueType load_relaxed_shared(const ValueType* ptr)
{
    return load(ptr, 0);
}


template <typename ValueType>
__device__ __forceinline__ ValueType load_acquire_shared(const ValueType* ptr)
{
    auto result = load(ptr, 0);
    __threadfence();
    return result;
}

template <typename ValueType>
__device__ __forceinline__ void store_relaxed_shared(ValueType* ptr,
                                                     ValueType value)
{
    store(ptr, 0, value);
}


template <typename ValueType>
__device__ __forceinline__ void store_release_shared(ValueType* ptr,
                                                     ValueType value)
{
    __threadfence();
    store(ptr, 0, value);
}


}  // namespace hip
}  // namespace kernels
}  // namespace gko

#endif  // GKO_HIP_COMPONENTS_MEMORY_HIP_HPP_
