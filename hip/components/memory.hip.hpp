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
__device__ __forceinline__ ValueType load_relaxed(ValueType* ptr)
{
    return load(ptr, 0);
}


template <typename ValueType>
__device__ __forceinline__ ValueType load_acquire(ValueType* ptr)
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
__device__ __forceinline__ ValueType load_relaxed_shared(ValueType* ptr)
{
    return load(ptr, 0);
}


template <typename ValueType>
__device__ __forceinline__ ValueType load_acquire_shared(ValueType* ptr)
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
