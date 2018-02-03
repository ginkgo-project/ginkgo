/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_GPU_COMPONENTS_SHUFLE_CUH_
#define GKO_GPU_COMPONENTS_SHUFLE_CUH_


#include "gpu/base/types.hpp"


#include <cassert>
#include <cstddef>


namespace gko {
namespace kernels {
namespace gpu {
namespace warp {


#if __CUDACC_VER_MAJOR__ < 9


template <typename T>
__device__ __forceinline__ T shuffle(T var, int32 src_lane,
                                     int32 width = warp_size,
                                     uint32 mask = full_lane_mask)
{
    static_assert(sizeof(T) % sizeof(int32) == 0,
                  "Unable to shuffle sizes which are not 4-byte multiples");
    assert(mask == full_lane_mask);
    constexpr auto size = sizeof(T) / sizeof(int32);
    T result;
    auto var_array = reinterpret_cast<int32 *>(&var);
    auto var_result = reinterpret_cast<int32 *>(&result);
#pragma unroll
    for (std::size_t i = 0; i < size; ++i) {
        var_result[i] = __shfl(var_array[i], src_lane, width);
    }
    return result;
}


template <typename T>
__device__ __forceinline__ T shuffle_up(T var, uint32 delta,
                                        int32 width = warp_size,
                                        uint32 mask = full_lane_mask)
{
    static_assert(sizeof(T) % sizeof(int32) == 0,
                  "Unable to shuffle sizes which are not 4-byte multiples");
    assert(mask == full_lane_mask);
    constexpr auto size = sizeof(T) / sizeof(int32);
    T result;
    auto var_array = reinterpret_cast<int32 *>(&var);
    auto var_result = reinterpret_cast<int32 *>(&result);
#pragma unroll
    for (std::size_t i = 0; i < size; ++i) {
        var_result[i] = __shfl_up(var_array[i], delta, width);
    }
    return result;
}


template <typename T>
__device__ __forceinline__ T shuffle_down(T var, uint32 delta,
                                          int32 width = warp_size,
                                          uint32 mask = full_lane_mask)
{
    static_assert(sizeof(T) % sizeof(int32) == 0,
                  "Unable to shuffle sizes which are not 4-byte multiples");
    assert(mask == full_lane_mask);
    constexpr auto size = sizeof(T) / sizeof(int32);
    T result;
    auto var_array = reinterpret_cast<int32 *>(&var);
    auto var_result = reinterpret_cast<int32 *>(&result);
#pragma unroll
    for (std::size_t i = 0; i < size; ++i) {
        var_result[i] = __shfl_down(var_array[i], delta, width);
    }
    return result;
}


template <typename T>
__device__ __forceinline__ T shuffle_xor(T var, int32 lane_mask,
                                         int32 width = warp_size,
                                         uint32 mask = full_lane_mask)
{
    static_assert(sizeof(T) % sizeof(int32) == 0,
                  "Unable to shuffle sizes which are not 4-byte multiples");
    assert(mask == full_lane_mask);
    constexpr auto size = sizeof(T) / sizeof(int32);
    T result;
    auto var_array = reinterpret_cast<int32 *>(&var);
    auto var_result = reinterpret_cast<int32 *>(&result);
#pragma unroll
    for (std::size_t i = 0; i < size; ++i) {
        var_result[i] = __shfl_xor(var_array[i], lane_mask, width);
    }
    return result;
}


#else  // __CUDACC_VER_MAJOR__ < 9


template <typename T>
__device__ __forceinline__ T shuffle(T var, int32 src_lane,
                                     int32 width = warp_size,
                                     uint32 mask = full_lane_mask)
{
    static_assert(sizeof(T) % sizeof(int32) == 0,
                  "Unable to shuffle sizes which are not 4-byte multiples");
    constexpr auto size = sizeof(T) / sizeof(int32);
    T result;
    auto var_array = reinterpret_cast<int32 *>(&var);
    auto var_result = reinterpret_cast<int32 *>(&result);
#pragma unroll
    for (std::size_t i = 0; i < size; ++i) {
        var_result[i] = __shfl_sync(mask, var_array[i], src_lane, width);
    }
    return result;
}


template <typename T>
__device__ __forceinline__ T shuffle_up(T var, uint32 delta,
                                        int32 width = warp_size,
                                        uint32 mask = full_lane_mask)
{
    static_assert(sizeof(T) % sizeof(int32) == 0,
                  "Unable to shuffle sizes which are not 4-byte multiples");
    constexpr auto size = sizeof(T) / sizeof(int32);
    T result;
    auto var_array = reinterpret_cast<int32 *>(&var);
    auto var_result = reinterpret_cast<int32 *>(&result);
#pragma unroll
    for (std::size_t i = 0; i < size; ++i) {
        var_result[i] = __shfl_up_sync(mask, var_array[i], delta, width);
    }
    return result;
}


template <typename T>
__device__ __forceinline__ T shuffle_down(T var, uint32 delta,
                                          int32 width = warp_size,
                                          uint32 mask = full_lane_mask)
{
    static_assert(sizeof(T) % sizeof(int32) == 0,
                  "Unable to shuffle sizes which are not 4-byte multiples");
    constexpr auto size = sizeof(T) / sizeof(int32);
    T result;
    auto var_array = reinterpret_cast<int32 *>(&var);
    auto var_result = reinterpret_cast<int32 *>(&result);
#pragma unroll
    for (std::size_t i = 0; i < size; ++i) {
        var_result[i] = __shfl_down_sync(mask, var_array[i], delta, width);
    }
    return result;
}


template <typename T>
__device__ __forceinline__ T shuffle_xor(T var, int32 lane_mask,
                                         int32 width = warp_size,
                                         uint32 mask = full_lane_mask)
{
    static_assert(sizeof(T) % sizeof(int32) == 0,
                  "Unable to shuffle sizes which are not 4-byte multiples");
    constexpr auto size = sizeof(T) / sizeof(int32);
    T result;
    auto var_array = reinterpret_cast<int32 *>(&var);
    auto var_result = reinterpret_cast<int32 *>(&result);
#pragma unroll
    for (std::size_t i = 0; i < size; ++i) {
        var_result[i] = __shfl_xor_sync(mask, var_array[i], lane_mask, width);
    }
    return result;
}


#endif  // __CUDACC_VER_MAJOR__ < 9


}  // namespace warp
}  // namespace gpu
}  // namespace kernels
}  // namespace gko


#endif  // GKO_GPU_COMPONENTS_SHUFFLE_CUH_
