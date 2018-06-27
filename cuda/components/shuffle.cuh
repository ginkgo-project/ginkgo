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

#ifndef GKO_CUDA_COMPONENTS_SHUFLE_CUH_
#define GKO_CUDA_COMPONENTS_SHUFLE_CUH_


#include "cuda/base/types.hpp"


#include <cassert>
#include <cstddef>


namespace gko {
namespace kernels {
namespace cuda {
namespace warp {
namespace detail {


template <typename ShuffleOperator, typename ValueType, typename SelectorType>
__device__ __forceinline__ ValueType shuffle_impl(ShuffleOperator shuffle,
                                                  const ValueType &var,
                                                  SelectorType selector,
                                                  int32 width, int32 mask)
{
    static_assert(sizeof(ValueType) % sizeof(int32) == 0,
                  "Unable to shuffle sizes which are not 4-byte multiples");
    constexpr auto value_size = sizeof(ValueType) / sizeof(int32);
    ValueType result;
    auto var_array = reinterpret_cast<const int32 *>(&var);
    auto result_array = reinterpret_cast<int32 *>(&result);
#pragma unroll
    for (std::size_t i = 0; i < value_size; ++i) {
        result_array[i] = shuffle(mask, var_array[i], selector, width);
    }
    return result;
}


}  // namespace detail


#if __CUDACC_VER_MAJOR__ < 9


#define GKO_ENABLE_SHUFFLE_OPERATION(_name, _intrinsic, SelectorType) \
    template <typename ValueType>                                     \
    __device__ __forceinline__ ValueType _name(                       \
        const ValueType &var, SelectorType selector,                  \
        int32 width = cuda_config::warp_size,                         \
        uint32 mask = cuda_config::full_lane_mask)                    \
    {                                                                 \
        GKO_ASSERT(mask == cuda_config::full_lane_mask);              \
        return detail::shuffle_impl(                                  \
            [](uint32 m, int32 v, SelectorType s, int32 w) {          \
                return _intrinsic(v, s, w);                           \
            },                                                        \
            var, selector, width, mask);                              \
    }

GKO_ENABLE_SHUFFLE_OPERATION(shuffle, __shfl, int32);
GKO_ENABLE_SHUFFLE_OPERATION(shuffle_up, __shfl_up, uint32);
GKO_ENABLE_SHUFFLE_OPERATION(shuffle_down, __shfl_down, uint32);
GKO_ENABLE_SHUFFLE_OPERATION(shuffle_xor, __shfl_xor, int32);

#undef GKO_ENABLE_SHUFFLE_OPERATION


#else  // __CUDACC_VER_MAJOR__ < 9


#define GKO_ENABLE_SHUFFLE_OPERATION(_name, _intrinsic, SelectorType)   \
    template <typename ValueType>                                       \
    __device__ __forceinline__ ValueType _name(                         \
        const ValueType &var, SelectorType selector,                    \
        int32 width = cuda_config::warp_size,                           \
        uint32 mask = cuda_config::full_lane_mask)                      \
    {                                                                   \
        return detail::shuffle_impl(                                    \
            static_cast<int32 (*)(uint32, int32, SelectorType, int32)>( \
                _intrinsic),                                            \
            var, selector, width, mask);                                \
    }

GKO_ENABLE_SHUFFLE_OPERATION(shuffle, __shfl_sync, int32);
GKO_ENABLE_SHUFFLE_OPERATION(shuffle_up, __shfl_up_sync, uint32);
GKO_ENABLE_SHUFFLE_OPERATION(shuffle_down, __shfl_down_sync, uint32);
GKO_ENABLE_SHUFFLE_OPERATION(shuffle_xor, __shfl_xor_sync, int32);

#undef GKO_ENABLE_SHUFFLE_OPERATION


#endif  // __CUDACC_VER_MAJOR__ < 9


}  // namespace warp
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_SHUFFLE_CUH_
