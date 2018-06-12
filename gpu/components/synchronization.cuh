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

#ifndef GKO_GPU_COMPONENTS_SYNCHRONIZATION_CUH_
#define GKO_GPU_COMPONENTS_SYNCHRONIZATION_CUH_


#include "gpu/base/types.hpp"


#include <cassert>


namespace gko {
namespace kernels {
namespace gpu {
namespace warp {


#if __CUDACC_VER_MAJOR__ < 9


__device__ __forceinline__ uint32 active_mask()
{
    // all threads are always active in CUDA < 9
    return cuda_config::full_lane_mask;
}


__device__ __forceinline__ void synchronize(
    uint32 mask = cuda_config::full_lane_mask)
{
    // warp is implicitly synchronized, only need to enforce memory ordering
    __threadfence_block();
}


__device__ __forceinline__ bool any(bool predicate,
                                    uint32 mask = cuda_config::full_lane_mask)
{
    GKO_ASSERT(mask == cuda_config::full_lane_mask);
    return __any(predicate);
}


__device__ __forceinline__ bool all(bool predicate,
                                    uint32 mask = cuda_config::full_lane_mask)
{
    GKO_ASSERT(mask == cuda_config::full_lane_mask);
    return __all(predicate);
}


__device__ __forceinline__ int32
count(bool predicate, uint32 mask = cuda_config::full_lane_mask)
{
    GKO_ASSERT(mask == cuda_config::full_lane_mask);
    return __popc(__ballot(predicate));
}


__device__ __forceinline__ uint32
ballot(bool predicate, uint32 mask = cuda_config::full_lane_mask)
{
    GKO_ASSERT(mask == cuda_config::full_lane_mask);
    return __ballot(predicate);
}


#else  // __CUDACC_VER_MAJOR__ < 9


__device__ __forceinline__ bool active_mask() { return __activemask(); }


__device__ __forceinline__ void synchronize(
    uint32 mask = cuda_config::full_lane_mask)
{
    __syncwarp(mask);
}


__device__ __forceinline__ bool any(bool predicate,
                                    uint32 mask = cuda_config::full_lane_mask)
{
    return __any_sync(mask, predicate);
}


__device__ __forceinline__ bool all(bool predicate,
                                    uint32 mask = cuda_config::full_lane_mask)
{
    return __all_sync(mask, predicate);
}


__device__ __forceinline__ int32
count(bool predicate, uint32 mask = cuda_config::full_lane_mask)
{
    return __popc(__ballot_sync(mask, predicate));
}


__device__ __forceinline__ uint32
ballot(bool predicate, uint32 mask = cuda_config::full_lane_mask)
{
    return __ballot_sync(mask, predicate);
}


#endif  // __CUDACC_VER_MAJOR__ < 9


}  // namespace warp


namespace block {


__device__ __forceinline__ void fence() { __threadfence_block(); }


__device__ __forceinline__ void synchronize() { __syncthreads(); }


__device__ __forceinline__ bool any(bool predicate)
{
    return __syncthreads_or(predicate);
}


__device__ __forceinline__ bool all(bool predicate)
{
    return __syncthreads_and(predicate);
}


__device__ __forceinline__ int32 count(bool predicate)
{
    return __syncthreads_count(predicate);
}


}  // namespace block


namespace device {


__device__ __forceinline__ void fence() { __threadfence(); }


}  // namespace device


namespace system {


__device__ __forceinline__ void fence() { __threadfence_system(); }


}  // namespace system
}  // namespace gpu
}  // namespace kernels
}  // namespace gko


#endif  // GKO_GPU_COMPONENTS_SYNCHRONIZATION_CUH_
