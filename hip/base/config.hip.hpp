/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_HIP_BASE_CONFIG_HIP_HPP_
#define GKO_HIP_BASE_CONFIG_HIP_HPP_


#include <hip/device_functions.h>


#include <ginkgo/core/base/types.hpp>


#include "hip/base/math.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


struct config {
    /**
     * The number of threads within a HIP warp. Here, we use the definition from
     * `device_functions.h`.
     */
#if GINKGO_HIP_PLATFORM_HCC
    static constexpr uint32 warp_size = warpSize;
#else  // GINKGO_HIP_PLATFORM_NVCC
    static constexpr uint32 warp_size = 32;
#endif

    /**
     * The bitmask of the entire warp.
     */
#if GINKGO_HIP_PLATFORM_HCC
    static constexpr uint64 full_lane_mask = ~zero<uint64>();
#else  // GINKGO_HIP_PLATFORM_NVCC
    static constexpr uint32 full_lane_mask = ~zero<uint32>();
#endif

    /**
     * The maximal number of threads allowed in a HIP warp.
     */
    static constexpr uint32 max_block_size = 1024;

    /**
     * The minimal amount of warps that need to be scheduled for each block
     * to maximize GPU occupancy.
     */
    static constexpr uint32 min_warps_per_block = 4;
};


}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_CONFIG_HIP_HPP_
