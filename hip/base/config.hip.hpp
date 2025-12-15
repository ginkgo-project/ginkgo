// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_BASE_CONFIG_HIP_HPP_
#define GKO_HIP_BASE_CONFIG_HIP_HPP_


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>

#include "common/cuda_hip/base/runtime.hpp"


namespace gko {
namespace kernels {
namespace hip {


struct config {
    /**
     * The type containing a bitmask over all lanes of a warp.
     */
#if GINKGO_HIP_PLATFORM_HCC
    using lane_mask_type = uint64;
#else  // GINKGO_HIP_PLATFORM_NVCC
    using lane_mask_type = uint32;
#endif

    /**
     * The number of threads within a HIP warp. Here, we use the definition from
     * `device_functions.h`.
     */
#if GINKGO_HIP_PLATFORM_HCC
    // workaround for ROCm >= 7, which does not give warpSize in compile time.
    // We can not define warpSize via compiler because amd_warp_functions.h
    // defines a struct variable called warpSize, too. No support for 32 on AMD
    // GPU yet.
    static constexpr uint32 warp_size = 64;
#else  // GINKGO_HIP_PLATFORM_NVCC
    static constexpr uint32 warp_size = 32;
#endif

    /**
     * The bitmask of the entire warp.
     */
    static constexpr auto full_lane_mask = ~zero<lane_mask_type>();

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
