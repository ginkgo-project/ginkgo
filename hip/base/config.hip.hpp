// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_BASE_CONFIG_HIP_HPP_
#define GKO_HIP_BASE_CONFIG_HIP_HPP_


#include <hip/hip_runtime.h>


#include <hip/device_functions.h>


#include <ginkgo/core/base/types.hpp>


#include "hip/base/math.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


struct config {
    /**
     * The type containing a bitmask over all lanes of a warp.
     */
    using lane_mask_type = uint64;

    /**
     * The number of threads within a HIP warp. Here, we use the definition from
     * `device_functions.h`.
     */
    static constexpr uint32 warp_size = warpSize;

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
