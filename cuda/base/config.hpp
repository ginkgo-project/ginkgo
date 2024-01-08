// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_BASE_CONFIG_HPP_
#define GKO_CUDA_BASE_CONFIG_HPP_


#include <ginkgo/core/base/types.hpp>


#include "cuda/base/math.hpp"


namespace gko {
namespace kernels {
namespace cuda {


struct config {
    /**
     * The type containing a bitmask over all lanes of a warp.
     */
    using lane_mask_type = uint32;

    /**
     * The number of threads within a CUDA warp.
     */
    static constexpr uint32 warp_size = 32;

    /**
     * The bitmask of the entire warp.
     */
    static constexpr auto full_lane_mask = ~zero<lane_mask_type>();

    /**
     * The maximal number of threads allowed in a CUDA warp.
     */
    static constexpr uint32 max_block_size = 1024;

    /**
     * The minimal amount of warps that need to be scheduled for each block
     * to maximize GPU occupancy.
     */
    static constexpr uint32 min_warps_per_block = 4;
};


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_CONFIG_HPP_
