// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_COMPONENTS_SEGMENT_SCAN_HIP_HPP_
#define GKO_HIP_COMPONENTS_SEGMENT_SCAN_HIP_HPP_


#include "hip/components/cooperative_groups.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


#include "common/cuda_hip/components/segment_scan.hpp.inc"


}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_COMPONENTS_SEGMENT_SCAN_HIP_HPP_
