// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_COMPONENTS_MERGING_HIP_HPP_
#define GKO_HIP_COMPONENTS_MERGING_HIP_HPP_


#include "core/base/utils.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/components/intrinsics.hip.hpp"
#include "hip/components/searching.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


#include "common/cuda_hip/components/merging.hpp.inc"


}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_COMPONENTS_MERGING_HIP_HPP_
