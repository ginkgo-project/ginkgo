// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_COMPONENTS_WARP_BLAS_HIP_HPP_
#define GKO_HIP_COMPONENTS_WARP_BLAS_HIP_HPP_


#include <cassert>
#include <type_traits>


#include <ginkgo/config.hpp>


#include "hip/base/math.hip.hpp"
#include "hip/components/reduction.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


#include "common/cuda_hip/components/warp_blas.hpp.inc"


}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_COMPONENTS_WARP_BLAS_HIP_HPP_
