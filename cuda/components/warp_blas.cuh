// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_COMPONENTS_WARP_BLAS_CUH_
#define GKO_CUDA_COMPONENTS_WARP_BLAS_CUH_


#include <cassert>
#include <type_traits>


#include <ginkgo/config.hpp>


#include "cuda/base/math.hpp"
#include "cuda/components/reduction.cuh"


namespace gko {
namespace kernels {
namespace cuda {


#include "common/cuda_hip/components/warp_blas.hpp.inc"


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_WARP_BLAS_CUH_
