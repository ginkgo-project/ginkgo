// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_COMPONENTS_MERGING_CUH_
#define GKO_CUDA_COMPONENTS_MERGING_CUH_


#include "core/base/utils.hpp"
#include "cuda/base/math.hpp"
#include "cuda/components/intrinsics.cuh"
#include "cuda/components/searching.cuh"


namespace gko {
namespace kernels {
namespace cuda {


#include "common/cuda_hip/components/merging.hpp.inc"


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_MERGING_CUH_
