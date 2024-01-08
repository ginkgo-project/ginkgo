// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_COMPONENTS_SORTING_CUH_
#define GKO_CUDA_COMPONENTS_SORTING_CUH_


#include "cuda/base/config.hpp"
#include "cuda/components/cooperative_groups.cuh"


namespace gko {
namespace kernels {
namespace cuda {


#include "common/cuda_hip/components/sorting.hpp.inc"


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_SORTING_CUH_
