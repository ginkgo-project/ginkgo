// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_COMPONENTS_MEMORY_CUH_
#define GKO_CUDA_COMPONENTS_MEMORY_CUH_


#include <type_traits>


#include <ginkgo/core/base/math.hpp>


#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {


#include "common/cuda_hip/components/memory.nvidia.hpp.inc"


}  // namespace cuda
}  // namespace kernels
}  // namespace gko

#endif  // GKO_CUDA_COMPONENTS_MEMORY_CUH_
