// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_COMPONENTS_ATOMIC_CUH_
#define GKO_CUDA_COMPONENTS_ATOMIC_CUH_


#include <type_traits>

#include "common/cuda_hip/base/types.hpp"
#include "cuda/base/math.hpp"


namespace gko {
namespace kernels {
namespace cuda {


#include "common/cuda_hip/components/atomic.hpp.inc"


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_ATOMIC_CUH_
