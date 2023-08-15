// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_COMPONENTS_VOLATILE_CUH_
#define GKO_CUDA_COMPONENTS_VOLATILE_CUH_


#include <type_traits>


#include <ginkgo/core/base/math.hpp>


#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {


#include "common/cuda_hip/components/volatile.hpp.inc"


}  // namespace cuda
}  // namespace kernels
}  // namespace gko

#endif  // GKO_CUDA_COMPONENTS_VOLATILE_CUH_
