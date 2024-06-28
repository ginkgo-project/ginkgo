// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_COMPONENTS_DIAGONAL_BLOCK_MANIPULATION_CUH_
#define GKO_CUDA_COMPONENTS_DIAGONAL_BLOCK_MANIPULATION_CUH_


#include <type_traits>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace csr {


#include "common/cuda_hip/components/diagonal_block_manipulation.hpp.inc"


}  // namespace csr
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_DIAGONAL_BLOCK_MANIPULATION_CUH_
