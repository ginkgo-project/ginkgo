// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_COMPONENTS_DIAGONAL_BLOCK_MANIPULATION_HIP_HPP_
#define GKO_HIP_COMPONENTS_DIAGONAL_BLOCK_MANIPULATION_HIP_HPP_


#include <type_traits>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"


namespace gko {
namespace kernels {
namespace hip {
namespace csr {


#include "common/cuda_hip/components/diagonal_block_manipulation.hpp.inc"


}  // namespace csr
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_COMPONENTS_DIAGONAL_BLOCK_MANIPULATION_HIP_HPP_
