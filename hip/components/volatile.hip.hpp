// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_COMPONENTS_VOLATILE_HIP_HPP_
#define GKO_HIP_COMPONENTS_VOLATILE_HIP_HPP_


#include <type_traits>


#include <ginkgo/core/base/math.hpp>


#include "hip/base/types.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


#include "common/cuda_hip/components/volatile.hpp.inc"


}  // namespace hip
}  // namespace kernels
}  // namespace gko

#endif  // GKO_HIP_COMPONENTS_VOLATILE_HIP_HPP_
