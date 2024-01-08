// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_COMPONENTS_SYNCFREE_HIP_HPP_
#define GKO_HIP_COMPONENTS_SYNCFREE_HIP_HPP_


#include <ginkgo/core/base/array.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/memory.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


#include "common/cuda_hip/components/syncfree.hpp.inc"


}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_COMPONENTS_SYNCFREE_HIP_HPP_
