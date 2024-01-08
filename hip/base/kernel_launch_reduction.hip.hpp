// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_REDUCTION_HPP_
#error \
    "This file can only be used from inside common/unified/base/kernel_launch_reduction.hpp"
#endif


#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


#include "common/cuda_hip/base/kernel_launch_reduction.hpp.inc"


}  // namespace hip
}  // namespace kernels
}  // namespace gko
