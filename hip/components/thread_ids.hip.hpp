// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_COMPONENTS_THREAD_IDS_HIP_HPP_
#define GKO_HIP_COMPONENTS_THREAD_IDS_HIP_HPP_


#include "hip/base/config.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The HIP thread namespace.
 *
 * @ingroup hip_thread
 */
namespace thread {


#include "common/cuda_hip/components/thread_ids.hpp.inc"


}  // namespace thread
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_COMPONENTS_THREAD_IDS_HIP_HPP_
