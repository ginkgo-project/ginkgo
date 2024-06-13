// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_COMPONENTS_THREAD_IDS_CUH_
#define GKO_CUDA_COMPONENTS_THREAD_IDS_CUH_


#include "cuda/base/config.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The CUDA thread namespace.
 *
 * @ingroup cuda_thread
 */
namespace thread {


#include "common/cuda_hip/components/thread_ids.hpp.inc"


}  // namespace thread
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_THREAD_IDS_CUH_
