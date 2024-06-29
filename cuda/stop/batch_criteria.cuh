// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_STOP_BATCH_CRITERIA_CUH_
#define GKO_CUDA_STOP_BATCH_CRITERIA_CUH_


#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_stop {


#include "common/cuda_hip/stop/batch_criteria.hpp.inc"


}  // namespace batch_stop
}  // namespace cuda
}  // namespace kernels
}  // namespace gko

#endif  // GKO_CUDA_STOP_BATCH_CRITERIA_CUH_
