// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_STOP_BATCH_CRITERIA_HIP_HPP_
#define GKO_HIP_STOP_BATCH_CRITERIA_HIP_HPP_


#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace kernels {
namespace hip {
namespace batch_stop {


#include "common/cuda_hip/stop/batch_criteria.hpp.inc"


}  // namespace batch_stop
}  // namespace hip
}  // namespace kernels
}  // namespace gko

#endif  // GKO_HIP_STOP_BATCH_CRITERIA_HIP_HPP_
