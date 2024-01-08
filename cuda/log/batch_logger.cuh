// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_LOG_BATCH_LOGGER_CUH_
#define GKO_CUDA_LOG_BATCH_LOGGER_CUH_


#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_log {


#include "common/cuda_hip/log/batch_logger.hpp.inc"


}  // namespace batch_log
}  // namespace cuda
}  // namespace kernels
}  // namespace gko

#endif  // GKO_CUDA_LOG_BATCH_LOGGER_CUH_
