// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_LOG_BATCH_LOGGER_HIP_HPP_
#define GKO_HIP_LOG_BATCH_LOGGER_HIP_HPP_


#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace hip {
namespace batch_log {

#include "common/cuda_hip/log/batch_logger.hpp.inc"


}  // namespace batch_log
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_LOG_BATCH_LOGGER_HIP_HPP_
