// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_LOG_BATCH_LOGGER_HPP_
#define GKO_COMMON_CUDA_HIP_LOG_BATCH_LOGGER_HPP_


#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace batch_log {

/**
 * @see reference/log/batch_logger.hpp
 */
template <typename RealType>
class SimpleFinalLogger final {
public:
    using real_type = RealType;
    using idx_type = int;

    SimpleFinalLogger(real_type* const batch_residuals,
                      idx_type* const batch_iters)
        : final_residuals_{batch_residuals}, final_iters_{batch_iters}
    {}

    __device__ __forceinline__ void log_iteration(const size_type batch_idx,
                                                  const int iter,
                                                  const real_type res_norm)
    {
        final_iters_[batch_idx] = iter;
        final_residuals_[batch_idx] = res_norm;
    }

private:
    real_type* const final_residuals_;
    idx_type* const final_iters_;
};


}  // namespace batch_log
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif
