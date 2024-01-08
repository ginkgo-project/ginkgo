// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_LOG_BATCH_LOGGER_HPP_
#define GKO_DPCPP_LOG_BATCH_LOGGER_HPP_


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/types.hpp>


#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_log {


/**
 * @see reference/log/batch_logger.hpp
 */
template <typename RealType>
class SimpleFinalLogger final {
public:
    using real_type = remove_complex<RealType>;

    SimpleFinalLogger(real_type* const batch_residuals, int* const batch_iters)
        : final_residuals_{batch_residuals}, final_iters_{batch_iters}
    {}

    __dpct_inline__ void log_iteration(const size_type batch_idx,
                                       const int iter, const real_type res_norm)
    {
        final_iters_[batch_idx] = iter;
        final_residuals_[batch_idx] = res_norm;
    }

private:
    real_type* const final_residuals_;
    int* const final_iters_;
};


}  // namespace batch_log
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko

#endif  // GKO_DPCPP_LOG_BATCH_LOGGER_HPP_
