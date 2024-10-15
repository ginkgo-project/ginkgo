// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace batch_log {


/**
 * Logs the final residual norm and iteration count for a batch solver.
 *
 * @note Supports only a single RHS per batch item.
 */
template <typename RealType>
class SimpleFinalLogger final {
public:
    using real_type = RealType;
    using idx_type = int;

    /**
     * Constructor
     *
     * @param batch_residuals  residuals norms of size
     *                         num_batch_items.
     * @param batch_iters  final iteration counts for each
     *                     linear system in the batch.
     */
    constexpr SimpleFinalLogger(real_type* const batch_residuals,
                                idx_type* const batch_iters)
        : final_residuals_{batch_residuals}, final_iters_{batch_iters}
    {}

    /**
     * Logs the final iteration count and the final residual norm.
     *
     * @param batch_idx  The index of linear system in the batch to log.
     * @param iter  The final iteration count (0-based).
     * @param res_norm  Norm of final residual norm
     */
    constexpr void log_iteration(const size_type batch_idx, const int iter,
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
}  // namespace gko
