/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_REFERENCE_LOG_BATCH_LOGGER_HPP_
#define GKO_REFERENCE_LOG_BATCH_LOGGER_HPP_


#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace host {
namespace batch_log {


/**
 * Logs the final residual and iteration count for a batch solver.
 *
 * Specialized for a single RHS.
 */
template <typename RealType>
class SimpleFinalLogger final {
public:
    using real_type = RealType;

    /**
     * Sets pre-allocated storage for logging.
     *
     * @param batch_residuals  Array of residuals norms of size
     *                         num_batches x num_rhs. Used as row major.
     * @param batch_iters  Array of final iteration counts for each
     *                     linear system and each RHS in the batch.
     */
    SimpleFinalLogger(real_type* const batch_residuals, int* const batch_iters)
        : final_residuals_{batch_residuals}, final_iters_{batch_iters}
    {}

    /**
     * Logs the iteration count and residual norm.
     *
     * @param batch_idx  The index of linear system in the batch to log.
     * @param iter  The current iteration count (0-based).
     * @param res_norm  Norm of current residual
     */
    void log_iteration(const size_type batch_idx, const int iter,
                       const real_type res_norm)
    {
        final_iters_[batch_idx] = iter;
        final_residuals_[batch_idx] = res_norm;
    }

private:
    real_type* const final_residuals_;
    int* const final_iters_;
};


}  // namespace batch_log
}  // namespace host
}  // namespace kernels
}  // namespace gko

#endif  // GKO_REFERENCE_LOG_BATCH_LOGGER_HPP_
