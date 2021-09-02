/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


namespace batch_log {


/**
 * Simple logger for final residuals and iteration counts of all
 * linear systems in a batch.
 */
template <typename RealType>
class FinalLogger final {
public:
    using real_type = RealType;

    /**
     * Sets pre-allocated storage for logging.
     *
     * @param num_rhs  The number of RHS vectors.
     * @param max_iters  The maximum iterations allowed.
     * @param batch_residuals  Array of residuals norms of size
     *                         num_batches x num_rhs. Used as row major.
     * @param batch_iters  Array of final iteration counts for each
     *                     linear system and each RHS in the batch.
     */
    FinalLogger(const int num_rhs, const int max_iters,
                real_type* const batch_residuals, int* const batch_iters)
        : nrhs_{num_rhs},
          max_iters_{max_iters},
          final_residuals_{batch_residuals},
          final_iters_{batch_iters},
          init_converged_(0 - (1 << num_rhs))
    {}

    /**
     * Logs an iteration of the solver, though does nothing for most iterations.
     * This MUST be called at the zeroth iteration of the solver for every
     * batch entry.
     *
     * Initially at iteration 0, the stored converged bitset is reset to start
     * a new solve, and the final iteration counts array is initialized.
     * Records the residual for all RHSs whenever the convergence bitset
     * changes state. Further, records the iteration count whenever the
     * convergence state of a specific RHS changes for the better.
     * Finally, records residuals for all RHS in the final iteration.
     *
     * @param batch_idx  The index of linear system in the batch to log.
     * @param iter  The current iteration count (0-based).
     * @param res_norm  Norms of current residuals for each RHS.
     *                  The jth thread accesses the jth RHS.
     * @param converged  Bitset representing convergence state for each RHS.
     */
    __device__ __forceinline__ void log_iteration(
        const size_type batch_idx, const int iter,
        const real_type* const res_norm, const uint32 converged)
    {
        if (iter == 0) {
            init_converged_ = 0 - (1 << nrhs_);
            for (int j = threadIdx.x; j < nrhs_; j += blockDim.x) {
                final_iters_[batch_idx * nrhs_ + j] = max_iters_ - 1;
            }
        }
        // Maybe this can/should be optimized? It might not be worth the effort.
        if (converged != init_converged_ || iter >= max_iters_ - 1) {
            for (int j = threadIdx.x; j < nrhs_; j += blockDim.x) {
                const uint32 jconv = converged & (1 << j);
                const uint32 old_jconv = init_converged_ & (1 << j);
                if (jconv && (old_jconv != jconv)) {
                    final_iters_[batch_idx * nrhs_ + j] = iter;
                }
                final_residuals_[batch_idx * nrhs_ + j] = res_norm[j];
            }

            init_converged_ = converged;
        }
    }

private:
    const int nrhs_;
    const int max_iters_;
    real_type* const final_residuals_;
    int* const final_iters_;
    uint32 init_converged_;
};


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
    __device__ __forceinline__ void log_iteration(const size_type batch_idx,
                                                  const int iter,
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
