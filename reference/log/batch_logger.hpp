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


#ifndef GKO_REFERENCE_LOG_BATCH_LOGGER_HPP_
#define GKO_REFERENCE_LOG_BATCH_LOGGER_HPP_


#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace reference {
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
     * @param batch_residuals  Array of residuals norms of size
     *                         num_batches x num_rhs. Used as row major.
     * @param batch_iters  Array of final iteration counts for each
     *                     linear system and each RHS in the batch.
     */
    FinalLogger(const int num_rhs, real_type *const batch_residuals,
                int *const batch_iters)
        : nrhs{num_rhs},
          final_residuals{batch_residuals},
          final_iters{batch_iters},
          init_converged(0 - (1 << num_rhs))
    {}

    void log_iteration(const size_type batch_idx, const int iter,
                       const real_type *const res_norm, const uint32 converged)
    {
        if (converged != init_converged) {
            for (int j = 0; j < nrhs; j++) {
                const uint32 jconv = converged & (1 << j);
                const uint32 old_jconv = init_converged & (1 << j);
                if (jconv && (old_jconv != jconv)) {
                    final_iters[batch_idx * nrhs + j] = iter;
                }
                final_residuals[batch_idx * nrhs + j] = res_norm[j];
            }

            init_converged = converged;
        }
    }

private:
    const int nrhs;
    real_type *const final_residuals;
    int *const final_iters;
    uint32 init_converged;
};


}  // namespace batch_log
}  // namespace reference
}  // namespace kernels
}  // namespace gko

#endif
