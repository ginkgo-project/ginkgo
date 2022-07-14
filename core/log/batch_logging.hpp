/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_CORE_LOG_BATCH_LOGGING_HPP_
#define GKO_CORE_LOG_BATCH_LOGGING_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


namespace gko {
namespace log {


/**
 * Types of batch loggers available.
 */
enum class BatchLogType {
    convergence_completion,
    simple_convergence_completion
};


struct BatchLogDataBase {
    /**
     * Stores convergence iteration counts for every matrix in the batch and
     * for every right-hand side.
     */
    array<int> iter_counts;

    virtual ~BatchLogDataBase() = default;
};


/**
 * Stores logging data for batch solver kernels.
 */
template <typename ValueType>
struct BatchLogData : public BatchLogDataBase {
    /**
     * Stores residual norm values for every linear system in the batch
     * for every right-hand side.
     */
    std::shared_ptr<matrix::BatchDense<remove_complex<ValueType>>> res_norms;
};


}  // namespace log
}  // namespace gko

#endif  // GKO_CORE_LOG_BATCH_LOGGING_HPP_
