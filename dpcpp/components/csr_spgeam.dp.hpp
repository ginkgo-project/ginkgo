/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_DPCPP_COMPONENTS_CSR_SPGEAM_DP_HPP_
#define GKO_DPCPP_COMPONENTS_CSR_SPGEAM_DP_HPP_


#include <limits>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/utils.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {


/**
 * Adds two (sorted) sparse matrices.
 *
 * Calls begin_cb(row) on each row to initialize row-local data
 * Calls entry_cb(row, col, a_val, b_val, local_data) on each output non-zero
 * Calls end_cb(row, local_data) on each row to finalize row-local data
 *
 * If the three functions are thread-safe, the whole invocation is.
 */
template <typename ValueType, typename IndexType, typename BeginCallback,
          typename EntryCallback, typename EndCallback>
void abstract_spgeam(const matrix::Csr<ValueType, IndexType> *a,
                     const matrix::Csr<ValueType, IndexType> *b,
                     BeginCallback begin_cb, EntryCallback entry_cb,
                     EndCallback end_cb)
{
    GKO_NOT_IMPLEMENTED;
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_CSR_SPGEAM_DP_HPP_
