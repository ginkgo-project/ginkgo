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

#include "core/distributed/preconditioner/bddc_kernels.hpp"


namespace gko {
namespace kernels {
namespace hip {
namespace distributed_bddc {


template <typename ValueType, typename IndexType>
void restrict_residual1(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Dense<ValueType>* global_residual,
    const array<IndexType>& local_to_local,
    const array<IndexType>& local_to_send_buffer,
    const matrix::Diagonal<ValueType>* weights, array<ValueType>& send_buffer,
    matrix::Dense<ValueType>* local_residual) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RESTRICT_RESIDUAL1);


template <typename ValueType, typename IndexType>
void restrict_residual2(
    std::shared_ptr<const DefaultExecutor> exec,
    const array<IndexType>& non_local_to_local,
    const array<IndexType>& global_to_recv_buffer,
    const array<IndexType>& non_local_idxs, const array<ValueType>& recv_buffer,
    matrix::Dense<ValueType>* local_residual) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RESTRICT_RESIDUAL2);


template <typename ValueType, typename IndexType>
void coarsen_residual1(std::shared_ptr<const DefaultExecutor> exec,
                       const array<IndexType>& coarse_local_to_local,
                       const array<IndexType>& coarse_local_to_send,
                       const matrix::Dense<ValueType>* local_coarse_residual,
                       array<ValueType>& coarse_send_buffer,
                       ValueType* coarse_residual,
                       ValueType* coarse_solution) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COARSEN_RESIDUAL1);


template <typename ValueType, typename IndexType>
void coarsen_residual2(std::shared_ptr<const DefaultExecutor> exec,
                       const array<IndexType>& coarse_recv_to_local,
                       const array<ValueType>& coarse_recv_buffer,
                       ValueType* coarse_residual) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COARSEN_RESIDUAL2);


template <typename ValueType, typename IndexType>
void prolong_coarse_solution(
    std::shared_ptr<const DefaultExecutor> exec,
    const array<IndexType>& coarse_local_to_local,
    const matrix::Dense<ValueType>* coarse_solution_local,
    const array<IndexType>& coarse_non_local_to_local,
    const array<IndexType>& coarse_local_to_non_local,
    const matrix::Dense<ValueType>* non_local,
    matrix::Dense<ValueType>* local_intermediate) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PROLONG_COARSE_SOLUTION);


}  // namespace distributed_bddc
}  // namespace hip
}  // namespace kernels
}  // namespace gko
