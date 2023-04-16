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
namespace omp {
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


template <typename ValueType, typename IndexType>
void prolong_coarse_solution1(std::shared_ptr<const DefaultExecutor> exec,
                              const array<IndexType>& coarse_recv_to_local,
                              const matrix::Dense<ValueType>* coarse_solution,
                              array<ValueType>& coarse_recv_buffer)
    GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PROLONG_COARSE_SOLUTION1);


template <typename ValueType, typename IndexType>
void prolong_coarse_solution2(std::shared_ptr<const DefaultExecutor> exec,
                              const array<IndexType>& coarse_local_to_local,
                              const matrix::Dense<ValueType>* coarse_solution,
                              const array<IndexType>& coarse_local_to_send,
                              const array<ValueType>& coarse_send_buffer,
                              matrix::Dense<ValueType>* local_intermediate)
    GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PROLONG_COARSE_SOLUTION2);


template <typename ValueType, typename IndexType>
void finalize1(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::Dense<ValueType>* coarse_solution,
               const matrix::Diagonal<ValueType>* weights,
               const array<IndexType>& recv_to_local,
               const array<IndexType>& non_local_to_local,
               array<ValueType>& recv_buffer,
               matrix::Dense<ValueType>* local_solution) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FINALIZE1);


template <typename ValueType, typename IndexType>
void finalize2(std::shared_ptr<const DefaultExecutor> exec,
               const array<ValueType>& send_buffer,
               const array<IndexType>& local_to_send_buffer,
               const array<IndexType>& local_to_local,
               matrix::Dense<ValueType>* local_solution,
               ValueType* global_solution) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FINALIZE2);


template <typename ValueType, typename IndexType>
void static_condensation1(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::Dense<ValueType>* residual,
                          const array<IndexType>& inner_to_local,
                          matrix::Dense<ValueType>* inner_residual)
    GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_STATIC_CONDENSATION1);


template <typename ValueType, typename IndexType>
void static_condensation2(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::Dense<ValueType>* inner_solution,
                          const array<IndexType>& inner_to_local,
                          ValueType* solution) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_STATIC_CONDENSATION2);


}  // namespace distributed_bddc
}  // namespace omp
}  // namespace kernels
}  // namespace gko
