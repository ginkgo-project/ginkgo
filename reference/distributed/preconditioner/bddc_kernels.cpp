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
#include <fstream>

namespace gko {
namespace kernels {
namespace reference {
namespace distributed_bddc {


template <typename ValueType, typename IndexType>
void restrict_residual1(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Dense<ValueType>* global_residual,
                        const array<IndexType>& local_to_local,
                        const array<IndexType>& local_to_send_buffer,
                        const matrix::Diagonal<ValueType>* weights,
                        array<ValueType>& send_buffer,
                        matrix::Dense<ValueType>* local_residual)
{
    auto local_to_local_data = local_to_local.get_const_data();
    auto local_to_send_data = local_to_send_buffer.get_const_data();
    auto send_data = send_buffer.get_data();
    auto w = weights->get_const_values();

    for (auto i = 0; i < global_residual->get_size()[0]; i++) {
        local_residual->at(local_to_local_data[i], 0) =
            w[local_to_local_data[i]] * global_residual->at(i, 0);
    }

    for (auto i = 0; i < local_to_send_buffer.get_num_elems(); i++) {
        send_data[i] = local_residual->at(local_to_send_data[i], 0);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RESTRICT_RESIDUAL1);


template <typename ValueType, typename IndexType>
void restrict_residual2(std::shared_ptr<const DefaultExecutor> exec,
                        const array<IndexType>& non_local_to_local,
                        const array<IndexType>& global_to_recv_buffer,
                        const array<IndexType>& non_local_idxs,
                        const array<ValueType>& recv_buffer,
                        matrix::Dense<ValueType>* local_residual)
{
    auto non_local_to_local_data = non_local_to_local.get_const_data();
    auto global_to_recv_data = global_to_recv_buffer.get_const_data();
    auto recv_data = recv_buffer.get_const_data();

    for (auto i = 0; i < non_local_to_local.get_num_elems(); i++) {
        local_residual->at(non_local_to_local_data[i]) =
            recv_data[global_to_recv_data[i]];
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RESTRICT_RESIDUAL2);


template <typename ValueType, typename IndexType>
void coarsen_residual1(std::shared_ptr<const DefaultExecutor> exec,
                       const array<IndexType>& coarse_local_to_local,
                       const array<IndexType>& coarse_local_to_send,
                       const matrix::Dense<ValueType>* local_coarse_residual,
                       array<ValueType>& coarse_send_buffer,
                       ValueType* coarse_residual, ValueType* coarse_solution)
{
    auto local_to_local_data = coarse_local_to_local.get_const_data();
    auto local_to_send_data = coarse_local_to_send.get_const_data();
    auto send_data = coarse_send_buffer.get_data();

    for (auto i = 0; i < coarse_local_to_local.get_num_elems(); i++) {
        coarse_residual[i] =
            local_coarse_residual->at(local_to_local_data[i], 0);
        coarse_solution[i] = zero<ValueType>();
    }

    for (auto i = 0; i < coarse_local_to_send.get_num_elems(); i++) {
        send_data[i] = local_coarse_residual->at(local_to_send_data[i], 0);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COARSEN_RESIDUAL1);


template <typename ValueType, typename IndexType>
void coarsen_residual2(std::shared_ptr<const DefaultExecutor> exec,
                       const array<IndexType>& coarse_recv_to_local,
                       const array<ValueType>& coarse_recv_buffer,
                       ValueType* coarse_residual)
{
    auto recv_to_local_data = coarse_recv_to_local.get_const_data();
    auto recv_data = coarse_recv_buffer.get_const_data();

    for (auto i = 0; i < coarse_recv_to_local.get_num_elems(); i++) {
        coarse_residual[recv_to_local_data[i]] += recv_data[i];
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COARSEN_RESIDUAL2);


template <typename ValueType, typename IndexType>
void prolong_coarse_solution(
    std::shared_ptr<const DefaultExecutor> exec,
    const array<IndexType>& coarse_local_to_local,
    const matrix::Dense<ValueType>* coarse_solution_local,
    const array<IndexType>& coarse_non_local_to_local,
    const array<IndexType>& coarse_local_to_non_local,
    const matrix::Dense<ValueType>* non_local,
    matrix::Dense<ValueType>* local_intermediate)
{
    auto local_to_local_data = coarse_local_to_local.get_const_data();
    auto non_local_to_local_data = coarse_non_local_to_local.get_const_data();
    auto local_to_non_local_data = coarse_local_to_non_local.get_const_data();

    for (auto i = 0; i < coarse_local_to_local.get_num_elems(); i++) {
        local_intermediate->at(local_to_local_data[i], 0) =
            coarse_solution_local->at(i, 0);
    }

    for (auto i = 0; i < coarse_local_to_non_local.get_num_elems(); i++) {
        if (local_to_non_local_data[i] != -1) {
            local_intermediate->at(non_local_to_local_data[i], 0) =
                non_local->at(local_to_non_local_data[i], 0);
        }
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PROLONG_COARSE_SOLUTION);


template <typename ValueType, typename IndexType>
void prolong_coarse_solution1(std::shared_ptr<const DefaultExecutor> exec,
                              const array<IndexType>& coarse_recv_to_local,
                              const matrix::Dense<ValueType>* coarse_solution,
                              array<ValueType>& coarse_recv_buffer)
{
    auto recv_to_local_data = coarse_recv_to_local.get_const_data();
    auto recv_data = coarse_recv_buffer.get_data();

    for (auto i = 0; i < coarse_recv_buffer.get_num_elems(); i++) {
        recv_data[i] = coarse_solution->at(recv_to_local_data[i], 0);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PROLONG_COARSE_SOLUTION1);


template <typename ValueType, typename IndexType>
void prolong_coarse_solution2(std::shared_ptr<const DefaultExecutor> exec,
                              const array<IndexType>& coarse_local_to_local,
                              const matrix::Dense<ValueType>* coarse_solution,
                              const array<IndexType>& coarse_local_to_send,
                              const array<ValueType>& coarse_send_buffer,
                              matrix::Dense<ValueType>* local_intermediate)
{
    auto local_to_local_data = coarse_local_to_local.get_const_data();
    auto local_to_send_data = coarse_local_to_send.get_const_data();
    auto send_data = coarse_send_buffer.get_const_data();

    for (auto i = 0; i < coarse_local_to_local.get_num_elems(); i++) {
        local_intermediate->at(local_to_local_data[i], 0) =
            coarse_solution->at(i, 0);
    }

    for (auto i = 0; i < coarse_send_buffer.get_num_elems(); i++) {
        local_intermediate->at(local_to_send_data[i], 0) = send_data[i];
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PROLONG_COARSE_SOLUTION2);


template <typename ValueType, typename IndexType>
void finalize1(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::Dense<ValueType>* coarse_solution,
               const matrix::Diagonal<ValueType>* weights,
               const array<IndexType>& recv_to_local,
               const array<IndexType>& non_local_to_local,
               array<ValueType>& recv_buffer,
               matrix::Dense<ValueType>* local_solution)
{
    auto num_rows = coarse_solution->get_size()[0];
    auto w = weights->get_const_values();
    auto non_local_to_local_data = non_local_to_local.get_const_data();
    auto recv_to_local_data = recv_to_local.get_const_data();
    auto recv_data = recv_buffer.get_data();

    for (auto i = 0; i < num_rows; i++) {
        local_solution->at(i, 0) =
            w[i] * (local_solution->at(i, 0) + coarse_solution->at(i, 0));
    }

    for (auto i = 0; i < non_local_to_local.get_num_elems(); i++) {
        recv_data[recv_to_local_data[i]] =
            local_solution->at(non_local_to_local_data[i], 0);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FINALIZE1);


template <typename ValueType, typename IndexType>
void finalize2(std::shared_ptr<const DefaultExecutor> exec,
               const array<ValueType>& send_buffer,
               const array<IndexType>& local_to_send_buffer,
               const array<IndexType>& local_to_local,
               matrix::Dense<ValueType>* local_solution,
               ValueType* global_solution)
{
    auto num_rows = local_to_local.get_num_elems();
    auto send_data = send_buffer.get_const_data();
    auto local_to_send_data = local_to_send_buffer.get_const_data();
    auto local_to_local_data = local_to_local.get_const_data();

    for (auto i = 0; i < send_buffer.get_num_elems(); i++) {
        local_solution->at(local_to_send_data[i], 0) += send_data[i];
    }

    for (auto i = 0; i < num_rows; i++) {
        global_solution[i] = local_solution->at(local_to_local_data[i], 0);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FINALIZE2);


template <typename ValueType, typename IndexType>
void static_condensation1(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::Dense<ValueType>* residual,
                          const array<IndexType>& inner_to_local,
                          matrix::Dense<ValueType>* inner_residual)
{
    for (auto i = 0; i < inner_to_local.get_num_elems(); i++) {
        if (inner_to_local.get_const_data()[i] != -1) {
            inner_residual->at(inner_to_local.get_const_data()[i], 0) =
                residual->at(
                    i,
                    0);  // i, 0) =
                         // residual->at(inner_to_local.get_const_data()[i], 0);
        }
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_STATIC_CONDENSATION1);


template <typename ValueType, typename IndexType>
void static_condensation2(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::Dense<ValueType>* inner_solution,
                          const array<IndexType>& inner_to_local,
                          ValueType* solution)
{
    for (auto i = 0; i < inner_to_local.get_num_elems(); i++) {
        if (inner_to_local.get_const_data()[i] != -1) {
            solution[i] +=
                inner_solution->at(inner_to_local.get_const_data()[i]);
        }
        // solution[inner_to_local.get_const_data()[i]] +=
        // inner_solution->at(i,0);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_STATIC_CONDENSATION2);


}  // namespace distributed_bddc
}  // namespace reference
}  // namespace kernels
}  // namespace gko
