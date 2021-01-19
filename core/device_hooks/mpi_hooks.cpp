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

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/version.hpp>


namespace gko {


version version_info::get_mpi_version() noexcept
{
    // We just return 1.0.0 with a special "not compiled" tag in placeholder
    // modules.
    return {1, 0, 0, "not compiled"};
}


int MpiExecutor::get_num_ranks(const MPI_Comm &comm) const { return 0; }


int MpiExecutor::get_my_rank(const MPI_Comm &comm) const GKO_NOT_COMPILED(mpi);


std::shared_ptr<MpiExecutor> MpiExecutor::create(
    std::shared_ptr<Executor> sub_executor)
{
    return std::shared_ptr<MpiExecutor>(new MpiExecutor(sub_executor));
}


std::string MpiError::get_error(int64)
{
    return "ginkgo MPI module is not compiled";
}


bool mpi::init_finalize::is_finalized() GKO_NOT_COMPILED(mpi);


bool mpi::init_finalize::is_initialized() GKO_NOT_COMPILED(mpi);


mpi::init_finalize::init_finalize(int &argc, char **&argv,
                                  const size_type num_threads)
    GKO_NOT_COMPILED(mpi);


mpi::init_finalize::~init_finalize() noexcept(false) GKO_NOT_COMPILED(mpi);


mpi::communicator::communicator(const MPI_Comm &comm) GKO_NOT_COMPILED(mpi);


mpi::communicator::communicator(const MPI_Comm &comm_in, int color, int key)
    GKO_NOT_COMPILED(mpi);


mpi::communicator::~communicator() {}


void MpiExecutor::synchronize_communicator(const MPI_Comm &comm) const
    GKO_NOT_COMPILED(mpi);


void MpiExecutor::synchronize() const GKO_NOT_COMPILED(mpi);


void MpiExecutor::set_communicator(const MPI_Comm &comm) GKO_NOT_COMPILED(mpi);


void MpiExecutor::wait(MPI_Request *req, MPI_Status *status)
    GKO_NOT_COMPILED(mpi);


MpiExecutor::request_manager<MPI_Request> MpiExecutor::create_requests_array(
    int size) GKO_NOT_COMPILED(mpi);


MPI_Op MpiExecutor::create_operation(
    std::function<void(void *, void *, int *, MPI_Datatype *)> func, void *arg1,
    void *arg2, int *len, MPI_Datatype *type) GKO_NOT_COMPILED(mpi);


template <typename SendType>
void MpiExecutor::send(const SendType *send_buffer, const int send_count,
                       const int destination_rank, const int send_tag,
                       MPI_Request *req) const GKO_NOT_COMPILED(mpi);


template <typename RecvType>
void MpiExecutor::recv(RecvType *recv_buffer, const int recv_count,
                       const int source_rank, const int recv_tag,
                       MPI_Request *req) const GKO_NOT_COMPILED(mpi);


template <typename BroadcastType>
void MpiExecutor::broadcast(BroadcastType *buffer, int count,
                            int root_rank) const GKO_NOT_COMPILED(mpi);


template <typename ReduceType>
void MpiExecutor::reduce(const ReduceType *send_buffer, ReduceType *recv_buffer,
                         int count, mpi::op_type op_enum, int root_rank,
                         MPI_Request *req) const GKO_NOT_COMPILED(mpi);


template <typename ReduceType>
void MpiExecutor::all_reduce(const ReduceType *send_buffer,
                             ReduceType *recv_buffer, int count,
                             mpi::op_type op_enum, MPI_Request *req) const
    GKO_NOT_COMPILED(mpi);


template <typename SendType, typename RecvType>
void MpiExecutor::gather(const SendType *send_buffer, const int send_count,
                         RecvType *recv_buffer, const int recv_count,
                         int root_rank) const GKO_NOT_COMPILED(mpi);


template <typename SendType, typename RecvType>
void MpiExecutor::gather(const SendType *send_buffer, const int send_count,
                         RecvType *recv_buffer, const int *recv_counts,
                         const int *displacements, int root_rank) const
    GKO_NOT_COMPILED(mpi);


template <typename SendType, typename RecvType>
void MpiExecutor::scatter(const SendType *send_buffer, const int send_count,
                          RecvType *recv_buffer, const int recv_count,
                          int root_rank) const GKO_NOT_COMPILED(mpi);


template <typename SendType, typename RecvType>
void MpiExecutor::scatter(const SendType *send_buffer, const int *send_counts,
                          const int *displacements, RecvType *recv_buffer,
                          const int recv_count, int root_rank) const
    GKO_NOT_COMPILED(mpi);


#define GKO_DECLARE_SEND(SendType)                                            \
    void MpiExecutor::send(const SendType *send_buffer, const int send_count, \
                           const int destination_rank, const int send_tag,    \
                           MPI_Request *req) const

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_SEND);


#define GKO_DECLARE_RECV(RecvType)                                      \
    void MpiExecutor::recv(RecvType *recv_buffer, const int recv_count, \
                           const int source_rank, const int recv_tag,   \
                           MPI_Request *req) const

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_RECV);


#define GKO_DECLARE_BCAST(BroadcastType)                          \
    void MpiExecutor::broadcast(BroadcastType *buffer, int count, \
                                int root_rank) const

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_BCAST);


#define GKO_DECLARE_REDUCE(ReduceType)                                     \
    void MpiExecutor::reduce(                                              \
        const ReduceType *send_buffer, ReduceType *recv_buffer, int count, \
        mpi::op_type operation, int root_rank, MPI_Request *req) const

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_REDUCE);


#define GKO_DECLARE_ALLREDUCE(ReduceType)                                  \
    void MpiExecutor::all_reduce(                                          \
        const ReduceType *send_buffer, ReduceType *recv_buffer, int count, \
        mpi::op_type operation, MPI_Request *req) const

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_ALLREDUCE);


#define GKO_DECLARE_GATHER1(SendType, RecvType)                           \
    void MpiExecutor::gather(const SendType *send_buffer,                 \
                             const int send_count, RecvType *recv_buffer, \
                             const int recv_count, int root_rank) const

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GATHER1);


#define GKO_DECLARE_GATHER2(SendType, RecvType)                                \
    void MpiExecutor::gather(const SendType *send_buffer,                      \
                             const int send_count, RecvType *recv_buffer,      \
                             const int *recv_counts, const int *displacements, \
                             int root_rank) const

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GATHER2);


#define GKO_DECLARE_SCATTER1(SendType, RecvType)                           \
    void MpiExecutor::scatter(const SendType *send_buffer,                 \
                              const int send_count, RecvType *recv_buffer, \
                              const int recv_count, int root_rank) const

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SCATTER1);


#define GKO_DECLARE_SCATTER2(SendType, RecvType)                               \
    void MpiExecutor::scatter(const SendType *send_buffer,                     \
                              const int *send_counts,                          \
                              const int *displacements, RecvType *recv_buffer, \
                              const int recv_count, int root_rank) const

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SCATTER2);


}  // namespace gko
