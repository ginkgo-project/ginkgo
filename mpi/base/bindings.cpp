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


#include <iostream>
#include <map>


#include "mpi/base/bindings.hpp"


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/mpi.hpp>


#include "mpi/base/helpers.hpp"


namespace gko {
namespace mpi {


MPI_Op create_operation(
    const std::function<void(void*, void*, int*, MPI_Datatype*)> func,
    void* arg1, void* arg2, int* len, MPI_Datatype* type)
{
    MPI_Op operation;
    bindings::create_op(func.target<void(void*, void*, int*, MPI_Datatype*)>(),
                        true, &operation);
    return operation;
}


template <typename SendType>
void send(const SendType* send_buffer, const int send_count,
          const int destination_rank, const int send_tag,
          std::shared_ptr<request> req,
          std::shared_ptr<const communicator> comm)
{
    auto send_type = helpers::get_mpi_type(send_buffer[0]);
    if (!req.get()) {
        bindings::send(send_buffer, send_count, send_type, destination_rank,
                       send_tag,
                       comm ? comm->get() : communicator::get_comm_world());
    } else {
        bindings::i_send(
            send_buffer, send_count, send_type, destination_rank, send_tag,
            comm ? comm->get() : communicator::get_comm_world(), req->get());
    }
}


template <typename RecvType>
void recv(RecvType* recv_buffer, const int recv_count, const int source_rank,
          const int recv_tag, std::shared_ptr<request> req,
          std::shared_ptr<status> status,
          std::shared_ptr<const communicator> comm)
{
    auto recv_type = helpers::get_mpi_type(recv_buffer[0]);
    if (!req.get()) {
        bindings::recv(recv_buffer, recv_count, recv_type, source_rank,
                       recv_tag,
                       comm ? comm->get() : communicator::get_comm_world(),
                       MPI_STATUS_IGNORE);
    } else {
        bindings::i_recv(
            recv_buffer, recv_count, recv_type, source_rank, recv_tag,
            comm ? comm->get() : communicator::get_comm_world(), req->get());
    }
}


template <typename PutType>
void put(const PutType* origin_buffer, const int origin_count,
         const int target_rank, const unsigned int target_disp,
         const int target_count, window<PutType>& window,
         std::shared_ptr<request> req)
{
    auto put_type = helpers::get_mpi_type(origin_buffer[0]);
    if (!req.get()) {
        bindings::put(origin_buffer, origin_count, put_type, target_rank,
                      target_disp, target_count, put_type, window.get());
    } else {
        bindings::req_put(origin_buffer, origin_count, put_type, target_rank,
                          target_disp, target_count, put_type, window.get(),
                          req->get());
    }
}


template <typename GetType>
void get(GetType* origin_buffer, const int origin_count, const int target_rank,
         const unsigned int target_disp, const int target_count,
         window<GetType>& window, std::shared_ptr<request> req)
{
    auto get_type = helpers::get_mpi_type(origin_buffer[0]);
    if (!req.get()) {
        bindings::get(origin_buffer, origin_count, get_type, target_rank,
                      target_disp, target_count, get_type, window.get());
    } else {
        bindings::req_get(origin_buffer, origin_count, get_type, target_rank,
                          target_disp, target_count, get_type, window.get(),
                          req->get());
    }
}


template <typename BroadcastType>
void broadcast(BroadcastType* buffer, int count, int root_rank,
               std::shared_ptr<const communicator> comm)
{
    auto bcast_type = helpers::get_mpi_type(buffer[0]);
    bindings::broadcast(buffer, count, bcast_type, root_rank,
                        comm ? comm->get() : communicator::get_comm_world());
}


template <typename ReduceType>
void reduce(const ReduceType* send_buffer, ReduceType* recv_buffer, int count,
            op_type op_enum, int root_rank,
            std::shared_ptr<const communicator> comm,
            std::shared_ptr<request> req)
{
    auto operation = helpers::get_operation<ReduceType>(op_enum);
    auto reduce_type = helpers::get_mpi_type(send_buffer[0]);
    if (!req.get()) {
        bindings::reduce(send_buffer, recv_buffer, count, reduce_type,
                         operation, root_rank,
                         comm ? comm->get() : communicator::get_comm_world());
    } else {
        bindings::i_reduce(
            send_buffer, recv_buffer, count, reduce_type, operation, root_rank,
            comm ? comm->get() : communicator::get_comm_world(), req->get());
    }
}


template <typename ReduceType>
void all_reduce(ReduceType* recv_buffer, int count, op_type op_enum,
                std::shared_ptr<const communicator> comm,
                std::shared_ptr<request> req)
{
    auto operation = helpers::get_operation<ReduceType>(op_enum);
    auto reduce_type = helpers::get_mpi_type(recv_buffer[0]);
    if (!req.get()) {
        bindings::all_reduce(
            bindings::in_place<ReduceType>(), recv_buffer, count, reduce_type,
            operation, comm ? comm->get() : communicator::get_comm_world());
    } else {
        bindings::i_all_reduce(
            bindings::in_place<ReduceType>(), recv_buffer, count, reduce_type,
            operation, comm ? comm->get() : communicator::get_comm_world(),
            req->get());
    }
}


template <typename ReduceType>
void all_reduce(const ReduceType* send_buffer, ReduceType* recv_buffer,
                int count, op_type op_enum,
                std::shared_ptr<const communicator> comm,
                std::shared_ptr<request> req)
{
    auto operation = helpers::get_operation<ReduceType>(op_enum);
    auto reduce_type = helpers::get_mpi_type(recv_buffer[0]);
    if (!req.get()) {
        bindings::all_reduce(
            send_buffer, recv_buffer, count, reduce_type, operation,
            comm ? comm->get() : communicator::get_comm_world());
    } else {
        bindings::i_all_reduce(
            send_buffer, recv_buffer, count, reduce_type, operation,
            comm ? comm->get() : communicator::get_comm_world(), req->get());
    }
}


template <typename SendType, typename RecvType>
void gather(const SendType* send_buffer, const int send_count,
            RecvType* recv_buffer, const int recv_count, int root_rank,
            std::shared_ptr<const communicator> comm)
{
    auto send_type = helpers::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::get_mpi_type(recv_buffer[0]);
    bindings::gather(send_buffer, send_count, send_type, recv_buffer,
                     recv_count, recv_type, root_rank,
                     comm ? comm->get() : communicator::get_comm_world());
}


template <typename SendType, typename RecvType>
void gather(const SendType* send_buffer, const int send_count,
            RecvType* recv_buffer, const int* recv_counts,
            const int* displacements, int root_rank,
            std::shared_ptr<const communicator> comm)
{
    auto send_type = helpers::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::get_mpi_type(recv_buffer[0]);
    bindings::gatherv(send_buffer, send_count, send_type, recv_buffer,
                      recv_counts, displacements, recv_type, root_rank,
                      comm ? comm->get() : communicator::get_comm_world());
}


template <typename SendType, typename RecvType>
void all_gather(const SendType* send_buffer, const int send_count,
                RecvType* recv_buffer, const int recv_count,
                std::shared_ptr<const communicator> comm)
{
    auto send_type = helpers::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::get_mpi_type(recv_buffer[0]);
    bindings::all_gather(send_buffer, send_count, send_type, recv_buffer,
                         recv_count, recv_type,
                         comm ? comm->get() : communicator::get_comm_world());
}


template <typename SendType, typename RecvType>
void scatter(const SendType* send_buffer, const int send_count,
             RecvType* recv_buffer, const int recv_count, int root_rank,
             std::shared_ptr<const communicator> comm)
{
    auto send_type = helpers::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::get_mpi_type(recv_buffer[0]);
    bindings::scatter(send_buffer, send_count, send_type, recv_buffer,
                      recv_count, recv_type, root_rank,
                      comm ? comm->get() : communicator::get_comm_world());
}


template <typename SendType, typename RecvType>
void scatter(const SendType* send_buffer, const int* send_counts,
             const int* displacements, RecvType* recv_buffer,
             const int recv_count, int root_rank,
             std::shared_ptr<const communicator> comm)
{
    auto send_type = helpers::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::get_mpi_type(recv_buffer[0]);
    bindings::scatterv(send_buffer, send_counts, displacements, send_type,
                       recv_buffer, recv_count, recv_type, root_rank,
                       comm ? comm->get() : communicator::get_comm_world());
}


template <typename RecvType>
void all_to_all(RecvType* recv_buffer, const int recv_count,
                std::shared_ptr<const communicator> comm,
                std::shared_ptr<request> req)
{
    auto recv_type = helpers::get_mpi_type(recv_buffer[0]);
    if (!req.get()) {
        bindings::all_to_all(
            bindings::in_place<RecvType>(), recv_count, recv_type, recv_buffer,
            recv_count, recv_type,
            comm ? comm->get() : communicator::get_comm_world());
    } else {
        bindings::i_all_to_all(
            bindings::in_place<RecvType>(), recv_count, recv_type, recv_buffer,
            recv_count, recv_type,
            comm ? comm->get() : communicator::get_comm_world(), req->get());
    }
}


template <typename SendType, typename RecvType>
void all_to_all(const SendType* send_buffer, const int send_count,
                RecvType* recv_buffer, const int recv_count,
                std::shared_ptr<const communicator> comm,
                std::shared_ptr<request> req)
{
    auto send_type = helpers::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::get_mpi_type(recv_buffer[0]);
    if (!req.get()) {
        bindings::all_to_all(
            send_buffer, send_count, send_type, recv_buffer,
            recv_count == 0 ? send_count : recv_count, recv_type,
            comm ? comm->get() : communicator::get_comm_world());
    } else {
        bindings::i_all_to_all(
            send_buffer, send_count, send_type, recv_buffer,
            recv_count == 0 ? send_count : recv_count, recv_type,
            comm ? comm->get() : communicator::get_comm_world(), req->get());
    }
}


template <typename SendType, typename RecvType>
void all_to_all(const SendType* send_buffer, const int* send_counts,
                const int* send_offsets, RecvType* recv_buffer,
                const int* recv_counts, const int* recv_offsets,
                const int stride, std::shared_ptr<const communicator> comm,
                std::shared_ptr<request> req)
{
    auto send_type = helpers::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::get_mpi_type(recv_buffer[0]);

    // auto new_type = mpi_type(stride, send_type);

    if (!req.get()) {
        bindings::all_to_all_v(
            send_buffer, send_counts, send_offsets, send_type, recv_buffer,
            recv_counts, recv_offsets, recv_type,
            comm ? comm->get() : communicator::get_comm_world());
    } else {
        bindings::i_all_to_all_v(
            send_buffer, send_counts, send_offsets, send_type, recv_buffer,
            recv_counts, recv_offsets, recv_type,
            comm ? comm->get() : communicator::get_comm_world(), req->get());
    }
}


template <typename ScanType>
void scan(const ScanType* send_buffer, ScanType* recv_buffer, int count,
          op_type op_enum, std::shared_ptr<const communicator> comm,
          std::shared_ptr<request> req)
{
    auto operation = helpers::get_operation<ScanType>(op_enum);
    auto scan_type = helpers::get_mpi_type(recv_buffer[0]);
    if (!req.get()) {
        bindings::scan(send_buffer, recv_buffer, count, scan_type, operation,
                       comm ? comm->get() : communicator::get_comm_world());
    } else {
        bindings::i_scan(send_buffer, recv_buffer, count, scan_type, operation,
                         comm ? comm->get() : communicator::get_comm_world(),
                         req->get());
    }
}


#define GKO_DECLARE_WINDOW(ValueType) class window<ValueType>

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_WINDOW);


#define GKO_DECLARE_SEND(SendType)                               \
    void send(const SendType* send_buffer, const int send_count, \
              const int destination_rank, const int send_tag,    \
              std::shared_ptr<request> req,                      \
              std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_SEND);


#define GKO_DECLARE_RECV(RecvType)                                          \
    void recv(RecvType* recv_buffer, const int recv_count,                  \
              const int source_rank, const int recv_tag,                    \
              std::shared_ptr<request> req, std::shared_ptr<status> status, \
              std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_RECV);


#define GKO_DECLARE_PUT(PutType)                                    \
    void put(const PutType* origin_buffer, const int origin_count,  \
             const int target_rank, const unsigned int target_disp, \
             const int target_count, window<PutType>& window,       \
             std::shared_ptr<request> req)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_PUT);


#define GKO_DECLARE_GET(GetType)                                    \
    void get(GetType* origin_buffer, const int origin_count,        \
             const int target_rank, const unsigned int target_disp, \
             const int target_count, window<GetType>& window,       \
             std::shared_ptr<request> req)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_GET);


#define GKO_DECLARE_BCAST(BroadcastType)                            \
    void broadcast(BroadcastType* buffer, int count, int root_rank, \
                   std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_BCAST);


#define GKO_DECLARE_REDUCE(ReduceType)                                  \
    void reduce(const ReduceType* send_buffer, ReduceType* recv_buffer, \
                int count, op_type operation, int root_rank,            \
                std::shared_ptr<const communicator> comm,               \
                std::shared_ptr<request> req)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_REDUCE);


#define GKO_DECLARE_ALLREDUCE1(ReduceType)                                 \
    void all_reduce(ReduceType* recv_buffer, int count, op_type operation, \
                    std::shared_ptr<const communicator> comm,              \
                    std::shared_ptr<request> req)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_ALLREDUCE1);


#define GKO_DECLARE_ALLREDUCE2(ReduceType)                                  \
    void all_reduce(const ReduceType* send_buffer, ReduceType* recv_buffer, \
                    int count, op_type operation,                           \
                    std::shared_ptr<const communicator> comm,               \
                    std::shared_ptr<request> req)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_ALLREDUCE2);


#define GKO_DECLARE_GATHER1(SendType, RecvType)                             \
    void gather(const SendType* send_buffer, const int send_count,          \
                RecvType* recv_buffer, const int recv_count, int root_rank, \
                std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GATHER1);


#define GKO_DECLARE_GATHER2(SendType, RecvType)                    \
    void gather(const SendType* send_buffer, const int send_count, \
                RecvType* recv_buffer, const int* recv_counts,     \
                const int* displacements, int root_rank,           \
                std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GATHER2);


#define GKO_DECLARE_ALLGATHER(SendType, RecvType)                      \
    void all_gather(const SendType* send_buffer, const int send_count, \
                    RecvType* recv_buffer, const int recv_count,       \
                    std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ALLGATHER);


#define GKO_DECLARE_SCATTER1(SendType, RecvType)                             \
    void scatter(const SendType* send_buffer, const int send_count,          \
                 RecvType* recv_buffer, const int recv_count, int root_rank, \
                 std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SCATTER1);


#define GKO_DECLARE_SCATTER2(SendType, RecvType)                      \
    void scatter(const SendType* send_buffer, const int* send_counts, \
                 const int* displacements, RecvType* recv_buffer,     \
                 const int recv_count, int root_rank,                 \
                 std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SCATTER2);


#define GKO_DECLARE_ALL_TO_ALL1(RecvType)                        \
    void all_to_all(RecvType* recv_buffer, const int recv_count, \
                    std::shared_ptr<const communicator> comm,    \
                    std::shared_ptr<request> req)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_ALL_TO_ALL1);


#define GKO_DECLARE_ALL_TO_ALL2(SendType, RecvType)                    \
    void all_to_all(const SendType* send_buffer, const int send_count, \
                    RecvType* recv_buffer, const int recv_count,       \
                    std::shared_ptr<const communicator> comm,          \
                    std::shared_ptr<request> req)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ALL_TO_ALL2);


#define GKO_DECLARE_ALL_TO_ALL_V(SendType, RecvType)                     \
    void all_to_all(const SendType* send_buffer, const int* send_counts, \
                    const int* send_offsets, RecvType* recv_buffer,      \
                    const int* recv_counts, const int* recv_offsets,     \
                    const int stride,                                    \
                    std::shared_ptr<const communicator> comm,            \
                    std::shared_ptr<request> req)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ALL_TO_ALL_V);


#define GKO_DECLARE_SCAN(ScanType)                                           \
    void scan(const ScanType* send_buffer, ScanType* recv_buffer, int count, \
              op_type op_enum, std::shared_ptr<const communicator> comm,     \
              std::shared_ptr<request> req)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_SCAN);


}  // namespace mpi
}  // namespace gko
