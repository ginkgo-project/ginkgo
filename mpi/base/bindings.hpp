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

#ifndef GKO_MPI_BINDINGS_HPP_
#define GKO_MPI_BINDINGS_HPP_


#include <iostream>


#include <mpi.h>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
/**
 * @brief The MPI namespace.
 *
 * @ingroup mpi
 */
namespace mpi {
/**
 * @brief The bindings namespace.
 *
 * @ingroup bindings
 */
namespace bindings {


inline void create_op(MPI_User_function* func, int commute, MPI_Op* op)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Op_create(func, commute, op));
}


template <typename T>
inline const T* in_place()
{
    return reinterpret_cast<const T*>(MPI_IN_PLACE);
}

inline void send(const void* send_buffer, const int send_count,
                 MPI_Datatype& send_type, const int destination_rank,
                 const int send_tag, const MPI_Comm comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Send(send_buffer, send_count, send_type,
                                      destination_rank, send_tag, comm));
}


inline void recv(void* recv_buffer, const int recv_count,
                 MPI_Datatype& recv_type, const int source_rank,
                 const int recv_tag, const MPI_Comm comm, MPI_Status* status)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Recv(recv_buffer, recv_count, recv_type,
                                      source_rank, recv_tag, comm, status));
}


inline void i_send(const void* send_buffer, const int send_count,
                   MPI_Datatype& send_type, const int destination_rank,
                   const int send_tag, const MPI_Comm comm,
                   MPI_Request* request)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Isend(send_buffer, send_count, send_type,
                                       destination_rank, send_tag, comm,
                                       request));
}


inline void i_recv(void* recv_buffer, const int recv_count,
                   MPI_Datatype& recv_type, const int source_rank,
                   const int recv_tag, const MPI_Comm comm,
                   MPI_Request* request)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Irecv(recv_buffer, recv_count, recv_type,
                                       source_rank, recv_tag, comm, request));
}


inline void put(const void* origin_buffer, const int origin_count,
                const MPI_Datatype& origin_type, const int target_rank,
                const unsigned int target_disp, const int target_count,
                const MPI_Datatype& target_type, MPI_Win window)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Put(origin_buffer, origin_count, origin_type,
                                     target_rank, target_disp, target_count,
                                     target_type, window));
}


inline void req_put(const void* origin_buffer, const int origin_count,
                    const MPI_Datatype& origin_type, const int target_rank,
                    const unsigned int target_disp, const int target_count,
                    const MPI_Datatype& target_type, MPI_Win window,
                    MPI_Request* request)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Rput(origin_buffer, origin_count, origin_type,
                                      target_rank, target_disp, target_count,
                                      target_type, window, request));
}


inline void get(void* origin_buffer, const int origin_count,
                const MPI_Datatype& origin_type, const int target_rank,
                const unsigned int target_disp, const int target_count,
                const MPI_Datatype& target_type, MPI_Win window)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Get(origin_buffer, origin_count, origin_type,
                                     target_rank, target_disp, target_count,
                                     target_type, window));
}


inline void req_get(void* origin_buffer, const int origin_count,
                    const MPI_Datatype& origin_type, const int target_rank,
                    const unsigned int target_disp, const int target_count,
                    const MPI_Datatype& target_type, MPI_Win window,
                    MPI_Request* request)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Rget(origin_buffer, origin_count, origin_type,
                                      target_rank, target_disp, target_count,
                                      target_type, window, request));
}


inline void broadcast(void* buffer, int count, MPI_Datatype& bcast_type,
                      int root_rank, const MPI_Comm& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Bcast(buffer, count, bcast_type, root_rank, comm));
}


inline void reduce(const void* send_buffer, void* recv_buffer, int count,
                   MPI_Datatype& reduce_type, MPI_Op operation, int root_rank,
                   const MPI_Comm& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Reduce(send_buffer, recv_buffer, count,
                                        reduce_type, operation, root_rank,
                                        comm));
}


inline void all_reduce(const void* send_buffer, void* recv_buffer, int count,
                       MPI_Datatype& reduce_type, MPI_Op operation,
                       const MPI_Comm& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Allreduce(send_buffer, recv_buffer, count,
                                           reduce_type, operation, comm));
}


inline void i_reduce(const void* send_buffer, void* recv_buffer, int count,
                     MPI_Datatype& reduce_type, MPI_Op operation, int root_rank,
                     const MPI_Comm& comm, MPI_Request* requests)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Ireduce(send_buffer, recv_buffer, count,
                                         reduce_type, operation, root_rank,
                                         comm, requests));
}


inline void i_all_reduce(const void* send_buffer, void* recv_buffer, int count,
                         MPI_Datatype& reduce_type, MPI_Op operation,
                         const MPI_Comm& comm, MPI_Request* requests)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Iallreduce(send_buffer, recv_buffer, count,
                                            reduce_type, operation, comm,
                                            requests));
}


inline void gather(const void* send_buffer, const int send_count,
                   MPI_Datatype& send_type, void* recv_buffer,
                   const int recv_count, MPI_Datatype& recv_type, int root,
                   const MPI_Comm& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Gather(send_buffer, send_count, send_type,
                                        recv_buffer, recv_count, recv_type,
                                        root, comm));
}


inline void gatherv(const void* send_buffer, const int send_count,
                    MPI_Datatype& send_type, void* recv_buffer,
                    const int* recv_counts, const int* displacements,
                    MPI_Datatype& recv_type, int root_rank,
                    const MPI_Comm& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Gatherv(send_buffer, send_count, send_type, recv_buffer,
                    recv_counts, displacements, recv_type, root_rank, comm));
}


inline void all_gather(const void* send_buffer, const int send_count,
                       MPI_Datatype& send_type, void* recv_buffer,
                       const int recv_count, MPI_Datatype& recv_type,
                       const MPI_Comm& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Allgather(send_buffer, send_count, send_type,
                                           recv_buffer, recv_count, recv_type,
                                           comm));
}


inline void scatter(const void* send_buffer, const int send_count,
                    MPI_Datatype& send_type, void* recv_buffer,
                    const int recv_count, MPI_Datatype& recv_type, int root,
                    const MPI_Comm& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Scatter(send_buffer, send_count, send_type,
                                         recv_buffer, recv_count, recv_type,
                                         root, comm));
}


inline void scatterv(const void* send_buffer, const int* send_counts,
                     const int* displacements, MPI_Datatype& send_type,
                     void* recv_buffer, const int recv_count,
                     MPI_Datatype& recv_type, int root_rank,
                     const MPI_Comm& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Scatterv(send_buffer, send_counts, displacements, send_type,
                     recv_buffer, recv_count, recv_type, root_rank, comm));
}


inline void all_to_all(const void* send_buffer, const int send_count,
                       MPI_Datatype& send_type, void* recv_buffer,
                       const int recv_count, MPI_Datatype& recv_type,
                       const MPI_Comm& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Alltoall(send_buffer, send_count, send_type,
                                          recv_buffer, recv_count, recv_type,
                                          comm));
}


inline void i_all_to_all(const void* send_buffer, const int send_count,
                         MPI_Datatype& send_type, void* recv_buffer,
                         const int recv_count, MPI_Datatype& recv_type,
                         const MPI_Comm& comm, MPI_Request* requests)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Ialltoall(send_buffer, send_count, send_type,
                                           recv_buffer, recv_count, recv_type,
                                           comm, requests));
}


inline void all_to_all_v(const void* send_buffer, const int* send_count,
                         const int* send_offsets, const MPI_Datatype& send_type,
                         void* recv_buffer, const int* recv_count,
                         const int* recv_offsets, const MPI_Datatype& recv_type,
                         const MPI_Comm& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Alltoallv(send_buffer, send_count, send_offsets, send_type,
                      recv_buffer, recv_count, recv_offsets, recv_type, comm));
}


inline void i_all_to_all_v(const void* send_buffer, const int* send_count,
                           const int* send_offsets,
                           const MPI_Datatype& send_type, void* recv_buffer,
                           const int* recv_count, const int* recv_offsets,
                           const MPI_Datatype& recv_type, const MPI_Comm& comm,
                           MPI_Request* requests)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Ialltoallv(
        send_buffer, send_count, send_offsets, send_type, recv_buffer,
        recv_count, recv_offsets, recv_type, comm, requests));
}


inline void scan(const void* send_buffer, void* recv_buffer, int count,
                 MPI_Datatype& reduce_type, MPI_Op operation,
                 const MPI_Comm& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Scan(send_buffer, recv_buffer, count,
                                      reduce_type, operation, comm));
}


inline void i_scan(const void* send_buffer, void* recv_buffer, int count,
                   MPI_Datatype& reduce_type, MPI_Op operation,
                   const MPI_Comm& comm, MPI_Request* requests)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Iscan(send_buffer, recv_buffer, count,
                                       reduce_type, operation, comm, requests));
}


}  // namespace bindings
}  // namespace mpi
}  // namespace gko


#endif  // GKO_MPI_BINDINGS_HPP_
