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
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/base/version.hpp>


namespace gko {


version version_info::get_mpi_version() noexcept
{
    // We just return 1.0.0 with a special "not compiled" tag in placeholder
    // modules.
    return {1, 0, 0, "not compiled"};
}


std::string MpiError::get_error(int64)
{
    return "ginkgo MPI module is not compiled";
}


namespace mpi {


bool init_finalize::is_finalized() GKO_NOT_COMPILED(mpi);


bool init_finalize::is_initialized() GKO_NOT_COMPILED(mpi);


init_finalize::init_finalize(int& argc, char**& argv,
                             const size_type num_threads) GKO_NOT_COMPILED(mpi);


init_finalize::~init_finalize() {}


communicator::communicator(const MPI_Comm& comm) GKO_NOT_COMPILED(mpi);


communicator::communicator(const MPI_Comm& comm_in, int color, int key)
    GKO_NOT_COMPILED(mpi);


communicator::~communicator() {}

info::info() GKO_NOT_COMPILED(mpi);

void info::add(std::string key, std::string value) GKO_NOT_COMPILED(mpi);


void info::remove(std::string key) GKO_NOT_COMPILED(mpi);


info::~info() {}


bool communicator::compare(const MPI_Comm& comm) const GKO_NOT_COMPILED(mpi);


template <typename ValueType>
window<ValueType>::window(ValueType* base, unsigned int size,
                          std::shared_ptr<const communicator> comm,
                          const int disp_unit, info input_info,
                          win_type create_type) GKO_NOT_COMPILED(mpi);


template <typename ValueType>
void window<ValueType>::fence(int assert) GKO_NOT_COMPILED(mpi);


template <typename ValueType>
void window<ValueType>::lock(int rank, int assert, lock_type lock_t)
    GKO_NOT_COMPILED(mpi);


template <typename ValueType>
void window<ValueType>::unlock(int rank) GKO_NOT_COMPILED(mpi);


template <typename ValueType>
void window<ValueType>::lock_all(int assert) GKO_NOT_COMPILED(mpi);


template <typename ValueType>
void window<ValueType>::unlock_all() GKO_NOT_COMPILED(mpi);


template <typename ValueType>
void window<ValueType>::flush(int rank) GKO_NOT_COMPILED(mpi);


template <typename ValueType>
void window<ValueType>::flush_local(int rank) GKO_NOT_COMPILED(mpi);


template <typename ValueType>
void window<ValueType>::flush_all() GKO_NOT_COMPILED(mpi);


template <typename ValueType>
void window<ValueType>::flush_all_local() GKO_NOT_COMPILED(mpi);


template <typename ValueType>
window<ValueType>::~window()
{}


MPI_Op create_operation(
    const std::function<void(void*, void*, int*, MPI_Datatype*)> func,
    void* arg1, void* arg2, int* len, MPI_Datatype* type) GKO_NOT_COMPILED(mpi);


double get_walltime() GKO_NOT_COMPILED(mpi);


int get_my_rank(const communicator& comm) GKO_NOT_COMPILED(mpi);


int get_local_rank(const communicator& comm) GKO_NOT_COMPILED(mpi);


int get_num_ranks(const communicator& comm) GKO_NOT_COMPILED(mpi);


void synchronize(const communicator& comm) GKO_NOT_COMPILED(mpi);


void wait(std::shared_ptr<request> req, std::shared_ptr<status> status)
    GKO_NOT_COMPILED(mpi);


template <typename SendType>
void send(const SendType* send_buffer, const int send_count,
          const int destination_rank, const int send_tag,
          std::shared_ptr<request> req,
          std::shared_ptr<const communicator> comm) GKO_NOT_COMPILED(mpi);


template <typename RecvType>
void recv(RecvType* recv_buffer, const int recv_count, const int source_rank,
          const int recv_tag, std::shared_ptr<request> req,
          std::shared_ptr<status> status,
          std::shared_ptr<const communicator> comm) GKO_NOT_COMPILED(mpi);


template <typename PutType>
void put(const PutType* origin_buffer, const int origin_count,
         const int target_rank, const unsigned int target_disp,
         const int target_count, window<PutType>& window,
         std::shared_ptr<request> req) GKO_NOT_COMPILED(mpi);


template <typename GetType>
void get(GetType* origin_buffer, const int origin_count, const int target_rank,
         const unsigned int target_disp, const int target_count,
         window<GetType>& window, std::shared_ptr<request> req)
    GKO_NOT_COMPILED(mpi);


template <typename BroadcastType>
void broadcast(BroadcastType* buffer, int count, int root_rank,
               std::shared_ptr<const communicator> comm) GKO_NOT_COMPILED(mpi);


template <typename ReduceType>
void reduce(const ReduceType* send_buffer, ReduceType* recv_buffer, int count,
            op_type op_enum, int root_rank, std::shared_ptr<request> req,
            std::shared_ptr<const communicator> comm) GKO_NOT_COMPILED(mpi);


template <typename ReduceType>
void all_reduce(const ReduceType* send_buffer, ReduceType* recv_buffer,
                int count, op_type op_enum, std::shared_ptr<request> req,
                std::shared_ptr<const communicator> comm) GKO_NOT_COMPILED(mpi);


template <typename SendType, typename RecvType>
void gather(const SendType* send_buffer, const int send_count,
            RecvType* recv_buffer, const int recv_count, int root_rank,
            std::shared_ptr<const communicator> comm) GKO_NOT_COMPILED(mpi);


template <typename SendType, typename RecvType>
void gather(const SendType* send_buffer, const int send_count,
            RecvType* recv_buffer, const int* recv_counts,
            const int* displacements, int root_rank,
            std::shared_ptr<const communicator> comm) GKO_NOT_COMPILED(mpi);


template <typename SendType, typename RecvType>
void all_gather(const SendType* send_buffer, const int send_count,
                RecvType* recv_buffer, const int recv_count,
                std::shared_ptr<const communicator> comm) GKO_NOT_COMPILED(mpi);


template <typename SendType, typename RecvType>
void scatter(const SendType* send_buffer, const int send_count,
             RecvType* recv_buffer, const int recv_count, int root_rank,
             std::shared_ptr<const communicator> comm) GKO_NOT_COMPILED(mpi);


template <typename SendType, typename RecvType>
void scatter(const SendType* send_buffer, const int* send_counts,
             const int* displacements, RecvType* recv_buffer,
             const int recv_count, int root_rank,
             std::shared_ptr<const communicator> comm) GKO_NOT_COMPILED(mpi);


template <typename ScanType>
void scan(const ScanType* send_buffer, ScanType* recv_buffer, int count,
          op_type op_enum, std::shared_ptr<const communicator> comm)
    GKO_NOT_COMPILED(mpi);


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
                std::shared_ptr<request> req,                           \
                std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_REDUCE);


#define GKO_DECLARE_ALLREDUCE(ReduceType)                                   \
    void all_reduce(const ReduceType* send_buffer, ReduceType* recv_buffer, \
                    int count, op_type operation,                           \
                    std::shared_ptr<request> req,                           \
                    std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_ALLREDUCE);


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


#define GKO_DECLARE_SCAN(ScanType)                                           \
    void scan(const ScanType* send_buffer, ScanType* recv_buffer, int count, \
              op_type op_enum, std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_SCAN);


}  // namespace mpi
}  // namespace gko
