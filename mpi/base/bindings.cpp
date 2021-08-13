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


bool init_finalize::is_initialized()
{
    int flag = 0;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Initialized(&flag));
    return flag;
}


bool init_finalize::is_finalized()
{
    int flag = 0;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Finalized(&flag));
    return flag;
}


init_finalize::init_finalize(int &argc, char **&argv,
                             const size_type num_threads)
{
    auto flag = is_initialized();
    if (!flag) {
        this->required_thread_support_ = MPI_THREAD_SERIALIZED;
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Init_thread(&argc, &argv, this->required_thread_support_,
                            &(this->provided_thread_support_)));
    } else {
        // GKO_MPI_INITIALIZED;
    }
}


init_finalize::~init_finalize()
{
    auto flag = is_finalized();
    if (!flag) MPI_Finalize();
}


mpi_type::mpi_type(const int count, MPI_Datatype &old)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Type_contiguous(count, old, &this->type_));
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Type_commit(&this->type_));
}


mpi_type::~mpi_type() { MPI_Type_free(&(this->type_)); }


communicator::communicator(const MPI_Comm &comm)
{
    this->comm_ = bindings::duplicate_comm(comm);
    this->size_ = bindings::get_comm_size(this->comm_);
    this->rank_ = bindings::get_comm_rank(this->comm_);
    this->local_rank_ = bindings::get_local_rank(this->comm_);
}


communicator::communicator(const MPI_Comm &comm_in, int color, int key)
{
    this->comm_ = bindings::create_comm(comm_in, color, key);
    this->size_ = bindings::get_comm_size(this->comm_);
    this->rank_ = bindings::get_comm_rank(this->comm_);
    this->local_rank_ = bindings::get_local_rank(this->comm_);
}


communicator::communicator()
{
    this->comm_ = bindings::duplicate_comm(MPI_COMM_WORLD);
    this->size_ = bindings::get_comm_size(this->comm_);
    this->rank_ = bindings::get_comm_rank(this->comm_);
}


communicator::communicator(communicator &other)
{
    this->comm_ = bindings::duplicate_comm(other.comm_);
    this->size_ = bindings::get_comm_size(this->comm_);
    this->rank_ = bindings::get_comm_rank(this->comm_);
    this->local_rank_ = bindings::get_local_rank(this->comm_);
}


communicator &communicator::operator=(const communicator &other)
{
    this->comm_ = bindings::duplicate_comm(other.comm_);
    this->size_ = bindings::get_comm_size(this->comm_);
    this->rank_ = bindings::get_comm_rank(this->comm_);
    this->local_rank_ = bindings::get_local_rank(this->comm_);
    return *this;
}


communicator::communicator(communicator &&other)
{
    this->comm_ = bindings::duplicate_comm(other.comm_);
    this->size_ = bindings::get_comm_size(this->comm_);
    this->rank_ = bindings::get_comm_rank(this->comm_);
    this->local_rank_ = bindings::get_local_rank(this->comm_);
    other.size_ = 0;
    other.rank_ = -1;
}


communicator &communicator::operator=(communicator &&other)
{
    this->comm_ = bindings::duplicate_comm(other.comm_);
    this->size_ = bindings::get_comm_size(this->comm_);
    this->rank_ = bindings::get_comm_rank(this->comm_);
    this->local_rank_ = bindings::get_local_rank(this->comm_);
    other.size_ = 0;
    other.rank_ = -1;
    return *this;
}


communicator::~communicator() { bindings::free_comm(this->comm_); }


info::info() { bindings::create_info(&this->info_); }


void info::add(std::string key, std::string value)
{
    this->key_value_[key] = value;
    bindings::add_info_key_value_pair(&this->info_, key.c_str(), value.c_str());
}


void info::remove(std::string key)
{
    bindings::remove_info_key_value_pair(&this->info_, key.c_str());
}


info::~info()
{
    if (this->info_ != MPI_INFO_NULL) bindings::free_info(&this->info_);
}


bool communicator::compare(const MPI_Comm &comm) const
{
    return bindings::compare_comm(this->comm_, comm);
}


template <typename ValueType>
window<ValueType>::window(ValueType *base, unsigned int size,
                          std::shared_ptr<const communicator> comm,
                          const int disp_unit, info input_info,
                          win_type create_type)
{
    if (create_type == win_type::create) {
        bindings::create_window(base, size, disp_unit, input_info.get(),
                                comm->get(), &this->window_);
    } else if (create_type == win_type::dynamic_create) {
        bindings::create_dynamic_window(input_info.get(), comm->get(),
                                        &this->window_);
    } else if (create_type == win_type::allocate) {
        bindings::allocate_window(size, disp_unit, input_info.get(),
                                  comm->get(), base, &this->window_);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType>
void window<ValueType>::fence(int assert)
{
    bindings::fence_window(assert, &this->window_);
}


template <typename ValueType>
void window<ValueType>::lock(int rank, int assert, lock_type lock_t)
{
    if (lock_t == lock_type::shared) {
        bindings::lock_window(MPI_LOCK_SHARED, rank, assert, &this->window_);
    } else if (lock_t == lock_type::exclusive) {
        bindings::lock_window(MPI_LOCK_EXCLUSIVE, rank, assert, &this->window_);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType>
void window<ValueType>::unlock(int rank)
{
    bindings::unlock_window(rank, &this->window_);
}


template <typename ValueType>
void window<ValueType>::lock_all(int assert)
{
    bindings::lock_all_windows(assert, &this->window_);
}


template <typename ValueType>
void window<ValueType>::unlock_all()
{
    bindings::unlock_all_windows(&this->window_);
}


template <typename ValueType>
void window<ValueType>::flush(int rank)
{
    bindings::flush_window(rank, &this->window_);
}


template <typename ValueType>
void window<ValueType>::flush_local(int rank)
{
    bindings::flush_local_window(rank, &this->window_);
}


template <typename ValueType>
void window<ValueType>::flush_all()
{
    bindings::flush_all_windows(&this->window_);
}


template <typename ValueType>
void window<ValueType>::flush_all_local()
{
    bindings::flush_all_local_windows(&this->window_);
}


template <typename ValueType>
window<ValueType>::~window()
{
    if (this->window_) {
        bindings::free_window(&this->window_);
    }
}


MPI_Op create_operation(
    const std::function<void(void *, void *, int *, MPI_Datatype *)> func,
    void *arg1, void *arg2, int *len, MPI_Datatype *type)
{
    MPI_Op operation;
    bindings::create_op(
        func.target<void(void *, void *, int *, MPI_Datatype *)>(), true,
        &operation);
    return operation;
}


double get_walltime() { return bindings::get_walltime(); }


int get_my_rank(const communicator &comm)
{
    return bindings::get_comm_rank(comm.get());
}


int get_local_rank(const communicator &comm)
{
    return bindings::get_local_rank(comm.get());
}


int get_num_ranks(const communicator &comm)
{
    return bindings::get_num_ranks(comm.get());
}


void synchronize(const communicator &comm) { bindings::barrier(comm.get()); }


void wait(std::shared_ptr<request> req, std::shared_ptr<status> status)
{
    if (status.get()) {
        bindings::wait(req->get_requests(), status->get_statuses());
    } else {
        bindings::wait(req->get_requests(), MPI_STATUS_IGNORE);
    }
}


template <typename SendType>
void send(const SendType *send_buffer, const int send_count,
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
        bindings::i_send(send_buffer, send_count, send_type, destination_rank,
                         send_tag,
                         comm ? comm->get() : communicator::get_comm_world(),
                         req->get_requests());
    }
}


template <typename RecvType>
void recv(RecvType *recv_buffer, const int recv_count, const int source_rank,
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
        bindings::i_recv(recv_buffer, recv_count, recv_type, source_rank,
                         recv_tag,
                         comm ? comm->get() : communicator::get_comm_world(),
                         req->get_requests());
    }
}


template <typename PutType>
void put(const PutType *origin_buffer, const int origin_count,
         const int target_rank, const unsigned int target_disp,
         const int target_count, window<PutType> &window,
         std::shared_ptr<request> req)
{
    auto put_type = helpers::get_mpi_type(origin_buffer[0]);
    if (!req.get()) {
        bindings::put(origin_buffer, origin_count, put_type, target_rank,
                      target_disp, target_count, put_type, window.get());
    } else {
        bindings::req_put(origin_buffer, origin_count, put_type, target_rank,
                          target_disp, target_count, put_type, window.get(),
                          req->get_requests());
    }
}


template <typename GetType>
void get(GetType *origin_buffer, const int origin_count, const int target_rank,
         const unsigned int target_disp, const int target_count,
         window<GetType> &window, std::shared_ptr<request> req)
{
    auto get_type = helpers::get_mpi_type(origin_buffer[0]);
    if (!req.get()) {
        bindings::get(origin_buffer, origin_count, get_type, target_rank,
                      target_disp, target_count, get_type, window.get());
    } else {
        bindings::req_get(origin_buffer, origin_count, get_type, target_rank,
                          target_disp, target_count, get_type, window.get(),
                          req->get_requests());
    }
}


template <typename BroadcastType>
void broadcast(BroadcastType *buffer, int count, int root_rank,
               std::shared_ptr<const communicator> comm)
{
    auto bcast_type = helpers::get_mpi_type(buffer[0]);
    bindings::broadcast(buffer, count, bcast_type, root_rank,
                        comm ? comm->get() : communicator::get_comm_world());
}


template <typename ReduceType>
void reduce(const ReduceType *send_buffer, ReduceType *recv_buffer, int count,
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
        bindings::i_reduce(send_buffer, recv_buffer, count, reduce_type,
                           operation, root_rank,
                           comm ? comm->get() : communicator::get_comm_world(),
                           req->get_requests());
    }
}


template <typename ReduceType>
void all_reduce(ReduceType *recv_buffer, int count, op_type op_enum,
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
            req->get_requests());
    }
}


template <typename ReduceType>
void all_reduce(const ReduceType *send_buffer, ReduceType *recv_buffer,
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
            comm ? comm->get() : communicator::get_comm_world(),
            req->get_requests());
    }
}


template <typename SendType, typename RecvType>
void gather(const SendType *send_buffer, const int send_count,
            RecvType *recv_buffer, const int recv_count, int root_rank,
            std::shared_ptr<const communicator> comm)
{
    auto send_type = helpers::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::get_mpi_type(recv_buffer[0]);
    bindings::gather(send_buffer, send_count, send_type, recv_buffer,
                     recv_count, recv_type, root_rank,
                     comm ? comm->get() : communicator::get_comm_world());
}


template <typename SendType, typename RecvType>
void gather(const SendType *send_buffer, const int send_count,
            RecvType *recv_buffer, const int *recv_counts,
            const int *displacements, int root_rank,
            std::shared_ptr<const communicator> comm)
{
    auto send_type = helpers::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::get_mpi_type(recv_buffer[0]);
    bindings::gatherv(send_buffer, send_count, send_type, recv_buffer,
                      recv_counts, displacements, recv_type, root_rank,
                      comm ? comm->get() : communicator::get_comm_world());
}


template <typename SendType, typename RecvType>
void all_gather(const SendType *send_buffer, const int send_count,
                RecvType *recv_buffer, const int recv_count,
                std::shared_ptr<const communicator> comm)
{
    auto send_type = helpers::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::get_mpi_type(recv_buffer[0]);
    bindings::all_gather(send_buffer, send_count, send_type, recv_buffer,
                         recv_count, recv_type,
                         comm ? comm->get() : communicator::get_comm_world());
}


template <typename SendType, typename RecvType>
void scatter(const SendType *send_buffer, const int send_count,
             RecvType *recv_buffer, const int recv_count, int root_rank,
             std::shared_ptr<const communicator> comm)
{
    auto send_type = helpers::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::get_mpi_type(recv_buffer[0]);
    bindings::scatter(send_buffer, send_count, send_type, recv_buffer,
                      recv_count, recv_type, root_rank,
                      comm ? comm->get() : communicator::get_comm_world());
}


template <typename SendType, typename RecvType>
void scatter(const SendType *send_buffer, const int *send_counts,
             const int *displacements, RecvType *recv_buffer,
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
void all_to_all(RecvType *recv_buffer, const int recv_count,
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
            comm ? comm->get() : communicator::get_comm_world(),
            req->get_requests());
    }
}


template <typename SendType, typename RecvType>
void all_to_all(const SendType *send_buffer, const int send_count,
                RecvType *recv_buffer, const int recv_count,
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
            comm ? comm->get() : communicator::get_comm_world(),
            req->get_requests());
    }
}


template <typename SendType, typename RecvType>
void all_to_all(const SendType *send_buffer, const int *send_counts,
                const int *send_offsets, RecvType *recv_buffer,
                const int *recv_counts, const int *recv_offsets,
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
            comm ? comm->get() : communicator::get_comm_world(),
            req->get_requests());
    }
}


template <typename ScanType>
void scan(const ScanType *send_buffer, ScanType *recv_buffer, int count,
          op_type op_enum, std::shared_ptr<const communicator> comm)
{
    auto operation = helpers::get_operation<ScanType>(op_enum);
    auto scan_type = helpers::get_mpi_type(recv_buffer[0]);
    bindings::scan(send_buffer, recv_buffer, count, scan_type, operation,
                   comm ? comm->get() : communicator::get_comm_world());
}


#define GKO_DECLARE_WINDOW(ValueType) class window<ValueType>

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_WINDOW);


#define GKO_DECLARE_SEND(SendType)                               \
    void send(const SendType *send_buffer, const int send_count, \
              const int destination_rank, const int send_tag,    \
              std::shared_ptr<request> req,                      \
              std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_SEND);


#define GKO_DECLARE_RECV(RecvType)                                          \
    void recv(RecvType *recv_buffer, const int recv_count,                  \
              const int source_rank, const int recv_tag,                    \
              std::shared_ptr<request> req, std::shared_ptr<status> status, \
              std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_RECV);


#define GKO_DECLARE_PUT(PutType)                                    \
    void put(const PutType *origin_buffer, const int origin_count,  \
             const int target_rank, const unsigned int target_disp, \
             const int target_count, window<PutType> &window,       \
             std::shared_ptr<request> req)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_PUT);


#define GKO_DECLARE_GET(GetType)                                    \
    void get(GetType *origin_buffer, const int origin_count,        \
             const int target_rank, const unsigned int target_disp, \
             const int target_count, window<GetType> &window,       \
             std::shared_ptr<request> req)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_GET);


#define GKO_DECLARE_BCAST(BroadcastType)                            \
    void broadcast(BroadcastType *buffer, int count, int root_rank, \
                   std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_BCAST);


#define GKO_DECLARE_REDUCE(ReduceType)                                  \
    void reduce(const ReduceType *send_buffer, ReduceType *recv_buffer, \
                int count, op_type operation, int root_rank,            \
                std::shared_ptr<const communicator> comm,               \
                std::shared_ptr<request> req)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_REDUCE);


#define GKO_DECLARE_ALLREDUCE1(ReduceType)                                 \
    void all_reduce(ReduceType *recv_buffer, int count, op_type operation, \
                    std::shared_ptr<const communicator> comm,              \
                    std::shared_ptr<request> req)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_ALLREDUCE1);


#define GKO_DECLARE_ALLREDUCE2(ReduceType)                                  \
    void all_reduce(const ReduceType *send_buffer, ReduceType *recv_buffer, \
                    int count, op_type operation,                           \
                    std::shared_ptr<const communicator> comm,               \
                    std::shared_ptr<request> req)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_ALLREDUCE2);


#define GKO_DECLARE_GATHER1(SendType, RecvType)                             \
    void gather(const SendType *send_buffer, const int send_count,          \
                RecvType *recv_buffer, const int recv_count, int root_rank, \
                std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GATHER1);


#define GKO_DECLARE_GATHER2(SendType, RecvType)                    \
    void gather(const SendType *send_buffer, const int send_count, \
                RecvType *recv_buffer, const int *recv_counts,     \
                const int *displacements, int root_rank,           \
                std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GATHER2);


#define GKO_DECLARE_ALLGATHER(SendType, RecvType)                      \
    void all_gather(const SendType *send_buffer, const int send_count, \
                    RecvType *recv_buffer, const int recv_count,       \
                    std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ALLGATHER);


#define GKO_DECLARE_SCATTER1(SendType, RecvType)                             \
    void scatter(const SendType *send_buffer, const int send_count,          \
                 RecvType *recv_buffer, const int recv_count, int root_rank, \
                 std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SCATTER1);


#define GKO_DECLARE_SCATTER2(SendType, RecvType)                      \
    void scatter(const SendType *send_buffer, const int *send_counts, \
                 const int *displacements, RecvType *recv_buffer,     \
                 const int recv_count, int root_rank,                 \
                 std::shared_ptr<const communicator> comm)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SCATTER2);


#define GKO_DECLARE_ALL_TO_ALL1(RecvType)                        \
    void all_to_all(RecvType *recv_buffer, const int recv_count, \
                    std::shared_ptr<const communicator> comm,    \
                    std::shared_ptr<request> req)

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_ALL_TO_ALL1);


#define GKO_DECLARE_ALL_TO_ALL2(SendType, RecvType)                    \
    void all_to_all(const SendType *send_buffer, const int send_count, \
                    RecvType *recv_buffer, const int recv_count,       \
                    std::shared_ptr<const communicator> comm,          \
                    std::shared_ptr<request> req)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ALL_TO_ALL2);


#define GKO_DECLARE_ALL_TO_ALL_V(SendType, RecvType)                     \
    void all_to_all(const SendType *send_buffer, const int *send_counts, \
                    const int *send_offsets, RecvType *recv_buffer,      \
                    const int *recv_counts, const int *recv_offsets,     \
                    const int stride,                                    \
                    std::shared_ptr<const communicator> comm,            \
                    std::shared_ptr<request> req)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ALL_TO_ALL_V);


#define GKO_DECLARE_SCAN(ScanType)                                           \
    void scan(const ScanType *send_buffer, ScanType *recv_buffer, int count, \
              op_type op_enum, std::shared_ptr<const communicator> comm)
GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_SCAN);


}  // namespace mpi
}  // namespace gko
