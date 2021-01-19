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


#include "mpi/base/mpi_bindings.hpp"

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "mpi/base/mpi_helpers.hpp"


namespace gko {


bool mpi::init_finalize::is_initialized()
{
    int flag = 0;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Initialized(&flag));
    return flag;
}


bool mpi::init_finalize::is_finalized()
{
    int flag = 0;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Finalized(&flag));
    return flag;
}


mpi::init_finalize::init_finalize(int &argc, char **&argv,
                                  const size_type num_threads)
{
    auto flag = is_initialized();
    if (!flag) {
        this->required_thread_support_ = MPI_THREAD_SERIALIZED;
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Init_thread(&argc, &argv, this->required_thread_support_,
                            &(this->provided_thread_support_)));
    } else {
        GKO_MPI_INITIALIZED;
    }
}


mpi::init_finalize::~init_finalize() noexcept(false)
{
    auto flag = is_finalized();
    if (!flag) {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Finalize());
    } else {
        GKO_MPI_FINALIZED;
    }
}


mpi::communicator::communicator(const MPI_Comm &comm)
{
    this->comm_ = bindings::mpi::duplicate_comm(comm);
}


mpi::communicator::communicator(const MPI_Comm &comm_in, int color, int key)
{
    this->comm_ = bindings::mpi::create_comm(comm_in, color, key);
}


mpi::communicator::~communicator() { bindings::mpi::free_comm(this->comm_); }


mpi::info::info() { bindings::mpi::create_info(&this->info_); }


void mpi::info::add(std::string key, std::string value)
{
    this->key_value_[key] = value;
    bindings::mpi::add_info_key_value_pair(&this->info_, key.c_str(),
                                           value.c_str());
}


void mpi::info::remove(std::string key)
{
    bindings::mpi::remove_info_key_value_pair(&this->info_, key.c_str());
}


mpi::info::~info() { bindings::mpi::free_info(&this->info_); }


bool mpi::communicator::compare(const MPI_Comm &comm) const
{
    return bindings::mpi::compare_comm(this->comm_, comm);
}


template <typename ValueType>
mpi::window<ValueType>::window(ValueType *base, unsigned int size,
                               const int disp_unit, MPI_Info info,
                               const MPI_Comm &comm, win_type create_type)
{
    MPI_Win window;
    if (create_type == win_type::create) {
        bindings::mpi::create_window(base, size, disp_unit, info, comm,
                                     &window);
    } else if (create_type == win_type::dynamic_create) {
        bindings::mpi::create_dynamic_window(info, comm, &window);
    } else if (create_type == win_type::allocate) {
        bindings::mpi::allocate_window(size, disp_unit, info, comm, base,
                                       &window);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
    this->window_ = window;
}


template <typename ValueType>
void mpi::window<ValueType>::fence(int assert)
{
    bindings::mpi::fence_window(assert, &this->window_);
}


template <typename ValueType>
void mpi::window<ValueType>::lock(int rank, int assert, lock_type lock_t)
{
    if (lock_t == lock_type::shared) {
        bindings::mpi::lock_window(MPI_LOCK_SHARED, rank, assert,
                                   &this->window_);
    } else if (lock_t == lock_type::exclusive) {
        bindings::mpi::lock_window(MPI_LOCK_EXCLUSIVE, rank, assert,
                                   &this->window_);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType>
void mpi::window<ValueType>::unlock(int rank)
{
    bindings::mpi::unlock_window(rank, &this->window_);
}


template <typename ValueType>
void mpi::window<ValueType>::lock_all(int assert)
{
    bindings::mpi::lock_all_windows(assert, &this->window_);
}


template <typename ValueType>
void mpi::window<ValueType>::unlock_all()
{
    bindings::mpi::unlock_all_windows(&this->window_);
}


template <typename ValueType>
void mpi::window<ValueType>::flush(int rank)
{
    bindings::mpi::flush_window(rank, &this->window_);
}


template <typename ValueType>
void mpi::window<ValueType>::flush_local(int rank)
{
    bindings::mpi::flush_local_window(rank, &this->window_);
}


template <typename ValueType>
void mpi::window<ValueType>::flush_all()
{
    bindings::mpi::flush_all_windows(&this->window_);
}


template <typename ValueType>
void mpi::window<ValueType>::flush_all_local()
{
    bindings::mpi::flush_all_local_windows(&this->window_);
}


template <typename ValueType>
mpi::window<ValueType>::~window()
{
    if (this->window_ != nullptr) {
        bindings::mpi::free_window(&this->window_);
    }
}


MpiExecutor::request_manager<MPI_Request> MpiExecutor::create_requests_array(
    int size)
{
    return MpiExecutor::request_manager<MPI_Request>{
        new MPI_Request[size], [size](MPI_Request *req) { delete req; }};
}


void MpiExecutor::synchronize_communicator(const MPI_Comm &comm) const
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Barrier(comm));
}


void MpiExecutor::synchronize() const
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Barrier(MPI_COMM_WORLD));
    this->sub_executor_->synchronize();
}


void MpiExecutor::set_communicator(const MPI_Comm &comm)
{
    comm_ = mpi::communicator(comm);
}


void MpiExecutor::wait(MPI_Request *req, MPI_Status *status)
{
    if (status) {
        bindings::mpi::wait(req, status);
    } else {
        bindings::mpi::wait(req, MPI_STATUS_IGNORE);
    }
}


std::shared_ptr<MpiExecutor> MpiExecutor::create(
    std::shared_ptr<Executor> sub_executor)
{
    return std::shared_ptr<MpiExecutor>(new MpiExecutor(sub_executor),
                                        [](MpiExecutor *exec) { delete exec; });
}


std::shared_ptr<MpiExecutor> MpiExecutor::create(
    std::shared_ptr<Executor> sub_executor, const MPI_Comm &comm)
{
    return std::shared_ptr<MpiExecutor>(new MpiExecutor(sub_executor, comm),
                                        [](MpiExecutor *exec) { delete exec; });
}


double MpiExecutor::get_walltime() const
{
    double wtime = 0.0;
    wtime = MPI_Wtime();
    return wtime;
}


int MpiExecutor::get_my_rank(const MPI_Comm &comm) const
{
    auto my_rank = 0;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_rank(comm, &my_rank));
    return my_rank;
}


int MpiExecutor::get_local_rank(const MPI_Comm &comm) const
{
    MPI_Comm local_comm;
    int rank;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0,
                                                 MPI_INFO_NULL, &local_comm));
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_rank(local_comm, &rank));
    MPI_Comm_free(&local_comm);
    return rank;
}


int MpiExecutor::get_num_ranks(const MPI_Comm &comm) const
{
    int size = 1;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_size(this->get_communicator(), &size));
    return size;
}


MPI_Op MpiExecutor::create_operation(
    const std::function<void(void *, void *, int *, MPI_Datatype *)> func,
    void *arg1, void *arg2, int *len, MPI_Datatype *type)
{
    MPI_Op operation;
    bindings::mpi::create_op(
        func.target<void(void *, void *, int *, MPI_Datatype *)>(), true,
        &operation);
    return operation;
}


template <typename SendType>
void MpiExecutor::send(const SendType *send_buffer, const int send_count,
                       const int destination_rank, const int send_tag,
                       MPI_Request *req) const
{
    auto send_type = helpers::mpi::get_mpi_type(send_buffer[0]);
    if (!req) {
        bindings::mpi::send(send_buffer, send_count, send_type,
                            destination_rank, send_tag,
                            this->get_communicator());
    } else {
        bindings::mpi::i_send(send_buffer, send_count, send_type,
                              destination_rank, send_tag,
                              this->get_communicator(), req);
    }
}


template <typename RecvType>
void MpiExecutor::recv(RecvType *recv_buffer, const int recv_count,
                       const int source_rank, const int recv_tag,
                       MPI_Request *req) const
{
    auto recv_type = helpers::mpi::get_mpi_type(recv_buffer[0]);
    if (!req) {
        bindings::mpi::recv(recv_buffer, recv_count, recv_type, source_rank,
                            recv_tag, this->get_communicator(),
                            (this->mpi_status_.get() ? this->mpi_status_.get()
                                                     : MPI_STATUS_IGNORE));
    } else {
        bindings::mpi::i_recv(recv_buffer, recv_count, recv_type, source_rank,
                              recv_tag, this->get_communicator(), req);
    }
}


template <typename PutType>
void MpiExecutor::put(const PutType *origin_buffer, const int origin_count,
                      const int target_rank, const unsigned int target_disp,
                      const int target_count, MPI_Win window,
                      MPI_Request *req) const
{
    auto put_type = helpers::mpi::get_mpi_type(origin_buffer[0]);
    if (!req) {
        bindings::mpi::put(origin_buffer, origin_count, put_type, target_rank,
                           target_disp, target_count, put_type, window);
    } else {
        bindings::mpi::req_put(origin_buffer, origin_count, put_type,
                               target_rank, target_disp, target_count, put_type,
                               window, req);
    }
}


template <typename GetType>
void MpiExecutor::get(GetType *origin_buffer, const int origin_count,
                      const int target_rank, const unsigned int target_disp,
                      const int target_count, MPI_Win window,
                      MPI_Request *req) const
{
    auto get_type = helpers::mpi::get_mpi_type(origin_buffer[0]);
    if (!req) {
        bindings::mpi::get(origin_buffer, origin_count, get_type, target_rank,
                           target_disp, target_count, get_type, window);
    } else {
        bindings::mpi::req_get(origin_buffer, origin_count, get_type,
                               target_rank, target_disp, target_count, get_type,
                               window, req);
    }
}


template <typename BroadcastType>
void MpiExecutor::broadcast(BroadcastType *buffer, int count,
                            int root_rank) const
{
    auto bcast_type = helpers::mpi::get_mpi_type(buffer[0]);
    bindings::mpi::broadcast(buffer, count, bcast_type, root_rank,
                             this->get_communicator());
}


template <typename ReduceType>
void MpiExecutor::reduce(const ReduceType *send_buffer, ReduceType *recv_buffer,
                         int count, mpi::op_type op_enum, int root_rank,
                         MPI_Request *req) const
{
    auto operation = helpers::mpi::get_operation<ReduceType>(op_enum);
    auto reduce_type = helpers::mpi::get_mpi_type(send_buffer[0]);
    if (!req) {
        bindings::mpi::reduce(send_buffer, recv_buffer, count, reduce_type,
                              operation, root_rank, this->get_communicator());
    } else {
        bindings::mpi::i_reduce(send_buffer, recv_buffer, count, reduce_type,
                                operation, root_rank, this->get_communicator(),
                                req);
    }
}


template <typename ReduceType>
void MpiExecutor::all_reduce(const ReduceType *send_buffer,
                             ReduceType *recv_buffer, int count,
                             mpi::op_type op_enum, MPI_Request *req) const
{
    auto operation = helpers::mpi::get_operation<ReduceType>(op_enum);
    auto reduce_type = helpers::mpi::get_mpi_type(send_buffer[0]);
    if (!req) {
        bindings::mpi::all_reduce(send_buffer, recv_buffer, count, reduce_type,
                                  operation, this->get_communicator());
    } else {
        bindings::mpi::i_all_reduce(send_buffer, recv_buffer, count,
                                    reduce_type, operation,
                                    this->get_communicator(), req);
    }
}


template <typename SendType, typename RecvType>
void MpiExecutor::gather(const SendType *send_buffer, const int send_count,
                         RecvType *recv_buffer, const int recv_count,
                         int root_rank) const
{
    auto send_type = helpers::mpi::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::mpi::get_mpi_type(recv_buffer[0]);
    bindings::mpi::gather(send_buffer, send_count, send_type, recv_buffer,
                          recv_count, recv_type, root_rank,
                          this->get_communicator());
}


template <typename SendType, typename RecvType>
void MpiExecutor::gather(const SendType *send_buffer, const int send_count,
                         RecvType *recv_buffer, const int *recv_counts,
                         const int *displacements, int root_rank) const
{
    auto send_type = helpers::mpi::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::mpi::get_mpi_type(recv_buffer[0]);
    bindings::mpi::gatherv(send_buffer, send_count, send_type, recv_buffer,
                           recv_counts, displacements, recv_type, root_rank,
                           this->get_communicator());
}


template <typename SendType, typename RecvType>
void MpiExecutor::scatter(const SendType *send_buffer, const int send_count,
                          RecvType *recv_buffer, const int recv_count,
                          int root_rank) const
{
    auto send_type = helpers::mpi::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::mpi::get_mpi_type(recv_buffer[0]);
    bindings::mpi::scatter(send_buffer, send_count, send_type, recv_buffer,
                           recv_count, recv_type, root_rank,
                           this->get_communicator());
}


template <typename SendType, typename RecvType>
void MpiExecutor::scatter(const SendType *send_buffer, const int *send_counts,
                          const int *displacements, RecvType *recv_buffer,
                          const int recv_count, int root_rank) const
{
    auto send_type = helpers::mpi::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::mpi::get_mpi_type(recv_buffer[0]);
    bindings::mpi::scatterv(send_buffer, send_counts, displacements, send_type,
                            recv_buffer, recv_count, recv_type, root_rank,
                            this->get_communicator());
}


#define GKO_DECLARE_WINDOW(ValueType) class mpi::window<ValueType>

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_WINDOW);


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


#define GKO_DECLARE_PUT(PutType)                               \
    void MpiExecutor::put(                                     \
        const PutType *origin_buffer, const int origin_count,  \
        const int target_rank, const unsigned int target_disp, \
        const int target_count, MPI_Win window, MPI_Request *req) const

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_PUT);


#define GKO_DECLARE_GET(GetType)                                               \
    void MpiExecutor::get(                                                     \
        GetType *origin_buffer, const int origin_count, const int target_rank, \
        const unsigned int target_disp, const int target_count,                \
        MPI_Win window, MPI_Request *req) const

GKO_INSTANTIATE_FOR_EACH_POD_TYPE(GKO_DECLARE_GET);


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
