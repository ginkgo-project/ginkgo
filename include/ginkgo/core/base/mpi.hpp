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

#ifndef GKO_PUBLIC_CORE_BASE_MPI_HPP_
#define GKO_PUBLIC_CORE_BASE_MPI_HPP_


#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils_helper.hpp>


#if GKO_HAVE_MPI


#include <mpi.h>


namespace gko {
namespace mpi {


/*
 * This enum specifies the threading type to be used when creating an MPI
 * environment.
 */
enum class thread_type {
    serialized = MPI_THREAD_SERIALIZED,
    funneled = MPI_THREAD_FUNNELED,
    single = MPI_THREAD_SINGLE,
    multiple = MPI_THREAD_MULTIPLE
};


namespace detail {


template <typename T>
struct mpi_type_impl {
    constexpr static MPI_Datatype get_type() { return MPI_DATATYPE_NULL; }
};


template <>
constexpr MPI_Datatype mpi_type_impl<char>::get_type()
{
    return MPI_CHAR;
}


template <>
constexpr MPI_Datatype mpi_type_impl<unsigned char>::get_type()
{
    return MPI_UNSIGNED_CHAR;
}


template <>
constexpr MPI_Datatype mpi_type_impl<unsigned>::get_type()
{
    return MPI_UNSIGNED;
}


template <>
constexpr MPI_Datatype mpi_type_impl<int>::get_type()
{
    return MPI_INT;
}


template <>
constexpr MPI_Datatype mpi_type_impl<unsigned short>::get_type()
{
    return MPI_UNSIGNED_SHORT;
}


template <>
constexpr MPI_Datatype mpi_type_impl<unsigned long>::get_type()
{
    return MPI_UNSIGNED_LONG;
}


template <>
constexpr MPI_Datatype mpi_type_impl<long>::get_type()
{
    return MPI_LONG;
}


template <>
constexpr MPI_Datatype mpi_type_impl<float>::get_type()
{
    return MPI_FLOAT;
}


template <>
constexpr MPI_Datatype mpi_type_impl<double>::get_type()
{
    return MPI_DOUBLE;
}


template <>
constexpr MPI_Datatype mpi_type_impl<long double>::get_type()
{
    return MPI_LONG_DOUBLE;
}


template <>
constexpr MPI_Datatype mpi_type_impl<std::complex<float>>::get_type()
{
    return MPI_C_COMPLEX;
}


template <>
constexpr MPI_Datatype mpi_type_impl<std::complex<double>>::get_type()
{
    return MPI_C_DOUBLE_COMPLEX;
}

template <typename T>
inline const T* in_place()
{
    return reinterpret_cast<const T*>(MPI_IN_PLACE);
}


}  // namespace detail

/*
 * Class that sets up and finalizes the MPI exactly once per program execution.
 * using the singleton pattern. This must be called before any of the MPI
 * functions.
 */
class init_finalize {
public:
    static bool is_finalized()
    {
        int flag = 0;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Finalized(&flag));
        return flag;
    }

    static bool is_initialized()
    {
        int flag = 0;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Initialized(&flag));
        return flag;
    }

    init_finalize(int& argc, char**& argv,
                  const thread_type thread_t = thread_type::serialized)
    {
        this->required_thread_support_ = static_cast<int>(thread_t);
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Init_thread(&argc, &argv, this->required_thread_support_,
                            &(this->provided_thread_support_)));
    }

    init_finalize() = delete;

    ~init_finalize() { MPI_Finalize(); }

    int get_provided_thread_support() { return provided_thread_support_; }

private:
    int required_thread_support_;
    int provided_thread_support_;
};


namespace {


class comm_deleter {
public:
    using pointer = MPI_Comm*;
    void operator()(pointer comm) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_free(comm));
    }
};


}  // namespace


/**
 * A communicator class that takes in the given communicator and duplicates it
 * for our purposes. As the class or object goes out of scope, the communicator
 * is freed.
 */
class communicator {
public:
    communicator(MPI_Comm comm)
    {
        this->comm_ =
            comm_manager(new MPI_Comm(comm), null_deleter<MPI_Comm>{});
        this->size_ = get_num_ranks();
        this->rank_ = get_my_rank();
        this->node_local_rank_ = get_node_local_rank();
    }

    communicator(const MPI_Comm& comm, int color, int key)
    {
        MPI_Comm comm_out;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_split(comm, color, key, &comm_out));
        this->comm_ = comm_manager(new MPI_Comm(comm_out), comm_deleter{});
        this->size_ = get_num_ranks();
        this->rank_ = get_my_rank();
        this->node_local_rank_ = get_node_local_rank();
    }

    communicator(communicator& other)
    {
        MPI_Comm comm;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_dup(other.get(), &comm));
        this->comm_ = comm_manager(new MPI_Comm(comm), comm_deleter{});
        this->size_ = other.size_;
        this->rank_ = other.rank_;
        this->node_local_rank_ = other.node_local_rank_;
    }

    communicator& operator=(const communicator& other)
    {
        MPI_Comm comm;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_dup(other.get(), &comm));
        this->comm_ = comm_manager(new MPI_Comm(comm), comm_deleter{});
        this->size_ = other.size_;
        this->rank_ = other.rank_;
        this->node_local_rank_ = other.node_local_rank_;
        return *this;
    }

    communicator(communicator&& other)
    {
        if (other.is_owning()) {
            this->comm_ = std::move(other.comm_);
            this->size_ = other.size_;
            this->rank_ = other.rank_;
            this->node_local_rank_ = other.node_local_rank_;
            other.size_ = 0;
            other.rank_ = -1;
        } else {
            // If we don't own the communicator, then we can't move from it.
            GKO_NOT_SUPPORTED(other);
        }
    }

    communicator& operator=(communicator&& other)
    {
        if (other.is_owning()) {
            this->comm_ = std::move(other.comm_);
            this->size_ = other.size_;
            this->rank_ = other.rank_;
            this->node_local_rank_ = other.node_local_rank_;
            other.size_ = 0;
            other.rank_ = -1;
        } else {
            // If we don't own the communicator, then we can't move from it.
            GKO_NOT_SUPPORTED(other);
        }
        return *this;
    }

    static communicator duplicate(const MPI_Comm& comm_in)
    {
        MPI_Comm comm;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_dup(comm_in, &comm));
        communicator comm_out(comm);
        return comm_out;
    }

    const MPI_Comm& get() const { return *(this->comm_.get()); }

    int size() const { return size_; }

    int rank() const { return rank_; };

    int node_local_rank() const { return node_local_rank_; };

    bool operator==(const communicator& rhs) { return compare(rhs.get()); }

    bool is_owning()
    {
        return comm_.get_deleter().target_type() == typeid(comm_deleter);
    }

private:
    using comm_manager =
        std::unique_ptr<MPI_Comm, std::function<void(MPI_Comm*)>>;
    comm_manager comm_;
    int size_{};
    int rank_{};
    int node_local_rank_{};

    int get_my_rank()
    {
        int my_rank = 0;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_rank(get(), &my_rank));
        return my_rank;
    }

    int get_node_local_rank()
    {
        MPI_Comm local_comm;
        int rank;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_split_type(
            get(), MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm));
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_rank(local_comm, &rank));
        MPI_Comm_free(&local_comm);
        return rank;
    }

    int get_num_ranks()
    {
        int size = 1;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_size(get(), &size));
        return size;
    }

    bool compare(const MPI_Comm& other) const
    {
        int flag;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_compare(get(), other, &flag));
        return flag;
    }
};


/**
 * Get the rank in the communicator of the calling process.
 *
 * @param comm  the communicator
 */
inline double get_walltime() { return MPI_Wtime(); }


/**
 * This function is used to synchronize between the ranks of a given
 * communicator.
 *
 * @param comm  the communicator
 */
inline void synchronize(const communicator& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Barrier(comm.get()));
}


/**
 * Allows a rank to wait on a particular request handle.
 *
 * @param req  The request to wait on.
 * @param status  The status variable that can be queried.
 */
inline MPI_Status wait(MPI_Request& req)
{
    MPI_Status status;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Wait(&req, &status));
    return status;
}


/**
 * Allows a rank to wait on a particular request handle.
 *
 * @param req  The request to wait on.
 * @param status  The status variable that can be queried.
 */
inline std::vector<MPI_Status> wait_all(std::vector<MPI_Request>& req)
{
    std::vector<MPI_Status> status;
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Waitall(req.size(), req.data(), status.data()));
    return status;
}


/**
 * This class wraps the MPI_Window class with RAII functionality. Different
 * create and lock type methods are setup with enums.
 *
 * MPI_Window is primarily used for one sided communication and this class
 * provides functionalities to fence, lock, unlock and flush the communication
 * buffers.
 */
template <typename ValueType>
class window {
public:
    enum class win_type { allocate = 1, create = 2, dynamic_create = 3 };
    enum class lock_type { shared = 1, exclusive = 2 };

    window() : window_(MPI_WIN_NULL) {}
    window(const window& other) = delete;
    window& operator=(const window& other) = delete;
    window(window&& other) : window_{std::exchange(other.window_, MPI_WIN_NULL)}
    {}
    window& operator=(window&& other)
    {
        window_ = std::exchange(other.window_, MPI_WIN_NULL);
    }

    window(ValueType* base, unsigned int size, const communicator& comm,
           const int disp_unit = sizeof(ValueType),
           MPI_Info input_info = MPI_INFO_NULL,
           win_type create_type = win_type::create)
    {
        if (create_type == win_type::create) {
            GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_create(
                base, size, disp_unit, input_info, comm.get(), &this->window_));
        } else if (create_type == win_type::dynamic_create) {
            GKO_ASSERT_NO_MPI_ERRORS(
                MPI_Win_create_dynamic(input_info, comm.get(), &this->window_));
        } else if (create_type == win_type::allocate) {
            GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_allocate(
                size, disp_unit, input_info, comm.get(), base, &this->window_));
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    MPI_Win get() { return this->window_; }

    void fence(int assert = 0)
    {
        if (&this->window_) {
            GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_fence(assert, this->window_));
        }
    }

    void lock(int rank, int assert = 0, lock_type lock_t = lock_type::shared)
    {
        if (lock_t == lock_type::shared) {
            GKO_ASSERT_NO_MPI_ERRORS(
                MPI_Win_lock(MPI_LOCK_SHARED, rank, assert, this->window_));
        } else if (lock_t == lock_type::exclusive) {
            GKO_ASSERT_NO_MPI_ERRORS(
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, assert, this->window_));
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    void unlock(int rank)
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_unlock(rank, this->window_));
    }

    void lock_all(int assert = 0)
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_lock_all(assert, this->window_));
    }

    void unlock_all()
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_unlock_all(this->window_));
    }

    void flush(int rank)
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_flush(rank, this->window_));
    }

    void flush_local(int rank)
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_flush_local(rank, this->window_));
    }

    void flush_all()
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_flush_all(this->window_));
    }

    void flush_all_local()
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_flush_local_all(this->window_));
    }

    ~window()
    {
        if (this->window_ && this->window_ != MPI_WIN_NULL) {
            MPI_Win_free(&this->window_);
        }
    }

private:
    MPI_Win window_;
};


/**
 * Send (Blocking) data from calling process to destination rank.
 *
 * @param send_buffer  the buffer to send
 * @param send_count  the number of elements to send
 * @param destination_rank  the rank to send the data to
 * @param send_tag  the tag for the send call
 * @param comm  the communicator
 */
template <typename SendType>
inline void send(const SendType* send_buffer, const int send_count,
                 const int destination_rank, const int send_tag,
                 const communicator& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Send(
        send_buffer, send_count, detail::mpi_type_impl<SendType>::get_type(),
        destination_rank, send_tag, comm.get()));
}


/**
 * Send (Non-blocking, Immediate return) data from calling process to
 * destination rank.
 *
 * @param send_buffer  the buffer to send
 * @param send_count  the number of elements to send
 * @param destination_rank  the rank to send the data to
 * @param send_tag  the tag for the send call
 * @param comm  the communicator
 *
 * @return  the request handle for the send call
 */
template <typename SendType>
inline MPI_Request i_send(const SendType* send_buffer, const int send_count,
                          const int destination_rank, const int send_tag,
                          const communicator& comm)
{
    MPI_Request req;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Isend(
        send_buffer, send_count, detail::mpi_type_impl<SendType>::get_type(),
        destination_rank, send_tag, comm.get(), &req));
    return req;
}


/**
 * Receive data from source rank.
 *
 * @param recv_buffer  the buffer to send
 * @param recv_count  the number of elements to send
 * @param source_rank  the rank to send the data to
 * @param recv_tag  the tag for the send call
 * @param comm  the communicator
 *
 * @return  the status of completion of this call
 */
template <typename RecvType>
inline MPI_Status recv(RecvType* recv_buffer, const int recv_count,
                       const int source_rank, const int recv_tag,
                       const communicator& comm)
{
    MPI_Status status;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Recv(
        recv_buffer, recv_count, detail::mpi_type_impl<RecvType>::get_type(),
        source_rank, recv_tag, comm.get(), &status));
    return status;
}


/**
 * Receive (Non-blocking, Immediate return) data from source rank.
 *
 * @param recv_buffer  the buffer to send
 * @param recv_count  the number of elements to send
 * @param source_rank  the rank to send the data to
 * @param recv_tag  the tag for the send call
 * @param req  the request handle for the send call
 * @param comm  the communicator
 *
 * @return  the request handle for the send call
 */
template <typename RecvType>
inline MPI_Request i_recv(RecvType* recv_buffer, const int recv_count,
                          const int source_rank, const int recv_tag,
                          const communicator& comm)
{
    MPI_Request req;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Irecv(
        recv_buffer, recv_count, detail::mpi_type_impl<RecvType>::get_type(),
        source_rank, recv_tag, comm.get(), &req));
    return req;
}


/**
 * Put data into the target window.
 *
 * @param origin_buffer  the buffer to send
 * @param origin_count  the number of elements to put
 * @param target_rank  the rank to put the data to
 * @param target_disp  the displacement at the target window
 * @param target_count  the request handle for the send call
 * @param window  the window to put the data into
 */
template <typename PutType>
inline void put(const PutType* origin_buffer, const int origin_count,
                const int target_rank, const unsigned int target_disp,
                const int target_count, window<PutType>& window)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Put(
        origin_buffer, origin_count, detail::mpi_type_impl<PutType>::get_type(),
        target_rank, target_disp, target_count,
        detail::mpi_type_impl<PutType>::get_type(), window.get()));
}


/**
 * Put data into the target window.
 *
 * @param origin_buffer  the buffer to send
 * @param origin_count  the number of elements to put
 * @param target_rank  the rank to put the data to
 * @param target_disp  the displacement at the target window
 * @param target_count  the request handle for the send call
 * @param window  the window to put the data into
 *
 * @return  the request handle for the send call
 */
template <typename PutType>
inline MPI_Request r_put(const PutType* origin_buffer, const int origin_count,
                         const int target_rank, const unsigned int target_disp,
                         const int target_count, window<PutType>& window)
{
    MPI_Request req;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Rput(
        origin_buffer, origin_count, detail::mpi_type_impl<PutType>::get_type(),
        target_rank, target_disp, target_count,
        detail::mpi_type_impl<PutType>::get_type(), window.get(), &req));
    return req;
}


/**
 * Get data from the target window.
 *
 * @param origin_buffer  the buffer to send
 * @param origin_count  the number of elements to get
 * @param target_rank  the rank to get the data from
 * @param target_disp  the displacement at the target window
 * @param target_count  the request handle for the send call
 * @param window  the window to put the data into
 */
template <typename GetType>
inline void get(GetType* origin_buffer, const int origin_count,
                const int target_rank, const unsigned int target_disp,
                const int target_count, window<GetType>& window)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Get(
        origin_buffer, origin_count, detail::mpi_type_impl<GetType>::get_type(),
        target_rank, target_disp, target_count,
        detail::mpi_type_impl<GetType>::get_type(), window.get()));
}


/**
 * Get data (with handle) from the target window.
 *
 * @param origin_buffer  the buffer to send
 * @param origin_count  the number of elements to get
 * @param target_rank  the rank to get the data from
 * @param target_disp  the displacement at the target window
 * @param target_count  the request handle for the send call
 * @param window  the window to put the data into
 *
 * @return  the request handle for the send call
 */
template <typename GetType>
inline MPI_Request r_get(GetType* origin_buffer, const int origin_count,
                         const int target_rank, const unsigned int target_disp,
                         const int target_count, window<GetType>& window)
{
    MPI_Request req;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Rget(
        origin_buffer, origin_count, detail::mpi_type_impl<GetType>::get_type(),
        target_rank, target_disp, target_count,
        detail::mpi_type_impl<GetType>::get_type(), window, &req));
    return req;
}


/**
 * Broadcast data from calling process to all ranks in the communicator
 *
 * @param buffer  the buffer to broadcsat
 * @param count  the number of elements to broadcast
 * @param root_rank  the rank to broadcast from
 * @param comm  the communicator
 */
template <typename BroadcastType>
inline void broadcast(BroadcastType* buffer, int count, int root_rank,
                      const communicator& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Bcast(
        buffer, count, detail::mpi_type_impl<BroadcastType>::get_type(),
        root_rank, comm.get()));
}


/**
 * Reduce data into root from all calling processes on the same communicator.
 *
 * @param send_buffer  the buffer to reduce
 * @param recv_buffer  the reduced result
 * @param count  the number of elements to reduce
 * @param operation  the MPI_Op type reduce operation.
 * @param comm  the communicator
 */
template <typename ReduceType>
inline void reduce(const ReduceType* send_buffer, ReduceType* recv_buffer,
                   int count, MPI_Op operation, int root_rank,
                   const communicator& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Reduce(send_buffer, recv_buffer, count,
                   detail::mpi_type_impl<ReduceType>::get_type(), operation,
                   root_rank, comm.get()));
}


/**
 * Reduce data into root from all calling processes on the same communicator.
 *
 * @param send_buffer  the buffer to reduce
 * @param recv_buffer  the reduced result
 * @param count  the number of elements to reduce
 * @param operation  the MPI_Op type reduce operation.
 * @param comm  the communicator
 *
 * @return  the request handle for the call
 */
template <typename ReduceType>
inline MPI_Request i_reduce(const ReduceType* send_buffer,
                            ReduceType* recv_buffer, int count,
                            MPI_Op operation, int root_rank,
                            const communicator& comm)
{
    MPI_Request req;
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Ireduce(send_buffer, recv_buffer, count,
                    detail::mpi_type_impl<ReduceType>::get_type(), operation,
                    root_rank, comm.get(), &req));
    return req;
}


/**
 * Reduce data from all calling processes from all calling processes on same
 * communicator.
 *
 * @param recv_buffer  the data to reduce and the reduced result
 * @param count  the number of elements to reduce
 * @param operation  the MPI_Op type reduce operation.
 * @param comm  the communicator
 */
template <typename ReduceType>
inline void all_reduce(ReduceType* recv_buffer, int count, MPI_Op operation,
                       const communicator& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Allreduce(
        detail::in_place<ReduceType>(), recv_buffer, count,
        detail::mpi_type_impl<ReduceType>::get_type(), operation, comm.get()));
}


/**
 * Reduce data from all calling processes from all calling processes on same
 * communicator.
 *
 * @param recv_buffer  the data to reduce and the reduced result
 * @param count  the number of elements to reduce
 * @param operation  the reduce operation. See @MPI_Op
 * @param comm  the communicator
 *
 * @return  the request handle for the call
 */
template <typename ReduceType>
inline MPI_Request i_all_reduce(ReduceType* recv_buffer, int count,
                                MPI_Op operation, const communicator& comm)
{
    MPI_Request req;
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Iallreduce(detail::in_place<ReduceType>(), recv_buffer, count,
                       detail::mpi_type_impl<ReduceType>::get_type(), operation,
                       comm.get(), &req));
    return req;
}


/**
 * Reduce data from all calling processes from all calling processes on same
 * communicator.
 *
 * @param send_buffer  the data to reduce
 * @param recv_buffer  the reduced result
 * @param count  the number of elements to reduce
 * @param operation  the reduce operation. See @MPI_Op
 * @param comm  the communicator
 */
template <typename ReduceType>
inline void all_reduce(const ReduceType* send_buffer, ReduceType* recv_buffer,
                       int count, MPI_Op operation, const communicator& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Allreduce(
        send_buffer, recv_buffer, count,
        detail::mpi_type_impl<ReduceType>::get_type(), operation, comm.get()));
}


/**
 * Reduce data from all calling processes from all calling processes on same
 * communicator.
 *
 * @param send_buffer  the data to reduce
 * @param recv_buffer  the reduced result
 * @param count  the number of elements to reduce
 * @param operation  the reduce operation. See @MPI_Op
 * @param comm  the communicator
 *
 * @return  the request handle for the call
 */
template <typename ReduceType>
inline MPI_Request i_all_reduce(const ReduceType* send_buffer,
                                ReduceType* recv_buffer, int count,
                                MPI_Op operation, const communicator& comm)
{
    MPI_Request req;
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Iallreduce(send_buffer, recv_buffer, count,
                       detail::mpi_type_impl<ReduceType>::get_type(), operation,
                       comm.get(), &req));
    return req;
}


/**
 * Gather data onto the root rank from all ranks in the communicator.
 *
 * @param send_buffer  the buffer to gather from
 * @param send_count  the number of elements to send
 * @param recv_buffer  the buffer to gather into
 * @param recv_count  the number of elements to receive
 * @param root_rank  the rank to gather into
 * @param comm  the communicator
 */
template <typename SendType, typename RecvType>
inline void gather(const SendType* send_buffer, const int send_count,
                   RecvType* recv_buffer, const int recv_count, int root_rank,
                   const communicator& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Gather(
        send_buffer, send_count, detail::mpi_type_impl<SendType>::get_type(),
        recv_buffer, recv_count, detail::mpi_type_impl<RecvType>::get_type(),
        root_rank, comm.get()));
}


/**
 * Gather data onto the root rank from all ranks in the communicator with
 * offsets.
 *
 * @param send_buffer  the buffer to gather from
 * @param send_count  the number of elements to send
 * @param recv_buffer  the buffer to gather into
 * @param recv_count  the number of elements to receive
 * @param displacements  the offsets for the buffer
 * @param root_rank  the rank to gather into
 * @param comm  the communicator
 */
template <typename SendType, typename RecvType>
inline void gather_v(const SendType* send_buffer, const int send_count,
                     RecvType* recv_buffer, const int* recv_counts,
                     const int* displacements, int root_rank,
                     const communicator& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Gatherv(
        send_buffer, send_count, detail::mpi_type_impl<SendType>::get_type(),
        recv_buffer, recv_counts, displacements,
        detail::mpi_type_impl<RecvType>::get_type(), root_rank, comm.get()));
}


/**
 * Gather data onto all ranks from all ranks in the communicator.
 *
 * @param send_buffer  the buffer to gather from
 * @param send_count  the number of elements to send
 * @param recv_buffer  the buffer to gather into
 * @param recv_count  the number of elements to receive
 * @param comm  the communicator
 */
template <typename SendType, typename RecvType>
inline void all_gather(const SendType* send_buffer, const int send_count,
                       RecvType* recv_buffer, const int recv_count,
                       const communicator& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Allgather(
        send_buffer, send_count, detail::mpi_type_impl<SendType>::get_type(),
        recv_buffer, recv_count, detail::mpi_type_impl<RecvType>::get_type(),
        comm.get()));
}


/**
 * Scatter data from root rank to all ranks in the communicator.
 *
 * @param send_buffer  the buffer to gather from
 * @param send_count  the number of elements to send
 * @param recv_buffer  the buffer to gather into
 * @param recv_count  the number of elements to receive
 * @param comm  the communicator
 */
template <typename SendType, typename RecvType>
inline void scatter(const SendType* send_buffer, const int send_count,
                    RecvType* recv_buffer, const int recv_count, int root_rank,
                    const communicator& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Scatter(
        send_buffer, send_count, detail::mpi_type_impl<SendType>::get_type(),
        recv_buffer, recv_count, detail::mpi_type_impl<RecvType>::get_type(),
        root_rank, comm.get()));
}


/**
 * Scatter data from root rank to all ranks in the communicator with offsets.
 *
 * @param send_buffer  the buffer to gather from
 * @param send_count  the number of elements to send
 * @param recv_buffer  the buffer to gather into
 * @param recv_count  the number of elements to receive
 * @param displacements  the offsets for the buffer
 * @param comm  the communicator
 */
template <typename SendType, typename RecvType>
inline void scatter_v(const SendType* send_buffer, const int* send_counts,
                      const int* displacements, RecvType* recv_buffer,
                      const int recv_count, int root_rank,
                      const communicator& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Scatterv(
        send_buffer, send_counts, displacements,
        detail::mpi_type_impl<SendType>::get_type(), recv_buffer, recv_count,
        detail::mpi_type_impl<RecvType>::get_type(), root_rank, comm.get()));
}


/**
 * Communicate data from all ranks to all other ranks in place (MPI_Alltoall).
 * See MPI documentation for more details.
 *
 * @param buffer  the buffer to send and the buffer receive
 * @param recv_count  the number of elements to receive
 * @param comm  the communicator
 *
 * @note This overload uses MPI_IN_PLACE and the source and destination buffers
 *       are the same.
 */
template <typename RecvType>
inline void all_to_all(RecvType* recv_buffer, const int recv_count,
                       const communicator& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Alltoall(
        detail::in_place<RecvType>(), recv_count,
        detail::mpi_type_impl<RecvType>::get_type(), recv_buffer, recv_count,
        detail::mpi_type_impl<RecvType>::get_type(), comm.get()));
}


/**
 * Communicate data from all ranks to all other ranks in place (MPI_Alltoall).
 * See MPI documentation for more details.
 *
 * @param buffer  the buffer to send and the buffer receive
 * @param recv_count  the number of elements to receive
 * @param comm  the communicator
 *
 * @return  the request handle for the call
 *
 * @note This overload uses MPI_IN_PLACE and the source and destination buffers
 *       are the same.
 */
template <typename RecvType>
inline MPI_Request i_all_to_all(RecvType* recv_buffer, const int recv_count,
                                const communicator& comm)
{
    MPI_Request req;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Ialltoall(
        detail::in_place<RecvType>(), recv_count,
        detail::mpi_type_impl<RecvType>::get_type(), recv_buffer, recv_count,
        detail::mpi_type_impl<RecvType>::get_type(), comm.get(), &req));
    return req;
}


/**
 * Communicate data from all ranks to all other ranks (MPI_Alltoall).
 * See MPI documentation for more details.
 *
 * @param send_buffer  the buffer to send
 * @param send_count  the number of elements to send
 * @param recv_buffer  the buffer to receive
 * @param recv_count  the number of elements to receive
 * @param comm  the communicator
 */
template <typename SendType, typename RecvType>
inline void all_to_all(const SendType* send_buffer, const int send_count,
                       RecvType* recv_buffer, const int recv_count,
                       const communicator& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Alltoall(
        send_buffer, send_count, detail::mpi_type_impl<SendType>::get_type(),
        recv_buffer, recv_count, detail::mpi_type_impl<RecvType>::get_type(),
        comm.get()));
}


/**
 * Communicate data from all ranks to all other ranks (MPI_Alltoall).
 * See MPI documentation for more details.
 *
 * @param send_buffer  the buffer to send
 * @param send_count  the number of elements to send
 * @param recv_buffer  the buffer to receive
 * @param recv_count  the number of elements to receive
 * @param comm  the communicator
 *
 * @return  the request handle for the call
 */
template <typename SendType, typename RecvType>
inline MPI_Request i_all_to_all(const SendType* send_buffer,
                                const int send_count, RecvType* recv_buffer,
                                const int recv_count, const communicator& comm)
{
    MPI_Request req;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Ialltoall(
        send_buffer, send_count, detail::mpi_type_impl<SendType>::get_type(),
        recv_buffer, recv_count, detail::mpi_type_impl<RecvType>::get_type(),
        comm.get(), &req));
    return req;
}


/**
 * Communicate data from all ranks to all other ranks with
 * offsets (MPI_Alltoallv). See MPI documentation for more details.
 *
 * @param send_buffer  the buffer to send
 * @param send_count  the number of elements to send
 * @param send_offsets  the offsets for the send buffer
 * @param recv_buffer  the buffer to gather into
 * @param recv_count  the number of elements to receive
 * @param recv_offsets  the offsets for the recv buffer
 * @param comm  the communicator
 */
template <typename SendType, typename RecvType>
inline void all_to_all_v(const SendType* send_buffer, const int* send_counts,
                         const int* send_offsets, RecvType* recv_buffer,
                         const int* recv_counts, const int* recv_offsets,
                         const communicator& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Alltoallv(
        send_buffer, send_counts, send_offsets,
        detail::mpi_type_impl<SendType>::get_type(), recv_buffer, recv_counts,
        recv_offsets, detail::mpi_type_impl<RecvType>::get_type(), comm.get()));
}


/**
 * Communicate data from all ranks to all other ranks with
 * offsets (MPI_Alltoallv). See MPI documentation for more details.
 *
 * @param send_buffer  the buffer to send
 * @param send_count  the number of elements to send
 * @param send_offsets  the offsets for the send buffer
 * @param recv_buffer  the buffer to gather into
 * @param recv_count  the number of elements to receive
 * @param recv_offsets  the offsets for the recv buffer
 * @param comm  the communicator
 *
 * @return  the request handle for the call
 */
template <typename SendType, typename RecvType>
inline MPI_Request i_all_to_all_v(const SendType* send_buffer,
                                  const int* send_counts,
                                  const int* send_offsets,
                                  RecvType* recv_buffer, const int* recv_counts,
                                  const int* recv_offsets,
                                  const communicator& comm)
{
    MPI_Request req;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Ialltoallv(
        send_buffer, send_counts, send_offsets,
        detail::mpi_type_impl<SendType>::get_type(), recv_buffer, recv_counts,
        recv_offsets, detail::mpi_type_impl<RecvType>::get_type(), comm.get(),
        &req));
    return req;
}


/**
 * Does a scan operation with the given operator.
 * (MPI_Scan). See MPI documentation for more details.
 *
 * @param send_buffer  the buffer to scan from
 * @param recv_buffer  the result buffer
 * @param recv_count  the number of elements to scan
 * @param operation  the operation type to be used for the scan. See @MPI_Op
 * @param comm  the communicator
 * @param req  the request handle
 */
template <typename ScanType>
inline void scan(const ScanType* send_buffer, ScanType* recv_buffer, int count,
                 MPI_Op operation, const communicator& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Scan(
        send_buffer, recv_buffer, count,
        detail::mpi_type_impl<ScanType>::get_type(), operation, comm.get()));
}


/**
 * Does a scan operation with the given operator.
 * (MPI_Scan). See MPI documentation for more details.
 *
 * @param send_buffer  the buffer to scan from
 * @param recv_buffer  the result buffer
 * @param recv_count  the number of elements to scan
 * @param operation  the operation type to be used for the scan. See @MPI_Op
 * @param comm  the communicator
 *
 * @return  the request handle for the call
 */
template <typename ScanType>
inline MPI_Request i_scan(const ScanType* send_buffer, ScanType* recv_buffer,
                          int count, MPI_Op operation, const communicator& comm)
{
    MPI_Request req;
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Iscan(send_buffer, recv_buffer, count,
                  detail::mpi_type_impl<ScanType>::get_type(), operation,
                  comm.get(), &req));
    return req;
}


}  // namespace mpi
}  // namespace gko


#endif  // GKO_HAVE_MPI


#endif  // GKO_PUBLIC_CORE_BASE_MPI_HPP_
