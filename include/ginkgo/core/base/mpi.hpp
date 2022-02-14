/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


#include <memory>
#include <sstream>
#include <string>
#include <type_traits>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils_helper.hpp>


#if GINKGO_BUILD_MPI


#include <mpi.h>


namespace gko {
namespace mpi {


/**
 * Return if GPU aware functionality is available
 */
static constexpr bool is_gpu_aware()
{
#if GINKGO_HAVE_GPU_AWARE_MPI
    return true;
#else
    return false;
#endif
}


#define GKO_REGISTER_MPI_TYPE(input_type, mpi_type)         \
    template <>                                             \
    struct type_impl<input_type> {                          \
        static MPI_Datatype get_type() { return mpi_type; } \
    }

template <typename T>
struct type_impl {};


GKO_REGISTER_MPI_TYPE(char, MPI_CHAR);
GKO_REGISTER_MPI_TYPE(unsigned char, MPI_UNSIGNED_CHAR);
GKO_REGISTER_MPI_TYPE(unsigned, MPI_UNSIGNED);
GKO_REGISTER_MPI_TYPE(int, MPI_INT);
GKO_REGISTER_MPI_TYPE(unsigned short, MPI_UNSIGNED_SHORT);
GKO_REGISTER_MPI_TYPE(unsigned long, MPI_UNSIGNED_LONG);
GKO_REGISTER_MPI_TYPE(long, MPI_LONG);
GKO_REGISTER_MPI_TYPE(long long, MPI_LONG_LONG_INT);
GKO_REGISTER_MPI_TYPE(float, MPI_FLOAT);
GKO_REGISTER_MPI_TYPE(double, MPI_DOUBLE);
GKO_REGISTER_MPI_TYPE(long double, MPI_LONG_DOUBLE);
GKO_REGISTER_MPI_TYPE(std::complex<float>, MPI_C_COMPLEX);
GKO_REGISTER_MPI_TYPE(std::complex<double>, MPI_C_DOUBLE_COMPLEX);


class contiguous_type {
public:
    contiguous_type(int count, MPI_Datatype old_type)
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Type_contiguous(count, old_type, &type_));
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Type_commit(&type_));
    }

    contiguous_type(const contiguous_type&) = delete;

    contiguous_type& operator=(const contiguous_type&) = delete;

    contiguous_type(contiguous_type&& other) noexcept
    {
        *this = std::move(other);
    }

    contiguous_type& operator=(contiguous_type&& other) noexcept
    {
        if (this != &other) {
            this->type_ = std::exchange(other.type_, MPI_DATATYPE_NULL);
        }
        return *this;
    }

    ~contiguous_type()
    {
        if (type_ != MPI_DATATYPE_NULL) {
            MPI_Type_free(&type_);
        }
    }

    MPI_Datatype get() { return type_; }

private:
    MPI_Datatype type_;
};


template <typename T>
inline const T* in_place()
{
    return reinterpret_cast<const T*>(MPI_IN_PLACE);
}


/**
 * This enum specifies the threading type to be used when creating an MPI
 * environment.
 */
enum class thread_type {
    serialized = MPI_THREAD_SERIALIZED,
    funneled = MPI_THREAD_FUNNELED,
    single = MPI_THREAD_SINGLE,
    multiple = MPI_THREAD_MULTIPLE
};


/**
 * Class that sets up and finalizes the MPI environment. This class is a simple
 * RAII wrapper to MPI_Init and MPI_Finalize.
 *
 * MPI_Init must have been called before calling any MPI functions.
 *
 * @note If MPI_Init has already been called, then this class should not be
 * used.
 */
class environment {
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

    /**
     * Return the provided thread support.
     *
     * @return the provided thread support
     */
    int get_provided_thread_support() const { return provided_thread_support_; }

    /**
     * Call MPI_Init_thread and initialize the MPI environment
     *
     * @param argc  the number of arguments to the main function.
     * @param argv  the arguments provided to the main function.
     * @param thread_t  the type of threading for initialization. See
     *                  @thread_type
     */
    environment(int& argc, char**& argv,
                const thread_type thread_t = thread_type::serialized)
    {
        this->required_thread_support_ = static_cast<int>(thread_t);
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Init_thread(&argc, &argv, this->required_thread_support_,
                            &(this->provided_thread_support_)));
    }

    /**
     * Call MPI_Finalize at the end of the scope of this class.
     */
    ~environment() { MPI_Finalize(); }

    environment(const environment&) = delete;
    environment(environment&&) = delete;
    environment& operator=(const environment&) = delete;
    environment& operator=(environment&&) = delete;

private:
    int required_thread_support_;
    int provided_thread_support_;
};


namespace {


/**
 * A deleter class that calls MPI_Comm_free on the owning MPI_Comm object and
 * deletes the underlying comm ptr
 */
class comm_deleter {
public:
    using pointer = MPI_Comm*;
    void operator()(pointer comm) const
    {
        GKO_ASSERT(*comm != MPI_COMM_NULL);
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_free(comm));
        delete comm;
    }
};


}  // namespace


/**
 * The status struct is a light wrapper around the MPI_Status struct.
 */
struct status {
    /**
     * The default constructor. It creates an empty MPI_Status
     */
    status() : status_(MPI_Status{}) {}

    /**
     * Get a pointer to the underlying MPI_Status object.
     *
     * @return  a pointer to MPI_Status object
     */
    MPI_Status* get() { return &this->status_; }

    /**
     * Get the count of the number of elements received by the communication
     * call.
     *
     * @tparam T  The datatype of the object that was received.
     *
     * @param data  The data object of type T that was received.
     *
     * @return  the count
     */
    template <typename T>
    int get_count(const T* data) const
    {
        int count;
        MPI_Get_count(&status_, type_impl<T>::get_type(), &count);
        return count;
    }

private:
    MPI_Status status_;
};


/**
 * The request class is a light wrapper around the MPI_Request handle class.
 */
class request {
public:
    /**
     * The default constructor. It creates a null MPI_Request of
     * MPI_REQUEST_NULL type.
     */
    request() : req_(MPI_REQUEST_NULL) {}

    /**
     * Get a pointer to the underlying MPI_Request handle.
     *
     * @return  a pointer to MPI_Request handle
     */
    MPI_Request* get() { return &this->req_; }

    /**
     * Allows a rank to wait on a particular request handle.
     *
     * @param req  The request to wait on.
     * @param status  The status variable that can be queried.
     */
    status wait()
    {
        status status;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Wait(&req_, status.get()));
        return status;
    }


private:
    MPI_Request req_;
};


/**
 * Allows a rank to wait on multiple request handles.
 *
 * @param req  The vector of request handles to be waited on.
 *
 * @return status  The vector of status objects that can be queried.
 */
inline std::vector<status> wait_all(std::vector<request>& req)
{
    std::vector<status> stat;
    for (auto i = 0; i < req.size(); ++i) {
        stat.emplace_back(req[i].wait());
    }
    return stat;
}


/**
 * A communicator class that takes in the given communicator and duplicates it
 * for our purposes. As the class or object goes out of scope, the communicator
 * is freed.
 */
class communicator {
public:
    /**
     * Non-owning constructor for an existing communicator of type MPI_Comm. The
     * MPI_Comm object will not be deleted after the communicator object has
     * been freed and an explicit MPI_Comm_free needs to be called on the
     * original MPI_Comm object.
     *
     * @param comm The input MPI_Comm object.
     */
    communicator(const MPI_Comm& comm)
    {
        this->comm_.reset(new MPI_Comm(comm));
    }

    /**
     * Create a communicator object from an existing MPI_Comm object using color
     * and key.
     *
     * @param comm The input MPI_Comm object.
     * @param color The color to split the original comm object
     * @param key  The key to split the comm object
     */
    communicator(const MPI_Comm& comm, int color, int key)
    {
        MPI_Comm comm_out;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_split(comm, color, key, &comm_out));
        this->comm_.reset(new MPI_Comm(comm_out), comm_deleter{});
    }

    /**
     * Create a communicator object from an existing MPI_Comm object using color
     * and key.
     *
     * @param comm The input communicator object.
     * @param color The color to split the original comm object
     * @param key  The key to split the comm object
     */
    communicator(const communicator& comm, int color, int key)
    {
        MPI_Comm comm_out;
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Comm_split(comm.get(), color, key, &comm_out));
        this->comm_.reset(new MPI_Comm(comm_out), comm_deleter{});
    }

    /**
     * Return the underlying MPI_Comm object.
     *
     * @return  the MPI_Comm object
     */
    const MPI_Comm& get() const { return *(this->comm_.get()); }

    /**
     * Return the size of the communicator (number of ranks).
     *
     * @return  the size
     */
    int size() const { return get_num_ranks(); }

    /**
     * Return the rank of the calling process in the communicator.
     *
     * @return  the rank
     */
    int rank() const { return get_my_rank(); };

    /**
     * Return the node local rank of the calling process in the communicator.
     *
     * @return  the node local rank
     */
    int node_local_rank() const { return get_node_local_rank(); };

    /**
     * Compare two communicator objects for equality.
     *
     * @return  if the two comm objects are equal
     */
    bool operator==(const communicator& rhs) const
    {
        return compare(rhs.get());
    }

    /**
     * Compare two communicator objects for non-equality.
     *
     * @return  if the two comm objects are not equal
     */
    bool operator!=(const communicator& rhs) const { return !(*this == rhs); }

    /**
     * This function is used to synchronize the ranks in the communicator.
     * Calls MPI_Barrier
     */
    void synchronize() const
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Barrier(this->get()));
    }

    /**
     * Send (Blocking) data from calling process to destination rank.
     *
     * @param send_buffer  the buffer to send
     * @param send_count  the number of elements to send
     * @param destination_rank  the rank to send the data to
     * @param send_tag  the tag for the send call
     */
    template <typename SendType>
    void send(const SendType* send_buffer, const int send_count,
              const int destination_rank, const int send_tag) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Send(send_buffer, send_count, type_impl<SendType>::get_type(),
                     destination_rank, send_tag, this->get()));
    }

    /**
     * Send (Non-blocking, Immediate return) data from calling process to
     * destination rank.
     *
     * @param send_buffer  the buffer to send
     * @param send_count  the number of elements to send
     * @param destination_rank  the rank to send the data to
     * @param send_tag  the tag for the send call
     *
     * @return  the request handle for the send call
     */
    template <typename SendType>
    request i_send(const SendType* send_buffer, const int send_count,
                   const int destination_rank, const int send_tag) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Isend(send_buffer, send_count, type_impl<SendType>::get_type(),
                      destination_rank, send_tag, this->get(), req.get()));
        return req;
    }

    /**
     * Receive data from source rank.
     *
     * @param recv_buffer  the buffer to receive
     * @param recv_count  the number of elements to receive
     * @param source_rank  the rank to receive the data from
     * @param recv_tag  the tag for the recv call
     *
     * @return  the status of completion of this call
     */
    template <typename RecvType>
    status recv(RecvType* recv_buffer, const int recv_count,
                const int source_rank, const int recv_tag) const
    {
        status st;
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Recv(recv_buffer, recv_count, type_impl<RecvType>::get_type(),
                     source_rank, recv_tag, this->get(), st.get()));
        return st;
    }

    /**
     * Receive (Non-blocking, Immediate return) data from source rank.
     *
     * @param recv_buffer  the buffer to send
     * @param recv_count  the number of elements to receive
     * @param source_rank  the rank to receive the data from
     * @param recv_tag  the tag for the recv call
     *
     * @return  the request handle for the recv call
     */
    template <typename RecvType>
    request i_recv(RecvType* recv_buffer, const int recv_count,
                   const int source_rank, const int recv_tag) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Irecv(recv_buffer, recv_count, type_impl<RecvType>::get_type(),
                      source_rank, recv_tag, this->get(), req.get()));
        return req;
    }

    /**
     * Broadcast data from calling process to all ranks in the communicator
     *
     * @param buffer  the buffer to broadcsat
     * @param count  the number of elements to broadcast
     * @param root_rank  the rank to broadcast from
     */
    template <typename BroadcastType>
    void broadcast(BroadcastType* buffer, int count, int root_rank) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Bcast(buffer, count,
                                           type_impl<BroadcastType>::get_type(),
                                           root_rank, this->get()));
    }

    /**
     * (Non-blocking) Broadcast data from calling process to all ranks in the
     * communicator
     *
     * @param buffer  the buffer to broadcsat
     * @param count  the number of elements to broadcast
     * @param root_rank  the rank to broadcast from
     *
     * @return  the request handle for the call
     */
    template <typename BroadcastType>
    request i_broadcast(BroadcastType* buffer, int count, int root_rank) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Ibcast(buffer, count, type_impl<BroadcastType>::get_type(),
                       root_rank, this->get(), req.get()));
        return req;
    }

    /**
     * Reduce data into root from all calling processes on the same
     * communicator.
     *
     * @param send_buffer  the buffer to reduce
     * @param recv_buffer  the reduced result
     * @param count  the number of elements to reduce
     * @param operation  the MPI_Op type reduce operation.
     */
    template <typename ReduceType>
    void reduce(const ReduceType* send_buffer, ReduceType* recv_buffer,
                int count, MPI_Op operation, int root_rank) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Reduce(send_buffer, recv_buffer, count,
                                            type_impl<ReduceType>::get_type(),
                                            operation, root_rank, this->get()));
    }

    /**
     * (Non-blocking) Reduce data into root from all calling processes on the
     * same communicator.
     *
     * @param send_buffer  the buffer to reduce
     * @param recv_buffer  the reduced result
     * @param count  the number of elements to reduce
     * @param operation  the MPI_Op type reduce operation.
     *
     * @return  the request handle for the call
     */
    template <typename ReduceType>
    request i_reduce(const ReduceType* send_buffer, ReduceType* recv_buffer,
                     int count, MPI_Op operation, int root_rank) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Ireduce(
            send_buffer, recv_buffer, count, type_impl<ReduceType>::get_type(),
            operation, root_rank, this->get(), req.get()));
        return req;
    }

    /**
     * (In-place) Reduce data from all calling processes from all calling
     * processes on same communicator.
     *
     * @param recv_buffer  the data to reduce and the reduced result
     * @param count  the number of elements to reduce
     * @param operation  the MPI_Op type reduce operation.
     */
    template <typename ReduceType>
    void all_reduce(ReduceType* recv_buffer, int count, MPI_Op operation) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Allreduce(
            in_place<ReduceType>(), recv_buffer, count,
            type_impl<ReduceType>::get_type(), operation, this->get()));
    }

    /**
     * (In-place, non-blocking) Reduce data from all calling processes from all
     * calling processes on same communicator.
     *
     * @param recv_buffer  the data to reduce and the reduced result
     * @param count  the number of elements to reduce
     * @param operation  the reduce operation. See @MPI_Op
     *
     * @return  the request handle for the call
     */
    template <typename ReduceType>
    request i_all_reduce(ReduceType* recv_buffer, int count,
                         MPI_Op operation) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Iallreduce(in_place<ReduceType>(), recv_buffer, count,
                           type_impl<ReduceType>::get_type(), operation,
                           this->get(), req.get()));
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
     */
    template <typename ReduceType>
    void all_reduce(const ReduceType* send_buffer, ReduceType* recv_buffer,
                    int count, MPI_Op operation) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Allreduce(
            send_buffer, recv_buffer, count, type_impl<ReduceType>::get_type(),
            operation, this->get()));
    }

    /**
     * Reduce data from all calling processes from all calling processes on same
     * communicator.
     *
     * @param send_buffer  the data to reduce
     * @param recv_buffer  the reduced result
     * @param count  the number of elements to reduce
     * @param operation  the reduce operation. See @MPI_Op
     *
     * @return  the request handle for the call
     */
    template <typename ReduceType>
    request i_all_reduce(const ReduceType* send_buffer, ReduceType* recv_buffer,
                         int count, MPI_Op operation) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Iallreduce(
            send_buffer, recv_buffer, count, type_impl<ReduceType>::get_type(),
            operation, this->get(), req.get()));
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
     */
    template <typename SendType, typename RecvType>
    void gather(const SendType* send_buffer, const int send_count,
                RecvType* recv_buffer, const int recv_count,
                int root_rank) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Gather(send_buffer, send_count, type_impl<SendType>::get_type(),
                       recv_buffer, recv_count, type_impl<RecvType>::get_type(),
                       root_rank, this->get()));
    }

    /**
     * (Non-blocking) Gather data onto the root rank from all ranks in the
     * communicator.
     *
     * @param send_buffer  the buffer to gather from
     * @param send_count  the number of elements to send
     * @param recv_buffer  the buffer to gather into
     * @param recv_count  the number of elements to receive
     * @param root_rank  the rank to gather into
     *
     * @return  the request handle for the call
     */
    template <typename SendType, typename RecvType>
    request i_gather(const SendType* send_buffer, const int send_count,
                     RecvType* recv_buffer, const int recv_count,
                     int root_rank) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Igather(
            send_buffer, send_count, type_impl<SendType>::get_type(),
            recv_buffer, recv_count, type_impl<RecvType>::get_type(), root_rank,
            this->get(), req.get()));
        return req;
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
     */
    template <typename SendType, typename RecvType>
    void gather_v(const SendType* send_buffer, const int send_count,
                  RecvType* recv_buffer, const int* recv_counts,
                  const int* displacements, int root_rank) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Gatherv(
            send_buffer, send_count, type_impl<SendType>::get_type(),
            recv_buffer, recv_counts, displacements,
            type_impl<RecvType>::get_type(), root_rank, this->get()));
    }

    /**
     * (Non-blocking) Gather data onto the root rank from all ranks in the
     * communicator with offsets.
     *
     * @param send_buffer  the buffer to gather from
     * @param send_count  the number of elements to send
     * @param recv_buffer  the buffer to gather into
     * @param recv_count  the number of elements to receive
     * @param displacements  the offsets for the buffer
     * @param root_rank  the rank to gather into
     *
     * @return  the request handle for the call
     */
    template <typename SendType, typename RecvType>
    request i_gather_v(const SendType* send_buffer, const int send_count,
                       RecvType* recv_buffer, const int* recv_counts,
                       const int* displacements, int root_rank) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Igatherv(
            send_buffer, send_count, type_impl<SendType>::get_type(),
            recv_buffer, recv_counts, displacements,
            type_impl<RecvType>::get_type(), root_rank, this->get(),
            req.get()));
        return req;
    }

    /**
     * Gather data onto all ranks from all ranks in the communicator.
     *
     * @param send_buffer  the buffer to gather from
     * @param send_count  the number of elements to send
     * @param recv_buffer  the buffer to gather into
     * @param recv_count  the number of elements to receive
     */
    template <typename SendType, typename RecvType>
    void all_gather(const SendType* send_buffer, const int send_count,
                    RecvType* recv_buffer, const int recv_count) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Allgather(
            send_buffer, send_count, type_impl<SendType>::get_type(),
            recv_buffer, recv_count, type_impl<RecvType>::get_type(),
            this->get()));
    }

    /**
     * (Non-blocking) Gather data onto all ranks from all ranks in the
     * communicator.
     *
     * @param send_buffer  the buffer to gather from
     * @param send_count  the number of elements to send
     * @param recv_buffer  the buffer to gather into
     * @param recv_count  the number of elements to receive
     *
     * @return  the request handle for the call
     */
    template <typename SendType, typename RecvType>
    request i_all_gather(const SendType* send_buffer, const int send_count,
                         RecvType* recv_buffer, const int recv_count) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Iallgather(
            send_buffer, send_count, type_impl<SendType>::get_type(),
            recv_buffer, recv_count, type_impl<RecvType>::get_type(),
            this->get(), req.get()));
        return req;
    }

    /**
     * Scatter data from root rank to all ranks in the communicator.
     *
     * @param send_buffer  the buffer to gather from
     * @param send_count  the number of elements to send
     * @param recv_buffer  the buffer to gather into
     * @param recv_count  the number of elements to receive
     */
    template <typename SendType, typename RecvType>
    void scatter(const SendType* send_buffer, const int send_count,
                 RecvType* recv_buffer, const int recv_count,
                 int root_rank) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Scatter(
            send_buffer, send_count, type_impl<SendType>::get_type(),
            recv_buffer, recv_count, type_impl<RecvType>::get_type(), root_rank,
            this->get()));
    }

    /**
     * (Non-blocking) Scatter data from root rank to all ranks in the
     * communicator.
     *
     * @param send_buffer  the buffer to gather from
     * @param send_count  the number of elements to send
     * @param recv_buffer  the buffer to gather into
     * @param recv_count  the number of elements to receive
     *
     * @return  the request handle for the call
     */
    template <typename SendType, typename RecvType>
    request i_scatter(const SendType* send_buffer, const int send_count,
                      RecvType* recv_buffer, const int recv_count,
                      int root_rank) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Iscatter(
            send_buffer, send_count, type_impl<SendType>::get_type(),
            recv_buffer, recv_count, type_impl<RecvType>::get_type(), root_rank,
            this->get(), req.get()));
        return req;
    }

    /**
     * Scatter data from root rank to all ranks in the communicator with
     * offsets.
     *
     * @param send_buffer  the buffer to gather from
     * @param send_count  the number of elements to send
     * @param recv_buffer  the buffer to gather into
     * @param recv_count  the number of elements to receive
     * @param displacements  the offsets for the buffer
     * @param comm  the communicator
     */
    template <typename SendType, typename RecvType>
    void scatter_v(const SendType* send_buffer, const int* send_counts,
                   const int* displacements, RecvType* recv_buffer,
                   const int recv_count, int root_rank) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Scatterv(
            send_buffer, send_counts, displacements,
            type_impl<SendType>::get_type(), recv_buffer, recv_count,
            type_impl<RecvType>::get_type(), root_rank, this->get()));
    }

    /**
     * (Non-blocking) Scatter data from root rank to all ranks in the
     * communicator with offsets.
     *
     * @param send_buffer  the buffer to gather from
     * @param send_count  the number of elements to send
     * @param recv_buffer  the buffer to gather into
     * @param recv_count  the number of elements to receive
     * @param displacements  the offsets for the buffer
     * @param comm  the communicator
     *
     * @return  the request handle for the call
     */
    template <typename SendType, typename RecvType>
    request i_scatter_v(const SendType* send_buffer, const int* send_counts,
                        const int* displacements, RecvType* recv_buffer,
                        const int recv_count, int root_rank) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Iscatterv(send_buffer, send_counts, displacements,
                          type_impl<SendType>::get_type(), recv_buffer,
                          recv_count, type_impl<RecvType>::get_type(),
                          root_rank, this->get(), req.get()));
        return req;
    }

    /**
     * (In-place) Communicate data from all ranks to all other ranks in place
     * (MPI_Alltoall). See MPI documentation for more details.
     *
     * @param buffer  the buffer to send and the buffer receive
     * @param recv_count  the number of elements to receive
     * @param comm  the communicator
     *
     * @note This overload uses MPI_IN_PLACE and the source and destination
     * buffers are the same.
     */
    template <typename RecvType>
    void all_to_all(RecvType* recv_buffer, const int recv_count) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Alltoall(
            in_place<RecvType>(), recv_count, type_impl<RecvType>::get_type(),
            recv_buffer, recv_count, type_impl<RecvType>::get_type(),
            this->get()));
    }

    /**
     * (In-place, Non-blocking) Communicate data from all ranks to all other
     * ranks in place (MPI_Alltoall). See MPI documentation for more details.
     *
     * @param buffer  the buffer to send and the buffer receive
     * @param recv_count  the number of elements to receive
     * @param comm  the communicator
     *
     * @return  the request handle for the call
     *
     * @note This overload uses MPI_IN_PLACE and the source and destination
     * buffers are the same.
     */
    template <typename RecvType>
    request i_all_to_all(RecvType* recv_buffer, const int recv_count) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Ialltoall(
            in_place<RecvType>(), recv_count, type_impl<RecvType>::get_type(),
            recv_buffer, recv_count, type_impl<RecvType>::get_type(),
            this->get(), req.get()));
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
     */
    template <typename SendType, typename RecvType>
    void all_to_all(const SendType* send_buffer, const int send_count,
                    RecvType* recv_buffer, const int recv_count) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Alltoall(
            send_buffer, send_count, type_impl<SendType>::get_type(),
            recv_buffer, recv_count, type_impl<RecvType>::get_type(),
            this->get()));
    }

    /**
     * (Non-blocking) Communicate data from all ranks to all other ranks
     * (MPI_Alltoall). See MPI documentation for more details.
     *
     * @param send_buffer  the buffer to send
     * @param send_count  the number of elements to send
     * @param recv_buffer  the buffer to receive
     * @param recv_count  the number of elements to receive
     *
     * @return  the request handle for the call
     */
    template <typename SendType, typename RecvType>
    request i_all_to_all(const SendType* send_buffer, const int send_count,
                         RecvType* recv_buffer, const int recv_count) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Ialltoall(
            send_buffer, send_count, type_impl<SendType>::get_type(),
            recv_buffer, recv_count, type_impl<RecvType>::get_type(),
            this->get(), req.get()));
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
    void all_to_all_v(const SendType* send_buffer, const int* send_counts,
                      const int* send_offsets, RecvType* recv_buffer,
                      const int* recv_counts, const int* recv_offsets) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Alltoallv(
            send_buffer, send_counts, send_offsets,
            type_impl<SendType>::get_type(), recv_buffer, recv_counts,
            recv_offsets, type_impl<RecvType>::get_type(), this->get()));
    }

    /**
     * Communicate data from all ranks to all other ranks with
     * offsets (MPI_Alltoallv). See MPI documentation for more details.
     *
     * @param send_buffer  the buffer to send
     * @param send_count  the number of elements to send
     * @param send_offsets  the offsets for the send buffer
     * @param send_type  the MPI_Datatype for the send buffer
     * @param recv_buffer  the buffer to gather into
     * @param recv_count  the number of elements to receive
     * @param recv_offsets  the offsets for the recv buffer
     * @param recv_type  the MPI_Datatype for the recv buffer
     *
     * @return  the request handle for the call
     */
    request i_all_to_all_v(const void* send_buffer, const int* send_counts,
                           const int* send_offsets, MPI_Datatype send_type,
                           void* recv_buffer, const int* recv_counts,
                           const int* recv_offsets,
                           MPI_Datatype recv_type) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Ialltoallv(
            send_buffer, send_counts, send_offsets, send_type, recv_buffer,
            recv_counts, recv_offsets, recv_type, this->get(), req.get()));
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
     *
     * @return  the request handle for the call
     */
    template <typename SendType, typename RecvType>
    request i_all_to_all_v(const SendType* send_buffer, const int* send_counts,
                           const int* send_offsets, RecvType* recv_buffer,
                           const int* recv_counts,
                           const int* recv_offsets) const
    {
        return this->i_all_to_all_v(send_buffer, send_counts, send_offsets,
                                    type_impl<SendType>::get_type(),
                                    recv_buffer, recv_counts, recv_offsets,
                                    type_impl<RecvType>::get_type());
    }

    /**
     * Does a scan operation with the given operator.
     * (MPI_Scan). See MPI documentation for more details.
     *
     * @param send_buffer  the buffer to scan from
     * @param recv_buffer  the result buffer
     * @param recv_count  the number of elements to scan
     * @param operation  the operation type to be used for the scan. See @MPI_Op
     */
    template <typename ScanType>
    void scan(const ScanType* send_buffer, ScanType* recv_buffer, int count,
              MPI_Op operation) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Scan(send_buffer, recv_buffer, count,
                                          type_impl<ScanType>::get_type(),
                                          operation, this->get()));
    }

    /**
     * Does a scan operation with the given operator.
     * (MPI_Scan). See MPI documentation for more details.
     *
     * @param send_buffer  the buffer to scan from
     * @param recv_buffer  the result buffer
     * @param recv_count  the number of elements to scan
     * @param operation  the operation type to be used for the scan. See @MPI_Op
     *
     * @return  the request handle for the call
     */
    template <typename ScanType>
    request i_scan(const ScanType* send_buffer, ScanType* recv_buffer,
                   int count, MPI_Op operation) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Iscan(send_buffer, recv_buffer, count,
                                           type_impl<ScanType>::get_type(),
                                           operation, this->get(), req.get()));
        return req;
    }

private:
    std::shared_ptr<MPI_Comm> comm_;
    int size_{};
    int rank_{};
    int node_local_rank_{};

    int get_my_rank() const
    {
        int my_rank = 0;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_rank(get(), &my_rank));
        return my_rank;
    }

    int get_node_local_rank() const
    {
        MPI_Comm local_comm;
        int rank;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_split_type(
            this->get(), MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm));
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_rank(local_comm, &rank));
        MPI_Comm_free(&local_comm);
        return rank;
    }

    int get_num_ranks() const
    {
        int size = 1;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_size(this->get(), &size));
        return size;
    }

    bool compare(const MPI_Comm& other) const
    {
        int flag;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_compare(get(), other, &flag));
        return flag == MPI_IDENT;
    }
};


/**
 * Get the rank in the communicator of the calling process.
 *
 * @param comm  the communicator
 */
inline double get_walltime() { return MPI_Wtime(); }


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
    /**
     * The create type for the window object.
     */
    enum class create_type { allocate = 1, create = 2, dynamic_create = 3 };

    /**
     * The lock type for passive target synchronization of the windows.
     */
    enum class lock_type { shared = 1, exclusive = 2 };

    /**
     * The default constructor. It creates a null window of MPI_WIN_NULL type.
     */
    window() : window_(MPI_WIN_NULL) {}

    window(const window& other) = delete;

    window& operator=(const window& other) = delete;

    /**
     * The move constructor. Move the other object and replace it with
     * MPI_WIN_NULL
     *
     * @param other the window object to be moved.
     */
    window(window&& other) : window_{std::exchange(other.window_, MPI_WIN_NULL)}
    {}

    /**
     * The move assignment operator. Move the other object and replace it with
     * MPI_WIN_NULL
     *
     * @param other the window object to be moved.
     */
    window& operator=(window&& other)
    {
        window_ = std::exchange(other.window_, MPI_WIN_NULL);
    }

    /**
     * Create a window object with a given data pointer and type. A collective
     * operation.
     *
     * @param base  the base pointer for the window object.
     * @param num_elems  the num_elems of type ValueType the window points to.
     * @param comm  the communicator whose ranks will have windows created.
     * @param disp_unit  the displacement from base for the window object.
     * @param input_info  the MPI_Info object used to set certain properties.
     * @param c_type  the type of creation method to use to create the window.
     */
    window(ValueType* base, int num_elems, const communicator& comm,
           const int disp_unit = sizeof(ValueType),
           MPI_Info input_info = MPI_INFO_NULL,
           create_type c_type = create_type::create)
    {
        unsigned size = num_elems * sizeof(ValueType);
        if (c_type == create_type::create) {
            GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_create(
                base, size, disp_unit, input_info, comm.get(), &this->window_));
        } else if (c_type == create_type::dynamic_create) {
            GKO_ASSERT_NO_MPI_ERRORS(
                MPI_Win_create_dynamic(input_info, comm.get(), &this->window_));
        } else if (c_type == create_type::allocate) {
            GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_allocate(
                size, disp_unit, input_info, comm.get(), base, &this->window_));
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    /**
     * Get the underlying window object of MPI_Win type.
     *
     * @return the underlying window object.
     */
    MPI_Win get_window() const { return this->window_; }

    /**
     * The active target synchronization using MPI_Win_fence for the window
     * object. This is called on all associated ranks.
     *
     * @param assert  the optimization level. 0 is always valid.
     */
    void fence(int assert = 0)
    {
        if (&this->window_) {
            GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_fence(assert, this->window_));
        }
    }

    /**
     * Create an epoch using MPI_Win_lock for the window
     * object.
     *
     * @param rank  the target rank.
     * @param lock_t  the type of the lock: shared or exclusive
     * @param assert  the optimization level. 0 is always valid.
     */
    void lock(int rank, lock_type lock_t = lock_type::shared, int assert = 0)
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

    /**
     * Close the epoch using MPI_Win_unlock for the window
     * object.
     *
     * @param rank  the target rank.
     */
    void unlock(int rank)
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_unlock(rank, this->window_));
    }

    /**
     * Create the epoch on all ranks using MPI_Win_lock_all for the window
     * object.
     *
     * @param assert  the optimization level. 0 is always valid.
     */
    void lock_all(int assert = 0)
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_lock_all(assert, this->window_));
    }

    /**
     * Close the epoch on all ranks using MPI_Win_unlock_all for the window
     * object.
     */
    void unlock_all()
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_unlock_all(this->window_));
    }

    /**
     * Flush the existing RDMA operations on the target rank for the calling
     * process for the window object.
     *
     * @param rank  the target rank.
     */
    void flush(int rank)
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_flush(rank, this->window_));
    }

    /**
     * Flush the existing RDMA operations on the calling rank from the target
     * rank for the window object.
     *
     * @param rank  the target rank.
     */
    void flush_local(int rank)
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_flush_local(rank, this->window_));
    }

    /**
     * Flush all the existing RDMA operations for the calling
     * process for the window object.
     */
    void flush_all()
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_flush_all(this->window_));
    }

    /**
     * Flush all the local existing RDMA operations on the calling rank for the
     * window object.
     */
    void flush_all_local()
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_flush_local_all(this->window_));
    }

    /**
     * Synchronize the public and private buffers for the window object
     */
    void sync() { GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_sync(this->window_)); }

    /**
     * The deleter which calls MPI_Win_free when the window leaves its scope.
     */
    ~window()
    {
        if (this->window_ && this->window_ != MPI_WIN_NULL) {
            MPI_Win_free(&this->window_);
        }
    }

    /**
     * Put data into the target window.
     *
     * @param origin_buffer  the buffer to send
     * @param origin_count  the number of elements to put
     * @param target_rank  the rank to put the data to
     * @param target_disp  the displacement at the target window
     * @param target_count  the request handle for the send call
     */
    template <typename PutType>
    void put(const PutType* origin_buffer, const int origin_count,
             const int target_rank, const unsigned int target_disp,
             const int target_count) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Put(origin_buffer, origin_count, type_impl<PutType>::get_type(),
                    target_rank, target_disp, target_count,
                    type_impl<PutType>::get_type(), this->get_window()));
    }

    /**
     * Put data into the target window.
     *
     * @param origin_buffer  the buffer to send
     * @param origin_count  the number of elements to put
     * @param target_rank  the rank to put the data to
     * @param target_disp  the displacement at the target window
     * @param target_count  the request handle for the send call
     *
     * @return  the request handle for the send call
     */
    template <typename PutType>
    request r_put(const PutType* origin_buffer, const int origin_count,
                  const int target_rank, const unsigned int target_disp,
                  const int target_count) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Rput(
            origin_buffer, origin_count, type_impl<PutType>::get_type(),
            target_rank, target_disp, target_count,
            type_impl<PutType>::get_type(), this->get_window(), req.get()));
        return req;
    }

    /**
     * Accumulate data into the target window.
     *
     * @param origin_buffer  the buffer to send
     * @param origin_count  the number of elements to put
     * @param target_rank  the rank to put the data to
     * @param target_disp  the displacement at the target window
     * @param target_count  the request handle for the send call
     * @param operation  the reduce operation. See @MPI_Op
     */
    template <typename PutType>
    void accumulate(const PutType* origin_buffer, const int origin_count,
                    const int target_rank, const unsigned int target_disp,
                    const int target_count, MPI_Op operation) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Accumulate(
            origin_buffer, origin_count, type_impl<PutType>::get_type(),
            target_rank, target_disp, target_count,
            type_impl<PutType>::get_type(), operation, this->get_window()));
    }

    /**
     * (Non-blocking) Accumulate data into the target window.
     *
     * @param origin_buffer  the buffer to send
     * @param origin_count  the number of elements to put
     * @param target_rank  the rank to put the data to
     * @param target_disp  the displacement at the target window
     * @param target_count  the request handle for the send call
     * @param operation  the reduce operation. See @MPI_Op
     *
     * @return  the request handle for the send call
     */
    template <typename PutType>
    request r_accumulate(const PutType* origin_buffer, const int origin_count,
                         const int target_rank, const unsigned int target_disp,
                         const int target_count, MPI_Op operation) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Raccumulate(
            origin_buffer, origin_count, type_impl<PutType>::get_type(),
            target_rank, target_disp, target_count,
            type_impl<PutType>::get_type(), operation, this->get_window(),
            req.get()));
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
     */
    template <typename GetType>
    void get(GetType* origin_buffer, const int origin_count,
             const int target_rank, const unsigned int target_disp,
             const int target_count) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Get(origin_buffer, origin_count, type_impl<GetType>::get_type(),
                    target_rank, target_disp, target_count,
                    type_impl<GetType>::get_type(), this->get_window()));
    }

    /**
     * Get data (with handle) from the target window.
     *
     * @param origin_buffer  the buffer to send
     * @param origin_count  the number of elements to get
     * @param target_rank  the rank to get the data from
     * @param target_disp  the displacement at the target window
     * @param target_count  the request handle for the send call
     *
     * @return  the request handle for the send call
     */
    template <typename GetType>
    request r_get(GetType* origin_buffer, const int origin_count,
                  const int target_rank, const unsigned int target_disp,
                  const int target_count) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Rget(
            origin_buffer, origin_count, type_impl<GetType>::get_type(),
            target_rank, target_disp, target_count,
            type_impl<GetType>::get_type(), this->get_window(), req.get()));
        return req;
    }

    /**
     * Get Accumulate data from the target window.
     *
     * @param origin_buffer  the buffer to send
     * @param origin_count  the number of elements to get
     * @param result_buffer  the buffer to receive the target data
     * @param result_count  the number of elements to get
     * @param target_rank  the rank to get the data from
     * @param target_disp  the displacement at the target window
     * @param target_count  the request handle for the send call
     * @param operation  the reduce operation. See @MPI_Op
     */
    template <typename GetType>
    void get_accumulate(GetType* origin_buffer, const int origin_count,
                        GetType* result_buffer, const int result_count,
                        const int target_rank, const unsigned int target_disp,
                        const int target_count, MPI_Op operation) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Get_accumulate(
            origin_buffer, origin_count, type_impl<GetType>::get_type(),
            result_buffer, result_count, type_impl<GetType>::get_type(),
            target_rank, target_disp, target_count,
            type_impl<GetType>::get_type(), operation, this->get_window()));
    }

    /**
     * (Non-blocking) Get Accumulate data (with handle) from the target window.
     *
     * @param origin_buffer  the buffer to send
     * @param origin_count  the number of elements to get
     * @param result_buffer  the buffer to receive the target data
     * @param result_count  the number of elements to get
     * @param target_rank  the rank to get the data from
     * @param target_disp  the displacement at the target window
     * @param target_count  the request handle for the send call
     * @param operation  the reduce operation. See @MPI_Op
     *
     * @return  the request handle for the send call
     */
    template <typename GetType>
    request r_get_accumulate(GetType* origin_buffer, const int origin_count,
                             GetType* result_buffer, const int result_count,
                             const int target_rank,
                             const unsigned int target_disp,
                             const int target_count, MPI_Op operation) const
    {
        request req;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Rget_accumulate(
            origin_buffer, origin_count, type_impl<GetType>::get_type(),
            result_buffer, result_count, type_impl<GetType>::get_type(),
            target_rank, target_disp, target_count,
            type_impl<GetType>::get_type(), operation, this->get_window(),
            req.get()));
        return req;
    }

    /**
     * Fetch and operate on data from the target window (An optimized version of
     * Get_accumulate).
     *
     * @param origin_buffer  the buffer to send
     * @param target_rank  the rank to get the data from
     * @param target_disp  the displacement at the target window
     * @param operation  the reduce operation. See @MPI_Op
     */
    template <typename GetType>
    void fetch_and_op(GetType* origin_buffer, GetType* result_buffer,
                      const int target_rank, const unsigned int target_disp,
                      MPI_Op operation) const
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Fetch_and_op(
            origin_buffer, result_buffer, type_impl<GetType>::get_type(),
            target_rank, target_disp, operation, this->get_window()));
    }

private:
    MPI_Win window_;
};


}  // namespace mpi
}  // namespace gko


#endif  // GKO_HAVE_MPI


#endif  // GKO_PUBLIC_CORE_BASE_MPI_HPP_
