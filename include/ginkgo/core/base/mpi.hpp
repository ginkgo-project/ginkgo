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


#if GKO_HAVE_MPI


#include <mpi.h>


namespace gko {
namespace mpi {


/*
 * This enum is used for selecting the operation type for functions that take
 * MPI_Op. For example the MPI_Reduce operations.
 */
enum class op_type {
    sum = 1,
    min = 2,
    max = 3,
    product = 4,
    custom = 5,
    logical_and = 6,
    bitwise_and = 7,
    logical_or = 8,
    bitwise_or = 9,
    logical_xor = 10,
    bitwise_xor = 11,
    max_val_and_loc = 12,
    min_val_and_loc = 13
};


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

#define GKO_MPI_DATATYPE(BaseType, MPIType)                                  \
    inline MPI_Datatype get_mpi_type(const BaseType&) { return MPIType; }    \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


GKO_MPI_DATATYPE(bool, MPI_C_BOOL);
GKO_MPI_DATATYPE(char, MPI_CHAR);
GKO_MPI_DATATYPE(unsigned char, MPI_UNSIGNED_CHAR);
GKO_MPI_DATATYPE(unsigned, MPI_UNSIGNED);
GKO_MPI_DATATYPE(int, MPI_INT);
GKO_MPI_DATATYPE(unsigned long, MPI_UNSIGNED_LONG);
GKO_MPI_DATATYPE(unsigned short, MPI_UNSIGNED_SHORT);
GKO_MPI_DATATYPE(long, MPI_LONG);
GKO_MPI_DATATYPE(float, MPI_FLOAT);
GKO_MPI_DATATYPE(double, MPI_DOUBLE);
GKO_MPI_DATATYPE(long double, MPI_LONG_DOUBLE);
GKO_MPI_DATATYPE(std::complex<float>, MPI_C_COMPLEX);
GKO_MPI_DATATYPE(std::complex<double>, MPI_C_DOUBLE_COMPLEX);


template <typename ValueType>
MPI_Op get_operation(gko::mpi::op_type op)
{
    switch (op) {
    case gko::mpi::op_type::sum:
        return MPI_SUM;
    case gko::mpi::op_type::min:
        return MPI_MIN;
    case gko::mpi::op_type::max:
        return MPI_MAX;
    case gko::mpi::op_type::product:
        return MPI_PROD;
    case gko::mpi::op_type::logical_and:
        return MPI_LAND;
    case gko::mpi::op_type::bitwise_and:
        return MPI_BAND;
    case gko::mpi::op_type::logical_or:
        return MPI_LOR;
    case gko::mpi::op_type::bitwise_or:
        return MPI_BOR;
    case gko::mpi::op_type::logical_xor:
        return MPI_LXOR;
    case gko::mpi::op_type::bitwise_xor:
        return MPI_BXOR;
    case gko::mpi::op_type::max_val_and_loc:
        return MPI_MAXLOC;
    case gko::mpi::op_type::min_val_and_loc:
        return MPI_MINLOC;
    default:
        GKO_NOT_SUPPORTED(op);
    }
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
        auto flag = is_initialized();
        if (!flag) {
            this->required_thread_support_ = static_cast<int>(thread_t);
            GKO_ASSERT_NO_MPI_ERRORS(
                MPI_Init_thread(&argc, &argv, this->required_thread_support_,
                                &(this->provided_thread_support_)));
        } else {
            GKO_MPI_INITIALIZED;
        }
    }

    init_finalize() = delete;

    ~init_finalize()
    {
        auto flag = is_finalized();
        if (!flag) MPI_Finalize();
    }

private:
    int num_args_;
    int required_thread_support_;
    int provided_thread_support_;
    char** args_;
};


/**
 * A class holding and operating on the MPI_Info class. Stores the key value
 * pair as a map and provides methods to access these values with keys as
 * strings.
 */
class info {
public:
    info() : info_(MPI_INFO_NULL) {}

    explicit info(MPI_Info input_info)
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Info_dup(input_info, &this->info_));
    }

    void create_default()
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Info_create(&this->info_));
    }

    void remove(std::string key)
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Info_delete(this->info_, key.c_str()));
    }

    std::string& at(std::string& key) { return this->key_value_.at(key); }

    void add(std::string key, std::string value)
    {
        this->key_value_[key] = value;
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Info_set(this->info_, key.c_str(), value.c_str()));
    }

    MPI_Info get() { return this->info_; }

    ~info()
    {
        if (this->info_ != MPI_INFO_NULL) {
            MPI_Info_free(&this->info_);
        }
    }

private:
    std::map<std::string, std::string> key_value_;
    MPI_Info info_;
};


/**
 * A request class that takes in the given request and duplicates it
 * for our purposes. As the class or object goes out of scope, the request
 * is freed.
 */
class request : public EnableSharedCreateMethod<request> {
public:
    explicit request(const int size) : req_(new MPI_Request[size]) {}

    request() : req_(new MPI_Request[1]) {}

    void free(MPI_Request* req)
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Request_free(req));
    }

    ~request()
    {
        // this->free(this->req_);
        delete[] req_;
    }

    MPI_Request* get() const { return req_; }

private:
    MPI_Request* req_;
};


/**
 * A status class that allows creation of MPI_Status and
 * frees the status array when it  goes out of scope
 */
class status : public EnableSharedCreateMethod<status> {
public:
    status(const int size) : status_(new MPI_Status[size]) {}

    status() : status_(new MPI_Status[1]) {}

    ~status()
    {
        if (status_) delete[] status_;
    }

    MPI_Status* get() const { return status_; }

private:
    MPI_Status* status_;
};


/**
 * A communicator class that takes in the given communicator and duplicates it
 * for our purposes. As the class or object goes out of scope, the communicator
 * is freed.
 */
class communicator : public EnableSharedCreateMethod<communicator> {
public:
    communicator(const MPI_Comm& comm)
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_dup(comm, &this->comm_));
        this->size_ = get_num_ranks();
        this->rank_ = get_my_rank();
        this->local_rank_ = get_local_rank();
    }

    communicator(const MPI_Comm& comm, int color, int key)
    {
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Comm_split(comm, color, key, &this->comm_));
        this->size_ = get_num_ranks();
        this->rank_ = get_my_rank();
        this->local_rank_ = get_local_rank();
    }

    communicator()
    {
        this->comm_ = MPI_COMM_NULL;
        this->size_ = 0;
        this->rank_ = -1;
    }

    communicator(communicator& other)
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_dup(other.comm_, &this->comm_));
        this->size_ = get_num_ranks();
        this->rank_ = get_my_rank();
        this->local_rank_ = get_local_rank();
    }

    communicator& operator=(const communicator& other)
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_dup(other.comm_, &this->comm_));
        this->size_ = get_num_ranks();
        this->rank_ = get_my_rank();
        this->local_rank_ = get_local_rank();
        return *this;
    }

    communicator(communicator&& other)
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_dup(other.comm_, &this->comm_));
        this->size_ = get_num_ranks();
        this->rank_ = get_my_rank();
        this->local_rank_ = get_local_rank();
        other.comm_ = MPI_COMM_NULL;
        other.size_ = 0;
        other.rank_ = -1;
    }

    communicator& operator=(communicator&& other)
    {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_dup(other.comm_, &this->comm_));
        this->size_ = get_num_ranks();
        this->rank_ = get_my_rank();
        this->local_rank_ = get_local_rank();
        other.size_ = 0;
        other.rank_ = -1;
        return *this;
    }

    static MPI_Comm get_comm_world() { return MPI_COMM_WORLD; }

    static std::shared_ptr<communicator> create_world()
    {
        return std::make_shared<communicator>(get_comm_world());
    }

    MPI_Comm get() const { return comm_; }

    int size() const { return size_; }

    int rank() const { return rank_; };

    int local_rank() const { return local_rank_; };

    bool compare(const MPI_Comm& other) const
    {
        int flag;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_compare(this->comm_, other, &flag));
        return flag;
    }

    bool operator==(const communicator& rhs) { return compare(rhs.get()); }

    ~communicator()
    {
        if (this->comm_ && this->comm_ != MPI_COMM_NULL) {
            MPI_Comm_free(&this->comm_);
        }
    }

private:
    MPI_Comm comm_;
    int size_{};
    int rank_{};
    int local_rank_{};

    int get_my_rank()
    {
        int my_rank = 0;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_rank(comm_, &my_rank));
        return my_rank;
    }

    int get_local_rank()
    {
        MPI_Comm local_comm;
        int rank;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_split_type(
            comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm));
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_rank(local_comm, &rank));
        MPI_Comm_free(&local_comm);
        return rank;
    }

    int get_num_ranks()
    {
        int size = 1;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_size(comm_, &size));
        return size;
    }
};


/**
 * Get the rank in the communicator of the calling process.
 *
 * @param comm  the communicator
 */
static double get_walltime() { return MPI_Wtime(); }


/**
 * This function is used to synchronize between the ranks of a given
 * communicator.
 *
 * @param comm  the communicator
 */
static void synchronize(const std::shared_ptr<communicator>& comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Barrier(comm->get()));
}


/**
 * Allows a rank to wait on a particular request handle.
 *
 * @param req  The request to wait on.
 * @param status  The status variable that can be queried.
 */
static void wait(std::shared_ptr<request> req,
                 std::shared_ptr<status> status = {})
{
    if (status.get()) {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Wait(req->get(), status->get()));
    } else {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Wait(req->get(), MPI_STATUS_IGNORE));
    }
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
    window(window& other) = default;
    window& operator=(const window& other) = default;
    window(window&& other) = default;
    window& operator=(window&& other) = default;

    window(ValueType* base, unsigned int size,
           std::shared_ptr<const communicator> comm,
           const int disp_unit = sizeof(ValueType), info input_info = info(),
           win_type create_type = win_type::create)
    {
        if (create_type == win_type::create) {
            GKO_ASSERT_NO_MPI_ERRORS(
                MPI_Win_create(base, size, disp_unit, input_info.get(),
                               comm->get(), &this->window_));
        } else if (create_type == win_type::dynamic_create) {
            GKO_ASSERT_NO_MPI_ERRORS(MPI_Win_create_dynamic(
                input_info.get(), comm->get(), &this->window_));
        } else if (create_type == win_type::allocate) {
            GKO_ASSERT_NO_MPI_ERRORS(
                MPI_Win_allocate(size, disp_unit, input_info.get(), comm->get(),
                                 base, &this->window_));
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
void send(const SendType* send_buffer, const int send_count,
          const int destination_rank, const int send_tag,
          std::shared_ptr<const communicator> comm)
{
    auto send_type = detail::get_mpi_type(send_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Send(send_buffer, send_count, send_type,
                                      destination_rank, send_tag, comm->get()));
}


/**
 * Send (Non-blocking) data from calling process to destination rank.
 *
 * @param send_buffer  the buffer to send
 * @param send_count  the number of elements to send
 * @param destination_rank  the rank to send the data to
 * @param send_tag  the tag for the send call
 * @param req  the request handle for the send call
 * @param comm  the communicator
 */
template <typename SendType>
void send(const SendType* send_buffer, const int send_count,
          const int destination_rank, const int send_tag,
          std::shared_ptr<request> req,
          std::shared_ptr<const communicator> comm)
{
    auto send_type = detail::get_mpi_type(send_buffer[0]);

    GKO_ASSERT_NO_MPI_ERRORS(MPI_Isend(send_buffer, send_count, send_type,
                                       destination_rank, send_tag, comm->get(),
                                       req->get()));
}


/**
 * Receive data from source rank.
 *
 * @param recv_buffer  the buffer to send
 * @param recv_count  the number of elements to send
 * @param source_rank  the rank to send the data to
 * @param recv_tag  the tag for the send call
 * @param comm  the communicator
 */
template <typename RecvType>
void recv(RecvType* recv_buffer, const int recv_count, const int source_rank,
          const int recv_tag, std::shared_ptr<const communicator> comm,
          std::shared_ptr<status> status = {})
{
    auto recv_type = detail::get_mpi_type(recv_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Recv(recv_buffer, recv_count, recv_type, source_rank, recv_tag,
                 comm->get(), status ? status->get() : MPI_STATUS_IGNORE));
}


/**
 * Receive data from source rank.
 *
 * @param recv_buffer  the buffer to send
 * @param recv_count  the number of elements to send
 * @param source_rank  the rank to send the data to
 * @param recv_tag  the tag for the send call
 * @param req  the request handle for the send call
 * @param comm  the communicator
 */
template <typename RecvType>
void recv(RecvType* recv_buffer, const int recv_count, const int source_rank,
          const int recv_tag, std::shared_ptr<request> req,
          std::shared_ptr<const communicator> comm)
{
    auto recv_type = detail::get_mpi_type(recv_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Irecv(recv_buffer, recv_count, recv_type,
                                       source_rank, recv_tag, comm->get(),
                                       req->get()));
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
void put(const PutType* origin_buffer, const int origin_count,
         const int target_rank, const unsigned int target_disp,
         const int target_count, window<PutType>& window)
{
    auto put_type = detail::get_mpi_type(origin_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Put(origin_buffer, origin_count, put_type,
                                     target_rank, target_disp, target_count,
                                     put_type, window.get()));
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
 * @param req  the request handle
 */
template <typename PutType>
void put(const PutType* origin_buffer, const int origin_count,
         const int target_rank, const unsigned int target_disp,
         const int target_count, window<PutType>& window,
         std::shared_ptr<request> req)
{
    auto put_type = detail::get_mpi_type(origin_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Rput(origin_buffer, origin_count, put_type,
                                      target_rank, target_disp, target_count,
                                      put_type, window.get(), req->get()));
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
void get(GetType* origin_buffer, const int origin_count, const int target_rank,
         const unsigned int target_disp, const int target_count,
         window<GetType>& window)
{
    auto get_type = detail::get_mpi_type(origin_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Get(origin_buffer, origin_count, get_type,
                                     target_rank, target_disp, target_count,
                                     get_type, window.get()));
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
 * @param req  the request handle
 */
template <typename GetType>
void get(GetType* origin_buffer, const int origin_count, const int target_rank,
         const unsigned int target_disp, const int target_count,
         window<GetType>& window, std::shared_ptr<request> req)
{
    auto get_type = detail::get_mpi_type(origin_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Rget(origin_buffer, origin_count, get_type,
                                      target_rank, target_disp, target_count,
                                      get_type, window, req->get()));
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
void broadcast(BroadcastType* buffer, int count, int root_rank,
               std::shared_ptr<const communicator> comm)
{
    auto bcast_type = detail::get_mpi_type(buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Bcast(buffer, count, bcast_type, root_rank, comm->get()));
}


/**
 * Reduce data into root from all calling processes on the same communicator.
 *
 * @param send_buffer  the buffer to reduce
 * @param recv_buffer  the reduced result
 * @param count  the number of elements to reduce
 * @param op_enum  the reduce operation. See @op_type
 * @param comm  the communicator
 */
template <typename ReduceType>
void reduce(const ReduceType* send_buffer, ReduceType* recv_buffer, int count,
            op_type op_enum, int root_rank,
            std::shared_ptr<const communicator> comm)
{
    auto operation = detail::get_operation<ReduceType>(op_enum);
    auto reduce_type = detail::get_mpi_type(send_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Reduce(send_buffer, recv_buffer, count,
                                        reduce_type, operation, root_rank,
                                        comm->get()));
}


/**
 * Reduce data into root from all calling processes on the same communicator.
 *
 * @param send_buffer  the buffer to reduce
 * @param recv_buffer  the reduced result
 * @param count  the number of elements to reduce
 * @param op_enum  the reduce operation. See @op_type
 * @param comm  the communicator
 * @param req  the request handle
 */
template <typename ReduceType>
void reduce(const ReduceType* send_buffer, ReduceType* recv_buffer, int count,
            op_type op_enum, int root_rank,
            std::shared_ptr<const communicator> comm,
            std::shared_ptr<request> req)
{
    auto operation = detail::get_operation<ReduceType>(op_enum);
    auto reduce_type = detail::get_mpi_type(send_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Ireduce(send_buffer, recv_buffer, count,
                                         reduce_type, operation, root_rank,
                                         comm->get(), req->get()));
}


/**
 * Reduce data from all calling processes from all calling processes on same
 * communicator.
 *
 * @param recv_buffer  the data to reduce and the reduced result
 * @param count  the number of elements to reduce
 * @param op_enum  the reduce operation. See @op_type
 * @param comm  the communicator
 */
template <typename ReduceType>
void all_reduce(ReduceType* recv_buffer, int count, op_type op_enum,
                std::shared_ptr<const communicator> comm)
{
    auto operation = detail::get_operation<ReduceType>(op_enum);
    auto reduce_type = detail::get_mpi_type(recv_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Allreduce(detail::in_place<ReduceType>(),
                                           recv_buffer, count, reduce_type,
                                           operation, comm->get()));
}


/**
 * Reduce data from all calling processes from all calling processes on same
 * communicator.
 *
 * @param recv_buffer  the data to reduce and the reduced result
 * @param count  the number of elements to reduce
 * @param op_enum  the reduce operation. See @op_type
 * @param comm  the communicator
 * @param req  the request handle
 */
template <typename ReduceType>
void all_reduce(ReduceType* recv_buffer, int count, op_type op_enum,
                std::shared_ptr<const communicator> comm,
                std::shared_ptr<request> req)
{
    auto operation = detail::get_operation<ReduceType>(op_enum);
    auto reduce_type = detail::get_mpi_type(recv_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Iallreduce(detail::in_place<ReduceType>(), recv_buffer, count,
                       reduce_type, operation, comm->get(), req->get()));
}


/**
 * Reduce data from all calling processes from all calling processes on same
 * communicator.
 *
 * @param send_buffer  the data to reduce
 * @param recv_buffer  the reduced result
 * @param count  the number of elements to reduce
 * @param op_enum  the reduce operation. See @op_type
 * @param comm  the communicator
 * @param req  the request handle
 */
template <typename ReduceType>
void all_reduce(const ReduceType* send_buffer, ReduceType* recv_buffer,
                int count, op_type op_enum,
                std::shared_ptr<const communicator> comm)
{
    auto operation = detail::get_operation<ReduceType>(op_enum);
    auto reduce_type = detail::get_mpi_type(recv_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Allreduce(
        send_buffer, recv_buffer, count, reduce_type, operation, comm->get()));
}


/**
 * Reduce data from all calling processes from all calling processes on same
 * communicator.
 *
 * @param send_buffer  the data to reduce
 * @param recv_buffer  the reduced result
 * @param count  the number of elements to reduce
 * @param op_enum  the reduce operation. See @op_type
 * @param comm  the communicator
 * @param req  the request handle
 */
template <typename ReduceType>
void all_reduce(const ReduceType* send_buffer, ReduceType* recv_buffer,
                int count, op_type op_enum,
                std::shared_ptr<const communicator> comm,
                std::shared_ptr<request> req)
{
    auto operation = detail::get_operation<ReduceType>(op_enum);
    auto reduce_type = detail::get_mpi_type(recv_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Iallreduce(send_buffer, recv_buffer, count,
                                            reduce_type, operation, comm->get(),
                                            req->get()));
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
void gather(const SendType* send_buffer, const int send_count,
            RecvType* recv_buffer, const int recv_count, int root_rank,
            std::shared_ptr<const communicator> comm)
{
    auto send_type = detail::get_mpi_type(send_buffer[0]);
    auto recv_type = detail::get_mpi_type(recv_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Gather(send_buffer, send_count, send_type,
                                        recv_buffer, recv_count, recv_type,
                                        root_rank, comm->get()));
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
void gatherv(const SendType* send_buffer, const int send_count,
             RecvType* recv_buffer, const int* recv_counts,
             const int* displacements, int root_rank,
             std::shared_ptr<const communicator> comm)
{
    auto send_type = detail::get_mpi_type(send_buffer[0]);
    auto recv_type = detail::get_mpi_type(recv_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Gatherv(
        send_buffer, send_count, send_type, recv_buffer, recv_counts,
        displacements, recv_type, root_rank, comm->get()));
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
void all_gather(const SendType* send_buffer, const int send_count,
                RecvType* recv_buffer, const int recv_count,
                std::shared_ptr<const communicator> comm)
{
    auto send_type = detail::get_mpi_type(send_buffer[0]);
    auto recv_type = detail::get_mpi_type(recv_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Allgather(send_buffer, send_count, send_type,
                                           recv_buffer, recv_count, recv_type,
                                           comm->get()));
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
void scatter(const SendType* send_buffer, const int send_count,
             RecvType* recv_buffer, const int recv_count, int root_rank,
             std::shared_ptr<const communicator> comm)
{
    auto send_type = detail::get_mpi_type(send_buffer[0]);
    auto recv_type = detail::get_mpi_type(recv_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Scatter(send_buffer, send_count, send_type,
                                         recv_buffer, recv_count, recv_type,
                                         root_rank, comm->get()));
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
void scatterv(const SendType* send_buffer, const int* send_counts,
              const int* displacements, RecvType* recv_buffer,
              const int recv_count, int root_rank,
              std::shared_ptr<const communicator> comm)
{
    auto send_type = detail::get_mpi_type(send_buffer[0]);
    auto recv_type = detail::get_mpi_type(recv_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Scatterv(
        send_buffer, send_counts, displacements, send_type, recv_buffer,
        recv_count, recv_type, root_rank, comm->get()));
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
void all_to_all(RecvType* recv_buffer, const int recv_count,
                std::shared_ptr<const communicator> comm)
{
    auto recv_type = detail::get_mpi_type(recv_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Alltoall(detail::in_place<RecvType>(),
                                          recv_count, recv_type, recv_buffer,
                                          recv_count, recv_type, comm->get()));
}


/**
 * Communicate data from all ranks to all other ranks in place (MPI_Alltoall).
 * See MPI documentation for more details.
 *
 * @param buffer  the buffer to send and the buffer receive
 * @param recv_count  the number of elements to receive
 * @param comm  the communicator
 * @param req  the request handle
 *
 * @note This overload uses MPI_IN_PLACE and the source and destination buffers
 *       are the same.
 */
template <typename RecvType>
void all_to_all(RecvType* recv_buffer, const int recv_count,
                std::shared_ptr<const communicator> comm,
                std::shared_ptr<request> req)
{
    auto recv_type = detail::get_mpi_type(recv_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Ialltoall(
        detail::in_place<RecvType>(), recv_count, recv_type, recv_buffer,
        recv_count, recv_type, comm->get(), req->get()));
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
void all_to_all(const SendType* send_buffer, const int send_count,
                RecvType* recv_buffer, const int recv_count,
                std::shared_ptr<const communicator> comm)
{
    auto send_type = detail::get_mpi_type(send_buffer[0]);
    auto recv_type = detail::get_mpi_type(recv_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Alltoall(send_buffer, send_count, send_type,
                                          recv_buffer, recv_count, recv_type,
                                          comm->get()));
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
 * @param req  the request handle
 */
template <typename SendType, typename RecvType>
void all_to_all(const SendType* send_buffer, const int send_count,
                RecvType* recv_buffer, const int recv_count,
                std::shared_ptr<const communicator> comm,
                std::shared_ptr<request> req)
{
    auto send_type = detail::get_mpi_type(send_buffer[0]);
    auto recv_type = detail::get_mpi_type(recv_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Ialltoall(send_buffer, send_count, send_type,
                                           recv_buffer, recv_count, recv_type,
                                           comm->get(), req->get()));
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
 * @param stride  the stride to be used in case of sending concatenated data
 * @param comm  the communicator
 */
template <typename SendType, typename RecvType>
void all_to_all_v(const SendType* send_buffer, const int* send_counts,
                  const int* send_offsets, RecvType* recv_buffer,
                  const int* recv_counts, const int* recv_offsets,
                  const int stride, std::shared_ptr<const communicator> comm)
{
    auto send_type = detail::get_mpi_type(send_buffer[0]);
    auto recv_type = detail::get_mpi_type(recv_buffer[0]);

    GKO_ASSERT_NO_MPI_ERRORS(MPI_Alltoallv(
        send_buffer, send_counts, send_offsets, send_type, recv_buffer,
        recv_counts, recv_offsets, recv_type, comm->get()));
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
 * @param stride  the stride to be used in case of sending concatenated data
 * @param comm  the communicator
 * @param req  the request handle
 */
template <typename SendType, typename RecvType>
void all_to_all_v(const SendType* send_buffer, const int* send_counts,
                  const int* send_offsets, RecvType* recv_buffer,
                  const int* recv_counts, const int* recv_offsets,
                  const int stride, std::shared_ptr<const communicator> comm,
                  std::shared_ptr<request> req)
{
    auto send_type = detail::get_mpi_type(send_buffer[0]);
    auto recv_type = detail::get_mpi_type(recv_buffer[0]);

    GKO_ASSERT_NO_MPI_ERRORS(MPI_Ialltoallv(
        send_buffer, send_counts, send_offsets, send_type, recv_buffer,
        recv_counts, recv_offsets, recv_type, comm->get(), req->get()));
}


/**
 * Does a scan operation with the given operator.
 * (MPI_Scan). See MPI documentation for more details.
 *
 * @param send_buffer  the buffer to scan from
 * @param recv_buffer  the result buffer
 * @param recv_count  the number of elements to scan
 * @param op_enum  the operation type to be used for the scan. See @op_type
 * @param comm  the communicator
 * @param req  the request handle
 */
template <typename ScanType>
void scan(const ScanType* send_buffer, ScanType* recv_buffer, int count,
          op_type op_enum, std::shared_ptr<const communicator> comm)
{
    auto operation = detail::get_operation<ScanType>(op_enum);
    auto scan_type = detail::get_mpi_type(recv_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Scan(send_buffer, recv_buffer, count,
                                      scan_type, operation, comm->get()));
}


/**
 * Does a scan operation with the given operator.
 * (MPI_Scan). See MPI documentation for more details.
 *
 * @param send_buffer  the buffer to scan from
 * @param recv_buffer  the result buffer
 * @param recv_count  the number of elements to scan
 * @param op_enum  the operation type to be used for the scan. See @op_type
 * @param comm  the communicator
 * @param req  the request handle
 */
template <typename ScanType>
void scan(const ScanType* send_buffer, ScanType* recv_buffer, int count,
          op_type op_enum, std::shared_ptr<const communicator> comm,
          std::shared_ptr<request> req)
{
    auto operation = detail::get_operation<ScanType>(op_enum);
    auto scan_type = detail::get_mpi_type(recv_buffer[0]);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Iscan(send_buffer, recv_buffer, count,
                                       scan_type, operation, comm->get(),
                                       req->get()));
}


}  // namespace mpi
}  // namespace gko


#endif  // GKO_HAVE_MPI


#endif  // GKO_PUBLIC_CORE_BASE_MPI_HPP_
