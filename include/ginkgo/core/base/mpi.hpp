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


#define GKO_REGISTER_MPI_TYPE(input_type, mpi_type)          \
    template <>                                              \
    constexpr MPI_Datatype type_impl<input_type>::get_type() \
    {                                                        \
        return mpi_type;                                     \
    }


template <typename T>
struct type_impl {
    constexpr static MPI_Datatype get_type() { return MPI_DATATYPE_NULL; }
};


GKO_REGISTER_MPI_TYPE(char, MPI_CHAR);
GKO_REGISTER_MPI_TYPE(unsigned char, MPI_UNSIGNED_CHAR);
GKO_REGISTER_MPI_TYPE(unsigned, MPI_UNSIGNED);
GKO_REGISTER_MPI_TYPE(int, MPI_INT);
GKO_REGISTER_MPI_TYPE(unsigned short, MPI_UNSIGNED_SHORT);
GKO_REGISTER_MPI_TYPE(unsigned long, MPI_UNSIGNED_LONG);
GKO_REGISTER_MPI_TYPE(long, MPI_LONG);
GKO_REGISTER_MPI_TYPE(float, MPI_FLOAT);
GKO_REGISTER_MPI_TYPE(double, MPI_DOUBLE);
GKO_REGISTER_MPI_TYPE(long double, MPI_LONG_DOUBLE);
GKO_REGISTER_MPI_TYPE(std::complex<float>, MPI_C_COMPLEX);
GKO_REGISTER_MPI_TYPE(std::complex<double>, MPI_C_DOUBLE_COMPLEX);


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
    int get_provided_thread_support() { return provided_thread_support_; }

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

    environment() = delete;

private:
    int required_thread_support_;
    int provided_thread_support_;
};


/**
 * Returns if GPU aware functionality has been enabled
 */
static bool is_gpu_aware()
{
#if GKO_HAVE_GPU_AWARE_MPI
    return true;
#else
    return false;
#endif
}


namespace {


/**
 * A deleter class that calls MPI_Comm_free on the owning MPI_Comm object
 */
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
    /**
     * Non-owning constructor for an existing communicator of type MPI_Comm. The
     * MPI_Comm object will not be deleted after the communicator object has
     * been freed and an explicit MPI_Comm_free needs to be called on the
     * original MPI_Comm_free object.
     *
     * @param comm The input MPI_Comm object.
     */
    communicator(const MPI_Comm& comm)
    {
        this->comm_ =
            comm_manager(new MPI_Comm(comm), null_deleter<MPI_Comm>{});
        this->size_ = get_num_ranks();
        this->rank_ = get_my_rank();
        this->node_local_rank_ = get_node_local_rank();
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
        this->comm_ = comm_manager(new MPI_Comm(comm_out), comm_deleter{});
        this->size_ = get_num_ranks();
        this->rank_ = get_my_rank();
        this->node_local_rank_ = get_node_local_rank();
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
        this->comm_ = comm_manager(new MPI_Comm(comm_out), comm_deleter{});
        this->size_ = get_num_ranks();
        this->rank_ = get_my_rank();
        this->node_local_rank_ = get_node_local_rank();
    }

    /**
     * Copy constructor. The underlying MPI_Comm object will be duplicated.
     *
     * @param other  the object to be copied
     */
    communicator(communicator& other)
    {
        MPI_Comm comm;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_dup(other.get(), &comm));
        this->comm_ = comm_manager(new MPI_Comm(comm), comm_deleter{});
        this->size_ = other.size_;
        this->rank_ = other.rank_;
        this->node_local_rank_ = other.node_local_rank_;
    }

    /**
     * Copy assignment operator. The underlying MPI_Comm object will be
     * duplicated.
     *
     * @param other  the object to be copied
     */
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

    /**
     * Move constructor. If we own the underlying communicator, then we move the
     * object over. If we don't, then we throw.
     *
     * @param other  the object to be moved
     */
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

    /**
     * Move assignment operator. If we own the underlying communicator, then we
     * move the object over. If we don't, then we throw.
     *
     * @param other  the object to be moved
     */
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

    /**
     * Duplicate and create an owning communicator from an input MPI_Comm
     * object.
     *
     * @param comm_in  the input MPI_Comm object to be duplicated
     */
    static communicator duplicate(const MPI_Comm& comm_in)
    {
        MPI_Comm comm;
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_dup(comm_in, &comm));
        communicator comm_out(comm);
        return comm_out;
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
    int size() const { return size_; }

    /**
     * Return the rank of the calling process in the communicator.
     *
     * @return  the rank
     */
    int rank() const { return rank_; };

    /**
     * Return the node local rank of the calling process in the communicator.
     *
     * @return  the node local rank
     */
    int node_local_rank() const { return node_local_rank_; };

    /**
     * Compare two communicator objects.
     *
     * @return  if the two comm objects are equal
     */
    bool operator==(const communicator& rhs) { return compare(rhs.get()); }

    /**
     * Check if the underlying comm object is owned
     *
     * @return  if the underlying comm object is owned
     */
    bool is_owning()
    {
        return comm_.get_deleter().target_type() == typeid(comm_deleter);
    }

    /**
     * This function is used to synchronize the ranks in the communicator.
     * Calls MPI_Barrier
     */
    void synchronize() const { GKO_ASSERT_NO_MPI_ERRORS(MPI_Barrier(get())); }

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
 * Allows a rank to wait on multiple request handles.
 *
 * @param req  The request handles to wait on.
 * @param status  The status variable that can be queried.
 */
inline std::vector<MPI_Status> wait_all(std::vector<MPI_Request>& req)
{
    std::vector<MPI_Status> status(req.size());
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
    MPI_Win get() { return this->window_; }

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
     * @param assert  the optimization level. 0 is always valid.
     */
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
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Send(send_buffer, send_count,
                                      type_impl<SendType>::get_type(),
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
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Isend(send_buffer, send_count, type_impl<SendType>::get_type(),
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
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Recv(recv_buffer, recv_count, type_impl<RecvType>::get_type(),
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
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Irecv(recv_buffer, recv_count, type_impl<RecvType>::get_type(),
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
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Put(origin_buffer, origin_count, type_impl<PutType>::get_type(),
                target_rank, target_disp, target_count,
                type_impl<PutType>::get_type(), window.get()));
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
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Rput(origin_buffer, origin_count, type_impl<PutType>::get_type(),
                 target_rank, target_disp, target_count,
                 type_impl<PutType>::get_type(), window.get(), &req));
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
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Get(origin_buffer, origin_count, type_impl<GetType>::get_type(),
                target_rank, target_disp, target_count,
                type_impl<GetType>::get_type(), window.get()));
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
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Rget(origin_buffer, origin_count, type_impl<GetType>::get_type(),
                 target_rank, target_disp, target_count,
                 type_impl<GetType>::get_type(), window, &req));
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
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Bcast(buffer, count,
                                       type_impl<BroadcastType>::get_type(),
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
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Reduce(send_buffer, recv_buffer, count,
                                        type_impl<ReduceType>::get_type(),
                                        operation, root_rank, comm.get()));
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
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Ireduce(
        send_buffer, recv_buffer, count, type_impl<ReduceType>::get_type(),
        operation, root_rank, comm.get(), &req));
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
        in_place<ReduceType>(), recv_buffer, count,
        type_impl<ReduceType>::get_type(), operation, comm.get()));
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
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Iallreduce(
        in_place<ReduceType>(), recv_buffer, count,
        type_impl<ReduceType>::get_type(), operation, comm.get(), &req));
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
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Allreduce(send_buffer, recv_buffer, count,
                                           type_impl<ReduceType>::get_type(),
                                           operation, comm.get()));
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
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Iallreduce(send_buffer, recv_buffer, count,
                                            type_impl<ReduceType>::get_type(),
                                            operation, comm.get(), &req));
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
        send_buffer, send_count, type_impl<SendType>::get_type(), recv_buffer,
        recv_count, type_impl<RecvType>::get_type(), root_rank, comm.get()));
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
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Gatherv(send_buffer, send_count, type_impl<SendType>::get_type(),
                    recv_buffer, recv_counts, displacements,
                    type_impl<RecvType>::get_type(), root_rank, comm.get()));
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
        send_buffer, send_count, type_impl<SendType>::get_type(), recv_buffer,
        recv_count, type_impl<RecvType>::get_type(), comm.get()));
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
        send_buffer, send_count, type_impl<SendType>::get_type(), recv_buffer,
        recv_count, type_impl<RecvType>::get_type(), root_rank, comm.get()));
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
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Scatterv(send_buffer, send_counts, displacements,
                     type_impl<SendType>::get_type(), recv_buffer, recv_count,
                     type_impl<RecvType>::get_type(), root_rank, comm.get()));
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
        in_place<RecvType>(), recv_count, type_impl<RecvType>::get_type(),
        recv_buffer, recv_count, type_impl<RecvType>::get_type(), comm.get()));
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
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Ialltoall(in_place<RecvType>(), recv_count,
                      type_impl<RecvType>::get_type(), recv_buffer, recv_count,
                      type_impl<RecvType>::get_type(), comm.get(), &req));
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
        send_buffer, send_count, type_impl<SendType>::get_type(), recv_buffer,
        recv_count, type_impl<RecvType>::get_type(), comm.get()));
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
        send_buffer, send_count, type_impl<SendType>::get_type(), recv_buffer,
        recv_count, type_impl<RecvType>::get_type(), comm.get(), &req));
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
        send_buffer, send_counts, send_offsets, type_impl<SendType>::get_type(),
        recv_buffer, recv_counts, recv_offsets, type_impl<RecvType>::get_type(),
        comm.get()));
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
        send_buffer, send_counts, send_offsets, type_impl<SendType>::get_type(),
        recv_buffer, recv_counts, recv_offsets, type_impl<RecvType>::get_type(),
        comm.get(), &req));
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
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Scan(send_buffer, recv_buffer, count,
                                      type_impl<ScanType>::get_type(),
                                      operation, comm.get()));
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
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Iscan(send_buffer, recv_buffer, count,
                                       type_impl<ScanType>::get_type(),
                                       operation, comm.get(), &req));
    return req;
}


}  // namespace mpi
}  // namespace gko


#endif  // GKO_HAVE_MPI


#endif  // GKO_PUBLIC_CORE_BASE_MPI_HPP_
