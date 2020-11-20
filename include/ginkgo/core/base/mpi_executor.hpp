/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_CORE_BASE_MPI_EXECUTOR_HPP_
#define GKO_CORE_BASE_MPI_EXECUTOR_HPP_


#include <mpi/mpi.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>

namespace gko {


template <typename T>
struct mpi_type_impl {
    static MPI_Datatype type() { return MPI_CHAR; }
    constexpr static int multiplier = sizeof(T);
};

template <>
struct mpi_type_impl<char> {
    static MPI_Datatype type() { return MPI_CHAR; }
    constexpr static int multiplier = 1;
};

template <>
struct mpi_type_impl<unsigned char> {
    static MPI_Datatype type() { return MPI_UNSIGNED_CHAR; }
    constexpr static int multiplier = 1;
};

template <>
struct mpi_type_impl<short> {
    static MPI_Datatype type() { return MPI_SHORT; }
    constexpr static int multiplier = 1;
};

template <>
struct mpi_type_impl<unsigned short> {
    static MPI_Datatype type() { return MPI_UNSIGNED_SHORT; }
    constexpr static int multiplier = 1;
};

template <>
struct mpi_type_impl<int> {
    static MPI_Datatype type() { return MPI_INT; }
    constexpr static int multiplier = 1;
};

template <>
struct mpi_type_impl<unsigned int> {
    static MPI_Datatype type() { return MPI_UNSIGNED; }
    constexpr static int multiplier = 1;
};

template <>
struct mpi_type_impl<long> {
    static MPI_Datatype type() { return MPI_LONG; }
    constexpr static int multiplier = 1;
};

template <>
struct mpi_type_impl<unsigned long> {
    static MPI_Datatype type() { return MPI_UNSIGNED_LONG; }
    constexpr static int multiplier = 1;
};

template <>
struct mpi_type_impl<long long> {
    static MPI_Datatype type() { return MPI_LONG_LONG; }
    constexpr static int multiplier = 1;
};

template <>
struct mpi_type_impl<unsigned long long> {
    static MPI_Datatype type() { return MPI_UNSIGNED_LONG_LONG; }
    constexpr static int multiplier = 1;
};

template <>
struct mpi_type_impl<float> {
    static MPI_Datatype type() { return MPI_FLOAT; }
    constexpr static int multiplier = 1;
};

template <>
struct mpi_type_impl<double> {
    static MPI_Datatype type() { return MPI_DOUBLE; }
    constexpr static int multiplier = 1;
};

template <typename T>
struct mpi_type_impl<std::complex<T>> {
    static MPI_Datatype type() { return mpi_type_impl<T>::value; }
    constexpr static int multiplier = 2;
};


class MpiExecutor : public Executor,
                    public std::enable_shared_from_this<MpiExecutor> {
public:
    /**
     * Creates a new MpiExecutor.
     */
    static std::shared_ptr<MpiExecutor> create(
        std::shared_ptr<Executor> local_exec, MPI_Comm comm = MPI_COMM_WORLD)
    {
        return std::shared_ptr<MpiExecutor>(
            new MpiExecutor(std::move(local_exec), comm));
    }

    void run(const Operation &op) const override
    {
        this->template log<log::Logger::operation_launched>(this, &op);
        local_exec_->run(op);
        this->template log<log::Logger::operation_completed>(this, &op);
    }

    std::shared_ptr<Executor> get_master() noexcept override
    {
        if (master_exec_) {
            return master_exec_;
        } else {
            return this->shared_from_this();
        }
    }

    std::shared_ptr<const Executor> get_master() const noexcept override
    {
        if (master_exec_) {
            return master_exec_;
        } else {
            return this->shared_from_this();
        }
    }

    std::shared_ptr<Executor> get_local() noexcept { return local_exec_; }

    std::shared_ptr<const Executor> get_local() const noexcept
    {
        return local_exec_;
    }

    int get_rank() const
    {
        int rank{};
        MPI_Comm_rank(this->get_comm(), &rank);
        return rank;
    }

    int get_size() const
    {
        int size{};
        MPI_Comm_size(this->get_comm(), &size);
        return size;
    }

    MPI_Comm get_comm() const { return comm_; }

    void synchronize() const override { MPI_Barrier(this->get_comm()); }

    template <typename T>
    void broadcast(T *data, size_type size, int root) const
    {
        auto type = mpi_type_impl<T>::type();
        size *= mpi_type_impl<T>::multiplier;
        MPI_Bcast(data, size, type, root, this->get_comm());
    }

    template <typename T>
    void allreduce(T *data, size_type size) const
    {
        auto type = mpi_type_impl<T>::type();
        size *= mpi_type_impl<T>::multiplier;
        MPI_Allreduce(MPI_IN_PLACE, data, size, type, MPI_SUM,
                      this->get_comm());
    }

    template <typename T>
    void alltoall(const T *in, T *out, size_type size) const
    {
        auto type = mpi_type_impl<T>::type();
        size *= mpi_type_impl<T>::multiplier;
        MPI_Alltoall(in, size, type, out, size, type, this->get_comm());
    }

    template <typename T>
    void allgather(const T *in, T *out, size_type size) const
    {
        auto type = mpi_type_impl<T>::type();
        size *= mpi_type_impl<T>::multiplier;
        MPI_Allgather(in, size, type, out, size, type, this->get_comm());
    }

    template <typename T>
    void alltoallv(const T *in, T *out, size_type multiplier,
                   const int *send_ofs, const int *recv_ofs) const
    {
        auto type = mpi_type_impl<T>::type();
        multiplier *= mpi_type_impl<T>::multiplier;
        auto mpi_size = this->get_size();
        int tag = 0;
        std::vector<MPI_Request> requests(2 * mpi_size);
        for (int src_rank = 0; src_rank < mpi_size; ++src_rank) {
            auto recv_begin = recv_ofs[src_rank];
            auto recv_size = recv_ofs[src_rank + 1] - recv_begin;
            MPI_Irecv(out + recv_begin, recv_size * multiplier, type, src_rank,
                      tag, this->get_comm(), &requests[src_rank]);
        }
        for (int dst_rank = 0; dst_rank < mpi_size; ++dst_rank) {
            auto recv_begin = recv_ofs[dst_rank];
            auto recv_size = recv_ofs[dst_rank + 1] - recv_begin;
            MPI_Isend(in + recv_begin, recv_size * multiplier, type, dst_rank,
                      tag, this->get_comm(), &requests[dst_rank + mpi_size]);
        }
        std::vector<MPI_Status> statuses(2 * mpi_size);
        MPI_Waitall(2 * mpi_size, requests.data(), statuses.data());
    }

protected:
    MpiExecutor(std::shared_ptr<Executor> local_exec, MPI_Comm comm)
        : local_exec_{std::move(local_exec)}, comm_{comm}
    {
        auto local_master = local_exec_->get_master();
        if (local_master != local_exec_) {
            master_exec_ = create(local_master, comm);
        }
    }

    void raw_copy_from(const Executor *src_exec, size_type n_bytes,
                       const void *src_ptr, void *dest_ptr) const override
    {
        auto src_mpi_exec = as<MpiExecutor>(src_exec);
        local_exec_->copy_from(src_mpi_exec->local_exec_.get(), n_bytes,
                               static_cast<const char *>(src_ptr),
                               static_cast<char *>(dest_ptr));
    }

    void raw_copy_to(const OmpExecutor *dst_exec, size_type, const void *,
                     void *) const override
    {
        GKO_NOT_SUPPORTED(dst_exec);
    }

    void raw_copy_to(const CudaExecutor *dst_exec, size_type, const void *,
                     void *) const override
    {
        GKO_NOT_SUPPORTED(dst_exec);
    }

    void raw_copy_to(const HipExecutor *dst_exec, size_type, const void *,
                     void *) const override
    {
        GKO_NOT_SUPPORTED(dst_exec);
    }

    void raw_copy_to(const DpcppExecutor *dst_exec, size_type, const void *,
                     void *) const override
    {
        GKO_NOT_SUPPORTED(dst_exec);
    }

    void *raw_alloc(size_type size) const override
    {
        return local_exec_->alloc<char>(size);
    }

    void raw_free(void *ptr) const noexcept override
    {
        return local_exec_->free(ptr);
    }

private:
    std::shared_ptr<Executor> master_exec_;
    std::shared_ptr<Executor> local_exec_;
    MPI_Comm comm_;
};


}  // namespace gko

#endif  // GKO_CORE_BASE_MPI_EXECUTOR_HPP_