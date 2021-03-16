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

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_COMMUNICATOR_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_COMMUNICATOR_HPP_


#include <ginkgo/config.hpp>


#if GKO_HAVE_MPI


#include <complex>


#include <mpi.h>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace distributed {


template <typename T>
struct mpi_type {};

#define GKO_REGISTER_MPI_TYPE(_type, _type_val, _mult)    \
    template <>                                           \
    struct mpi_type<_type> {                              \
        static MPI_Datatype value() { return _type_val; } \
        static constexpr auto multiplier = _mult;         \
    }


GKO_REGISTER_MPI_TYPE(char, MPI_CHAR, 1);
GKO_REGISTER_MPI_TYPE(unsigned char, MPI_UNSIGNED_CHAR, 1);
GKO_REGISTER_MPI_TYPE(short, MPI_SHORT, 1);
GKO_REGISTER_MPI_TYPE(unsigned short, MPI_UNSIGNED_SHORT, 1);
GKO_REGISTER_MPI_TYPE(int, MPI_INT, 1);
GKO_REGISTER_MPI_TYPE(unsigned int, MPI_UNSIGNED, 1);
GKO_REGISTER_MPI_TYPE(long int, MPI_LONG, 1);
GKO_REGISTER_MPI_TYPE(unsigned long int, MPI_UNSIGNED_LONG, 1);
GKO_REGISTER_MPI_TYPE(long long int, MPI_LONG_LONG_INT, 1);
GKO_REGISTER_MPI_TYPE(float, MPI_FLOAT, 1);
GKO_REGISTER_MPI_TYPE(double, MPI_DOUBLE, 1);
GKO_REGISTER_MPI_TYPE(std::complex<float>, MPI_FLOAT, 2);
GKO_REGISTER_MPI_TYPE(std::complex<double>, MPI_DOUBLE, 2);


#undef GKO_REGISTER_MPI_TYPE


class communicator {
public:
    using index_type = int;

    communicator() : communicator{MPI_COMM_WORLD} {}

    explicit communicator(MPI_Comm comm) : comm_{comm} {}

    index_type rank() const
    {
        index_type rank{};
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_rank(comm_, &rank));
        return rank;
    }

    index_type size() const
    {
        index_type size{};
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_size(comm_, &size));
        return size;
    }

    template <typename T>
    void broadcast(T *data, index_type size, index_type root) const
    {
        size *= mpi_type<T>::multiplier;
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Bcast(data, size, mpi_type<T>::value(), root, comm_));
    }

    template <typename T>
    void reduce(const T *in, T *out, index_type size, index_type root,
                MPI_Op op = MPI_SUM) const
    {
        size *= mpi_type<T>::multiplier;
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Reduce(in, out, size, mpi_type<T>::value(), op, comm_));
    }

    template <typename T>
    void reduce(T *data, index_type size, index_type root,
                MPI_Op op = MPI_SUM) const
    {
        reduce(inplace<T>(), data, size, root, op);
    }

    template <typename T>
    void allreduce(const T *in, T *out, index_type size,
                   MPI_Op op = MPI_SUM) const
    {
        size *= mpi_type<T>::multiplier;
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Allreduce(in, out, size, mpi_type<T>::value(), MPI_SUM, comm_));
    }

    template <typename T>
    void allreduce(T *data, index_type size, MPI_Op op = MPI_SUM) const
    {
        allreduce(inplace<T>(), data, size, op);
    }

    template <typename T>
    void alltoall(const T *in, T *out, index_type size) const
    {
        auto type = mpi_type<T>::value();
        size *= mpi_type<T>::multiplier;
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Alltoall(in, size, type, out, size, type, comm_));
    }

    template <typename T>
    void alltoall(T *data, index_type size) const
    {
        alltoall(inplace<T>(), data, size);
    }

    template <typename T>
    void alltoallv(const T *in, const index_type *send_counts,
                   const index_type *send_offsets, T *out,
                   const index_type *recv_counts,
                   const index_type *recv_offsets, index_type size) const
    {
        auto type = mpi_type<T>::value();
        size *= mpi_type<T>::multiplier;
        MPI_Datatype new_type{};
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Type_contiguous(size, type, &new_type));
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Type_commit(&new_type));
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Alltoallv(in, send_counts, send_offsets,
                                               new_type, out, recv_counts,
                                               recv_offsets, new_type, comm_));
        // TODO this leaks in case of errors, use RAII
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Type_free(&new_type));
    }

private:
    template <typename T>
    const T *inplace() const
    {
        return reinterpret_cast<const T *>(MPI_IN_PLACE);
    }

    MPI_Comm comm_;
};


}  // namespace distributed
}  // namespace gko

#endif  // GKO_HAVE_MPI
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_COMMUNICATOR_HPP_
