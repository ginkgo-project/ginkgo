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

#ifndef GKO_MPI_HELPERS_HPP_
#define GKO_MPI_HELPERS_HPP_

#include <functional>

#include <mpi.h>

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
/**
 * @brief The helpers namespace.
 *
 * @ingroup helper
 */
namespace helpers {
/**
 * @brief The MPI namespace.
 *
 * @ingroup mpi
 */
namespace mpi {

#define GKO_MPI_DATATYPE(BaseType, MPIType)                                  \
    inline MPI_Datatype get_mpi_type(const BaseType &) { return MPIType; }   \
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
GKO_MPI_DATATYPE(std::complex<float>, MPI_COMPLEX);
GKO_MPI_DATATYPE(std::complex<double>, MPI_DOUBLE_COMPLEX);


namespace detail {
namespace operations {

template <typename ValueType>
void custom(void *in, void *inout, int *size, MPI_Datatype *)
{
    auto l_in = reinterpret_cast<ValueType *>(in);
    auto l_inout = reinterpret_cast<ValueType *>(inout);
    ValueType sum = 0.0;
    for (auto i = 0; i < *size; ++i) {
    }
    *l_inout = (*l_in);
}


}  // namespace operations
}  // namespace detail


template <typename ValueType>
MPI_Op get_operation(gko::mpi::op_type op)
{
    switch (op) {
    case gko::mpi::op_type::sum:
        return MPI_SUM;
        break;
    case gko::mpi::op_type::min:
        return MPI_MIN;
        break;
    case gko::mpi::op_type::max:
        return MPI_MAX;
        break;
    case gko::mpi::op_type::product:
        return MPI_PROD;
        break;
    case gko::mpi::op_type::custom: {
        // TEMPLATE to create custom operations
        MPI_Op op;
        // auto op = std::unique_ptr<MPI_Op, std::function<void(T
        // *)>>([](MPI_Op){}); GKO_ASSERT_NO_MPI_ERRORS(
        //     MPI_Op_create(detail::operations::custom<ValueType>, true, &op));
        return op;
        break;
    }
    case gko::mpi::op_type::logical_and:
        return MPI_LAND;
        break;
    case gko::mpi::op_type::bitwise_and:
        return MPI_BAND;
        break;
    case gko::mpi::op_type::logical_or:
        return MPI_LOR;
        break;
    case gko::mpi::op_type::bitwise_or:
        return MPI_BOR;
        break;
    case gko::mpi::op_type::logical_xor:
        return MPI_LXOR;
        break;
    case gko::mpi::op_type::bitwise_xor:
        return MPI_BXOR;
        break;
    case gko::mpi::op_type::max_val_and_loc:
        return MPI_MAXLOC;
        break;
    case gko::mpi::op_type::min_val_and_loc:
        return MPI_MINLOC;
        break;
    default:
        GKO_NOT_SUPPORTED(op);
        break;
    }
}


}  // namespace mpi
}  // namespace helpers
}  // namespace gko


#endif  // GKO_MPI_HELPERS_HPP_
