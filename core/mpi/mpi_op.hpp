// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MPI_MPI_OP_HPP_
#define GKO_CORE_MPI_MPI_OP_HPP_

#include <complex>
#include <type_traits>


#if GINKGO_BUILD_MPI


#include <mpi.h>


namespace gko {
namespace experimental {
namespace mpi {
namespace detail {


template <typename ValueType>
inline void sum(void* input, void* output, int* len, MPI_Datatype* datatype)
{
    ValueType* input_ptr = static_cast<ValueType*>(input);
    ValueType* output_ptr = static_cast<ValueType*>(output);
    for (int i = 0; i < *len; i++) {
        output_ptr[i] += input_ptr[i];
    }
}

template <typename ValueType>
inline void max(void* input, void* output, int* len, MPI_Datatype* datatype)
{
    ValueType* input_ptr = static_cast<ValueType*>(input);
    ValueType* output_ptr = static_cast<ValueType*>(output);
    for (int i = 0; i < *len; i++) {
        if (input_ptr[i] > output_ptr[i]) {
            output_ptr[i] = input_ptr[i];
        }
    }
}


template <typename ValueType>
struct is_mpi_native {
    constexpr static bool value =
        std::is_arithmetic_v<ValueType> ||
        std::is_same_v<ValueType, std::complex<float>> ||
        std::is_same_v<ValueType, std::complex<double>>;
};


}  // namespace detail


using op_manager = std::shared_ptr<MPI_Op>;


// ginkgo custom mpi sum operation implementation
template <typename ValueType, typename = void>
class sum {};

template <typename ValueType>
class sum<ValueType,
          std::enable_if_t<detail::is_mpi_native<ValueType>::value>> {
public:
    MPI_Op get_op() { return MPI_SUM; }
};

template <typename ValueType>
class sum<ValueType,
          std::enable_if_t<!detail::is_mpi_native<ValueType>::value>> {
public:
    sum()
    {
        op_ = op_manager(
            []() {
                MPI_Op* operation = new MPI_Op;
                MPI_Op_create(&detail::sum<ValueType>, 1, operation);
                return operation;
            }(),
            [](MPI_Op* op) {
                MPI_Op_free(op);
                delete op;
            });
    }

    MPI_Op get_op() { return *op_.get(); }

private:
    op_manager op_;
};


// ginkgo custom mpi max operation implementation
template <typename ValueType, typename = void>
class max {};

template <typename ValueType>
class max<ValueType,
          std::enable_if_t<detail::is_mpi_native<ValueType>::value>> {
public:
    MPI_Op get_op() { return MPI_MAX; }
};

template <typename ValueType>
class max<ValueType,
          std::enable_if_t<!detail::is_mpi_native<ValueType>::value>> {
public:
    max()
    {
        op_ = op_manager(
            []() {
                MPI_Op* operation = new MPI_Op;
                MPI_Op_create(&detail::max<ValueType>, 1, operation);
                return operation;
            }(),
            [](MPI_Op* op) {
                MPI_Op_free(op);
                delete op;
            });
    }

    MPI_Op get_op() { return *op_.get(); }

private:
    op_manager op_;
};


}  // namespace mpi
}  // namespace experimental
}  // namespace gko

#endif
#endif  // GKO_CORE_MPI_MPI_OP_HPP_
