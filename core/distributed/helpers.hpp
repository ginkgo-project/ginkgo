// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_DISTRIBUTED_HELPERS_HPP_
#define GKO_CORE_DISTRIBUTED_HELPERS_HPP_


#include <memory>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/multivector.hpp>

#include "core/base/dispatch_helper.hpp"

#if GINKGO_BUILD_MPI
#include <ginkgo/core/base/mpi.hpp>

#include "core/mpi/mpi_op.hpp"
#endif


namespace gko {
namespace experimental {
namespace distributed {
namespace detail {
#if GINKGO_BUILD_MPI


template <typename ValueType>
matrix::MultiVector<ValueType>* get_local_mutable(Vector<ValueType>* mtx)
{
    return const_cast<matrix::MultiVector<ValueType>*>(mtx->get_local_vector());
}


template <typename ValueType>
const matrix::MultiVector<ValueType>* get_local(const Vector<ValueType>* mtx)
{
    return mtx->get_local_vector();
}


/**
 * Sums the entries of `values` over all ranks of `comm` in place. If the MPI
 * implementation cannot access the memory of the executor of `values`, the
 * data is communicated through `host_buffer`.
 *
 * @param comm  the communicator
 * @param values  the values to reduce, stored contiguously (stride equal to
 *                the number of columns)
 * @param host_buffer  the host buffer used if necessary
 */
template <typename ValueType>
void all_reduce_sum(const experimental::mpi::communicator& comm,
                    matrix::MultiVector<ValueType>* values,
                    const gko::detail::DenseCache<ValueType>& host_buffer)
{
    GKO_ASSERT(values->get_stride() == values->get_size()[1] ||
               values->get_size()[0] <= 1);
    auto exec = values->get_executor();
    const auto count =
        static_cast<int>(values->get_size()[0] * values->get_size()[1]);
    auto sum_op = experimental::mpi::sum<ValueType>();
    exec->synchronize();
    if (experimental::mpi::requires_host_buffer(exec, comm)) {
        host_buffer.init(exec->get_master(), values->get_size());
        host_buffer->copy_from(values);
        comm.all_reduce(exec->get_master(), host_buffer->get_values(), count,
                        sum_op.get_op());
        values->copy_from(host_buffer.get());
    } else {
        comm.all_reduce(exec, values->get_values(), count, sum_op.get_op());
    }
}


#endif


template <typename Arg>
bool is_distributed(Arg* linop)
{
#if GINKGO_BUILD_MPI
    return dynamic_cast<const DistributedBase*>(linop);
#else
    return false;
#endif
}


template <typename Arg, typename... Rest>
bool is_distributed(Arg* linop, Rest*... rest)
{
#if GINKGO_BUILD_MPI
    bool is_distributed_value = dynamic_cast<const DistributedBase*>(linop);
    GKO_ASSERT(is_distributed_value == is_distributed(rest...));
    return is_distributed_value;
#else
    return false;
#endif
}


#if GINKGO_BUILD_MPI


/**
 * Specialization of run for distributed matrices.
 */
template <typename T, typename F, typename... Args>
auto run_matrix(T* linop, F&& f, Args&&... args)
{
    using namespace gko::detail;
    return run<
        with_same_constness_t<Matrix<double, int32, int32>, T>,
        with_same_constness_t<Matrix<double, int32, int64>, T>,
        with_same_constness_t<Matrix<double, int64, int64>, T>,
        with_same_constness_t<Matrix<float, int32, int32>, T>,
        with_same_constness_t<Matrix<float, int32, int64>, T>,
        with_same_constness_t<Matrix<float, int64, int64>, T>,
#if GINKGO_ENABLE_HALF
        with_same_constness_t<Matrix<float16, int32, int32>, T>,
        with_same_constness_t<Matrix<float16, int32, int64>, T>,
        with_same_constness_t<Matrix<float16, int64, int64>, T>,
        with_same_constness_t<Matrix<std::complex<float16>, int32, int32>, T>,
        with_same_constness_t<Matrix<std::complex<float16>, int32, int64>, T>,
        with_same_constness_t<Matrix<std::complex<float16>, int64, int64>, T>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        with_same_constness_t<Matrix<bfloat16, int32, int32>, T>,
        with_same_constness_t<Matrix<bfloat16, int32, int64>, T>,
        with_same_constness_t<Matrix<bfloat16, int64, int64>, T>,
        with_same_constness_t<Matrix<std::complex<bfloat16>, int32, int32>, T>,
        with_same_constness_t<Matrix<std::complex<bfloat16>, int32, int64>, T>,
        with_same_constness_t<Matrix<std::complex<bfloat16>, int64, int64>, T>,
#endif
        with_same_constness_t<Matrix<std::complex<double>, int32, int32>, T>,
        with_same_constness_t<Matrix<std::complex<double>, int32, int64>, T>,
        with_same_constness_t<Matrix<std::complex<double>, int64, int64>, T>,
        with_same_constness_t<Matrix<std::complex<float>, int32, int32>, T>,
        with_same_constness_t<Matrix<std::complex<float>, int32, int64>, T>,
        with_same_constness_t<Matrix<std::complex<float>, int64, int64>, T>>(
        linop, std::forward<F>(f), std::forward<Args>(args)...);
}


#endif


inline const LinOp* get_local(const LinOp* mtx)
{
#if GINKGO_BUILD_MPI
    if (is_distributed(mtx)) {
        return run_matrix(mtx, [](auto concrete) {
            return concrete->get_diag_matrix().get();
        });
    }
#endif
    {
        return mtx;
    }
}


}  // namespace detail
}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GKO_CORE_DISTRIBUTED_HELPERS_HPP_
