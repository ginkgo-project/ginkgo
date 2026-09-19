// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_DISTRIBUTED_HELPERS_HPP_
#define GKO_CORE_DISTRIBUTED_HELPERS_HPP_


#include <memory>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/collective_communicator.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/dispatch_helper.hpp"


namespace gko {
namespace detail {


template <typename ValueType>
std::unique_ptr<matrix::Dense<ValueType>> create_with_config_of(
    const matrix::Dense<ValueType>* mtx)
{
    return matrix::Dense<ValueType>::create(mtx->get_executor(),
                                            mtx->get_size(), mtx->get_stride());
}


template <typename ValueType>
const matrix::Dense<ValueType>* get_local(const matrix::Dense<ValueType>* mtx)
{
    return mtx;
}


template <typename ValueType>
matrix::Dense<ValueType>* get_local(matrix::Dense<ValueType>* mtx)
{
    return mtx;
}


#if GINKGO_BUILD_MPI


template <typename ValueType>
std::unique_ptr<experimental::distributed::Vector<ValueType>>
create_with_config_of(const experimental::distributed::Vector<ValueType>* mtx)
{
    return experimental::distributed::Vector<ValueType>::create(
        mtx->get_executor(), mtx->get_communicator(), mtx->get_size(),
        mtx->get_local_vector()->get_size(),
        mtx->get_local_vector()->get_stride());
}


template <typename ValueType>
matrix::Dense<ValueType>* get_local(
    experimental::distributed::Vector<ValueType>* mtx)
{
    return const_cast<matrix::Dense<ValueType>*>(mtx->get_local_vector());
}


template <typename ValueType>
const matrix::Dense<ValueType>* get_local(
    const experimental::distributed::Vector<ValueType>* mtx)
{
    return mtx->get_local_vector();
}


#endif


template <typename Arg>
bool is_distributed(Arg* linop)
{
#if GINKGO_BUILD_MPI
    return dynamic_cast<const experimental::distributed::DistributedBase*>(
        linop);
#else
    return false;
#endif
}


template <typename Arg, typename... Rest>
bool is_distributed(Arg* linop, Rest*... rest)
{
#if GINKGO_BUILD_MPI
    bool is_distributed_value =
        dynamic_cast<const experimental::distributed::DistributedBase*>(linop);
    GKO_ASSERT(is_distributed_value == is_distributed(rest...));
    return is_distributed_value;
#else
    return false;
#endif
}


/**
 * Cast an input linop to the correct underlying vector type (dense/distributed)
 * and passes it to the given function.
 *
 * @tparam ValueType  The value type of the underlying dense or distributed
 * vector.
 * @tparam T  The linop type, either LinOp, or const LinOp.
 * @tparam F  The function type.
 * @tparam Args  The types for the additional arguments of f.
 *
 * @param linop  The linop to be casted into either a dense or distributed
 *               vector.
 * @param f  The function that is to be called with the correctly casted linop.
 * @param args  The additional arguments of f.
 */
template <typename ValueType, typename T, typename F, typename... Args>
void vector_dispatch(T* linop, F&& f, Args&&... args)
{
#if GINKGO_BUILD_MPI
    if (is_distributed(linop)) {
        using type = std::conditional_t<
            std::is_const<T>::value,
            const experimental::distributed::Vector<ValueType>,
            experimental::distributed::Vector<ValueType>>;
        f(dynamic_cast<type*>(linop), std::forward<Args>(args)...);
    } else
#endif
    {
        using type = std::conditional_t<std::is_const<T>::value,
                                        const matrix::Dense<ValueType>,
                                        matrix::Dense<ValueType>>;
        if (auto concrete_linop = dynamic_cast<type*>(linop)) {
            f(concrete_linop, std::forward<Args>(args)...);
        } else {
            GKO_NOT_SUPPORTED(linop);
        }
    }
}


#if GINKGO_BUILD_MPI


/**
 * Specialization of run for distributed matrices.
 */
template <typename T, typename F, typename... Args>
auto run_matrix(T* linop, F&& f, Args&&... args)
{
    using namespace gko::experimental::distributed;
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


#if GINKGO_BUILD_MPI


/**
 * Reusable buffers for exchange_with_neighbors, so that a loop which exchanges
 * in every iteration allocates only once.
 */
template <typename ValueType>
struct neighbor_exchange_buffers {
    explicit neighbor_exchange_buffers(std::shared_ptr<const Executor> exec)
        : recv{exec},
          host_send{exec->get_master()},
          host_recv{exec->get_master()}
    {}

    array<ValueType> recv;
    array<ValueType> host_send;
    array<ValueType> host_recv;
};


/**
 * Exchanges one value per halo index with the neighboring ranks.
 *
 * Distributed coarsening schemes use this to communicate a single value per
 * non-local index, such as the aggregate an index was assigned to (Pgm), or
 * the measure and the C/F status of a node (Pmis).
 *
 * The direction is fixed: the send buffer must match the communicator's send
 * size. To send halo contributions back to their owners, pass a communicator
 * obtained from CollectiveCommunicator::create_inverse(), which swaps the send
 * and receive roles.
 *
 * @param send_buffer  one value per send index of the collective communicator,
 *                     in its send index order
 * @param buffers      reusable buffers, shared across calls
 *
 * @return buffers.recv, one value per receive index in receive order
 */
template <typename ValueType>
const array<ValueType>& exchange_with_neighbors(
    std::shared_ptr<const Executor> exec,
    const experimental::mpi::communicator& comm,
    const experimental::mpi::CollectiveCommunicator* coll_comm,
    const array<ValueType>& send_buffer,
    neighbor_exchange_buffers<ValueType>& buffers)
{
    const auto total_send_size =
        static_cast<size_type>(coll_comm->get_send_size());
    const auto total_recv_size =
        static_cast<size_type>(coll_comm->get_recv_size());
    GKO_ASSERT_EQ(send_buffer.get_size(), total_send_size);
    buffers.recv.resize_and_reset(total_recv_size);

    // not every executor/MPI combination can send from device memory
    auto use_host_buffer = experimental::mpi::requires_host_buffer(exec, comm);
    if (use_host_buffer) {
        buffers.host_send.resize_and_reset(total_send_size);
        buffers.host_recv.resize_and_reset(total_recv_size);
        exec->get_master()->copy_from(exec, total_send_size,
                                      send_buffer.get_const_data(),
                                      buffers.host_send.get_data());
    }

    const auto send_ptr = use_host_buffer ? buffers.host_send.get_const_data()
                                          : send_buffer.get_const_data();
    auto recv_ptr = use_host_buffer ? buffers.host_recv.get_data()
                                    : buffers.recv.get_data();
    exec->synchronize();
    coll_comm
        ->i_all_to_all_v(use_host_buffer ? exec->get_master() : exec, send_ptr,
                         recv_ptr)
        .wait();
    if (use_host_buffer) {
        exec->copy_from(exec->get_master(), total_recv_size, recv_ptr,
                        buffers.recv.get_data());
    }
    return buffers.recv;
}


/**
 * Overload that allocates its own buffers.
 */
template <typename ValueType>
array<ValueType> exchange_with_neighbors(
    std::shared_ptr<const Executor> exec,
    const experimental::mpi::communicator& comm,
    const experimental::mpi::CollectiveCommunicator* coll_comm,
    const array<ValueType>& send_buffer)
{
    neighbor_exchange_buffers<ValueType> buffers{exec};
    exchange_with_neighbors(exec, comm, coll_comm, send_buffer, buffers);
    return std::move(buffers.recv);
}


#endif  // GINKGO_BUILD_MPI


}  // namespace detail
}  // namespace gko


#endif  // GKO_CORE_DISTRIBUTED_HELPERS_HPP_
