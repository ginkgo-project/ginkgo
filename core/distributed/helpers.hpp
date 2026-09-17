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
 * Exchanges one value per halo index with the neighboring ranks.
 *
 * Distributed coarsenings need to know what the owner of a non-local index
 * did with it: Pgm sends the aggregate an index was assigned to, Rs sends the
 * coarse index a forced C-point was renumbered to. Both are a single value per
 * halo index, so both reduce to this exchange.
 *
 * @param send_buffer  one value per send index of the collective communicator,
 *                     in its send index order
 *
 * @return one value per receive index, in the communicator's receive order
 */
template <typename ValueType>
array<ValueType> exchange_with_neighbors(
    std::shared_ptr<const Executor> exec,
    const experimental::mpi::communicator& comm,
    const experimental::mpi::CollectiveCommunicator* coll_comm,
    const array<ValueType>& send_buffer)
{
    const auto total_send_size =
        static_cast<size_type>(coll_comm->get_send_size());
    const auto total_recv_size =
        static_cast<size_type>(coll_comm->get_recv_size());
    GKO_ASSERT_EQ(send_buffer.get_size(), total_send_size);
    array<ValueType> recv_buffer(exec, total_recv_size);

    // not every executor/MPI combination can send from device memory
    auto use_host_buffer = experimental::mpi::requires_host_buffer(exec, comm);
    array<ValueType> host_send_buffer(exec->get_master());
    array<ValueType> host_recv_buffer(exec->get_master());
    if (use_host_buffer) {
        host_send_buffer.resize_and_reset(total_send_size);
        host_recv_buffer.resize_and_reset(total_recv_size);
        exec->get_master()->copy_from(exec, total_send_size,
                                      send_buffer.get_const_data(),
                                      host_send_buffer.get_data());
    }

    const auto send_ptr = use_host_buffer ? host_send_buffer.get_const_data()
                                          : send_buffer.get_const_data();
    auto recv_ptr =
        use_host_buffer ? host_recv_buffer.get_data() : recv_buffer.get_data();
    exec->synchronize();
    coll_comm
        ->i_all_to_all_v(use_host_buffer ? exec->get_master() : exec, send_ptr,
                         recv_ptr)
        .wait();
    if (use_host_buffer) {
        exec->copy_from(exec->get_master(), total_recv_size, recv_ptr,
                        recv_buffer.get_data());
    }
    return recv_buffer;
}


#endif


}  // namespace detail
}  // namespace gko


#endif  // GKO_CORE_DISTRIBUTED_HELPERS_HPP_
