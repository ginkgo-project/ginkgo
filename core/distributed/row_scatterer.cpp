// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/row_scatterer.hpp"

#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/event.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/distributed/dense_communicator.hpp>
#include <ginkgo/core/distributed/neighborhood_communicator.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/base/event_kernels.hpp"

namespace gko {
namespace experimental {
namespace distributed {


namespace event {
namespace {
GKO_REGISTER_OPERATION(record_event, event::record_event);
}
}  // namespace event


template <typename LocalIndexType>
mpi::request RowScatterer<LocalIndexType>::apply_async(
    ptr_param<const LinOp> local_values) const
{
    mpi::request req;
    auto exec = this->get_executor();
    auto use_host_buffer =
        mpi::requires_host_buffer(exec, coll_comm_->get_base_communicator());
    auto mpi_exec = use_host_buffer ? exec->get_master() : exec;

    // Dispatch on local_values as a distributed::Vector
    run<Vector,
#if GINKGO_ENABLE_HALF
        half, std::complex<half>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        bfloat16, std::complex<bfloat16>,
#endif
        double, float, std::complex<double>, std::complex<float>>(
        make_temporary_clone(exec, local_values).get(),
        [&](const auto* lv_global) {
            using ValueType =
                typename std::decay_t<decltype(*lv_global)>::value_type;
            distributed::precision_dispatch<ValueType>([&]() {
                auto lv_local = lv_global->get_local_vector();
                auto ncols = lv_local->get_size()[1];

                // Pack send buffer
                dim<2> send_size(coll_comm_->get_send_size(), ncols);
                auto send_size_in_bytes =
                    sizeof(ValueType) * send_size[0] * send_size[1];
                if (!send_workspace_.get_executor() ||
                    !mpi_exec->memory_accessible(
                        send_workspace_.get_executor())) {
                    send_workspace_.set_executor(mpi_exec);
                }
                if (send_size_in_bytes > send_workspace_.get_size()) {
                    send_workspace_.resize_and_reset(send_size_in_bytes);
                }
                auto send_buffer = matrix::Dense<ValueType>::create(
                    mpi_exec, send_size,
                    make_array_view(mpi_exec, send_size[0] * send_size[1],
                                    reinterpret_cast<ValueType*>(
                                        send_workspace_.get_data())),
                    send_size[1]);
                lv_local->convert_to(send_buffer);

                // Allocate recv buffer
                dim<2> recv_size(coll_comm_->get_recv_size(), ncols);
                auto recv_size_in_bytes =
                    sizeof(ValueType) * recv_size[0] * recv_size[1];
                if (!recv_workspace_.get_executor() ||
                    !mpi_exec->memory_accessible(
                        recv_workspace_.get_executor())) {
                    recv_workspace_.set_executor(mpi_exec);
                }
                if (recv_size_in_bytes > recv_workspace_.get_size()) {
                    recv_workspace_.resize_and_reset(recv_size_in_bytes);
                }

                // Synchronize before MPI (GPU stream safety)
                std::shared_ptr<const gko::detail::Event> ev = nullptr;
                lv_local->get_executor()->run(event::make_record_event(ev));
                ev->synchronize();

                // Start async MPI communication
                auto send_ptr = send_buffer->get_values();
                auto recv_ptr =
                    reinterpret_cast<ValueType*>(recv_workspace_.get_data());
                mpi::contiguous_type type(
                    ncols, mpi::type_impl<ValueType>::get_type());
                req = coll_comm_->i_all_to_all_v(mpi_exec, send_ptr, type.get(),
                                                 recv_ptr, type.get());
            });
        });
    return req;
}


template <typename LocalIndexType>
mpi::request RowScatterer<LocalIndexType>::apply_async(
    ptr_param<const LinOp> weights, ptr_param<const LinOp> local_values) const
{
    mpi::request req;
    auto exec = this->get_executor();
    auto use_host_buffer =
        mpi::requires_host_buffer(exec, coll_comm_->get_base_communicator());
    auto mpi_exec = use_host_buffer ? exec->get_master() : exec;

    run<Vector,
#if GINKGO_ENABLE_HALF
        half, std::complex<half>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        bfloat16, std::complex<bfloat16>,
#endif
        double, float, std::complex<double>, std::complex<float>>(
        make_temporary_clone(exec, local_values).get(),
        [&](const auto* lv_global) {
            using ValueType =
                typename std::decay_t<decltype(*lv_global)>::value_type;
            distributed::precision_dispatch<ValueType>([&]() {
                auto lv_local = lv_global->get_local_vector();
                auto ncols = lv_local->get_size()[1];

                // Pack send buffer
                dim<2> send_size(coll_comm_->get_send_size(), ncols);
                auto send_size_in_bytes =
                    sizeof(ValueType) * send_size[0] * send_size[1];
                if (!send_workspace_.get_executor() ||
                    !mpi_exec->memory_accessible(
                        send_workspace_.get_executor())) {
                    send_workspace_.set_executor(mpi_exec);
                }
                if (send_size_in_bytes > send_workspace_.get_size()) {
                    send_workspace_.resize_and_reset(send_size_in_bytes);
                }
                auto send_buffer = matrix::Dense<ValueType>::create(
                    mpi_exec, send_size,
                    make_array_view(mpi_exec, send_size[0] * send_size[1],
                                    reinterpret_cast<ValueType*>(
                                        send_workspace_.get_data())),
                    send_size[1]);
                lv_local->convert_to(send_buffer);

                // Apply weights element-wise
                auto dense_weights = gko::as<matrix::Dense<ValueType>>(
                    make_temporary_clone(exec, weights).get());
                GKO_THROW_IF_INVALID(
                    dense_weights->get_size() == lv_local->get_size(),
                    "The weights dimensions must match local_values "
                    "dimensions.");
                auto dense_weights_on_mpi =
                    make_temporary_clone(mpi_exec, dense_weights);
                for (size_type row = 0; row < send_size[0]; ++row) {
                    for (size_type col = 0; col < send_size[1]; ++col) {
                        send_buffer->at(row, col) *=
                            dense_weights_on_mpi->at(row, col);
                    }
                }

                // Allocate recv buffer
                dim<2> recv_size(coll_comm_->get_recv_size(), ncols);
                auto recv_size_in_bytes =
                    sizeof(ValueType) * recv_size[0] * recv_size[1];
                if (!recv_workspace_.get_executor() ||
                    !mpi_exec->memory_accessible(
                        recv_workspace_.get_executor())) {
                    recv_workspace_.set_executor(mpi_exec);
                }
                if (recv_size_in_bytes > recv_workspace_.get_size()) {
                    recv_workspace_.resize_and_reset(recv_size_in_bytes);
                }

                // Synchronize before MPI
                std::shared_ptr<const gko::detail::Event> ev = nullptr;
                lv_local->get_executor()->run(event::make_record_event(ev));
                ev->synchronize();

                // Start async MPI communication
                auto send_ptr = send_buffer->get_values();
                auto recv_ptr =
                    reinterpret_cast<ValueType*>(recv_workspace_.get_data());
                mpi::contiguous_type type(
                    ncols, mpi::type_impl<ValueType>::get_type());
                req = coll_comm_->i_all_to_all_v(mpi_exec, send_ptr, type.get(),
                                                 recv_ptr, type.get());
            });
        });
    return req;
}


template <typename LocalIndexType>
void RowScatterer<LocalIndexType>::wait_and_accumulate(
    mpi::request& req, ptr_param<LinOp> distributed_target) const
{
    req.wait();

    auto exec = this->get_executor();
    auto use_host_buffer =
        mpi::requires_host_buffer(exec, coll_comm_->get_base_communicator());
    auto mpi_exec = use_host_buffer ? exec->get_master() : exec;

    // Dispatch on the distributed target to get ValueType
    run<Vector,
#if GINKGO_ENABLE_HALF
        half, std::complex<half>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        bfloat16, std::complex<bfloat16>,
#endif
        double, float, std::complex<double>, std::complex<float>>(
        distributed_target.get(), [&](auto* target_global) {
            using ValueType =
                typename std::decay_t<decltype(*target_global)>::value_type;

            auto target_local = target_global->get_local_vector();
            auto target_dense =
                const_cast<matrix::Dense<ValueType>*>(target_local);
            auto ncols = target_dense->get_size()[1];

            dim<2> recv_size(coll_comm_->get_recv_size(), ncols);

            // If recv data is on host (non-GPU-aware MPI), copy to device
            std::unique_ptr<matrix::Dense<ValueType>> recv_on_exec;
            if (use_host_buffer) {
                auto host_recv = matrix::Dense<ValueType>::create(
                    mpi_exec, recv_size,
                    make_array_view(mpi_exec, recv_size[0] * recv_size[1],
                                    reinterpret_cast<ValueType*>(
                                        recv_workspace_.get_data())),
                    recv_size[1]);
                recv_on_exec =
                    matrix::Dense<ValueType>::create(exec, recv_size);
                recv_on_exec->copy_from(host_recv);
            } else {
                recv_on_exec = matrix::Dense<ValueType>::create(
                    exec, recv_size,
                    make_array_view(exec, recv_size[0] * recv_size[1],
                                    reinterpret_cast<ValueType*>(
                                        recv_workspace_.get_data())),
                    recv_size[1]);
            }

            // Accumulate using Dense::scatter_add
            target_dense->scatter_add(&recv_idxs_, recv_on_exec.get());
        });
}


template <typename LocalIndexType>
dim<2> RowScatterer<LocalIndexType>::get_size() const
{
    return size_;
}


template <typename LocalIndexType>
std::shared_ptr<const mpi::CollectiveCommunicator>
RowScatterer<LocalIndexType>::get_collective_communicator() const
{
    return coll_comm_;
}


template <typename T>
static T global_add(std::shared_ptr<const Executor> exec,
                    const mpi::communicator& comm, const T& value)
{
    T result;
    comm.all_reduce(std::move(exec), &value, &result, 1, MPI_SUM);
    return result;
}


template <typename LocalIndexType>
template <typename GlobalIndexType>
RowScatterer<LocalIndexType>::RowScatterer(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm,
    const index_map<LocalIndexType, GlobalIndexType>& imap)
    : EnablePolymorphicObject<RowScatterer>(exec),
      DistributedBase(coll_comm->get_base_communicator()),
      size_(dim<2>{global_add(exec, coll_comm->get_base_communicator(),
                              imap.get_non_local_size()),
                   imap.get_global_size()}),
      coll_comm_(std::move(coll_comm)),
      recv_idxs_(exec),
      send_workspace_(exec),
      recv_workspace_(exec)
{
    GKO_THROW_IF_INVALID(
        coll_comm_->get_recv_size() == imap.get_non_local_size(),
        "The collective communicator doesn't match the index map.");

    auto comm = coll_comm_->get_base_communicator();
    auto inverse_comm = coll_comm_->create_inverse();

    auto mpi_exec =
        mpi::requires_host_buffer(exec, coll_comm_->get_base_communicator())
            ? exec->get_master()
            : exec;
    auto temp_remote_local_idxs =
        make_temporary_clone(mpi_exec, &imap.get_remote_local_idxs());

    recv_idxs_.set_executor(mpi_exec);
    recv_idxs_.resize_and_reset(coll_comm_->get_send_size());
    inverse_comm
        ->i_all_to_all_v(exec, temp_remote_local_idxs->get_const_flat_data(),
                         recv_idxs_.get_data())
        .wait();
    recv_idxs_.set_executor(exec);

    // Use the inverse comm for the actual scatter operation
    coll_comm_ = coll_comm_->create_inverse();
}


template <typename LocalIndexType>
std::unique_ptr<RowScatterer<LocalIndexType>>
RowScatterer<LocalIndexType>::create_from_gatherer(
    std::shared_ptr<const Executor> exec,
    const RowGatherer<LocalIndexType>& gatherer)
{
    auto inverse_comm =
        gatherer.get_collective_communicator()->create_inverse();

    // The recv_idxs_ for the scatterer are the send_idxs_ of the gatherer
    auto num_send_idxs = gatherer.get_num_send_idxs();
    array<LocalIndexType> recv_idxs(exec, num_send_idxs);
    if (num_send_idxs > 0) {
        exec->copy_from(gatherer.get_executor(), num_send_idxs,
                        gatherer.get_const_send_idxs(), recv_idxs.get_data());
    }

    // Size is the transpose of the gatherer's size
    auto gatherer_size = gatherer.get_size();
    dim<2> size{gatherer_size[1], gatherer_size[0]};

    return std::unique_ptr<RowScatterer>(new RowScatterer(
        std::move(exec), std::move(inverse_comm), std::move(recv_idxs), size));
}


template <typename LocalIndexType>
std::unique_ptr<RowScatterer<LocalIndexType>>
RowScatterer<LocalIndexType>::create(std::shared_ptr<const Executor> exec,
                                     mpi::communicator comm)
{
    return std::unique_ptr<RowScatterer>(
        new RowScatterer(std::move(exec), comm));
}


template <typename LocalIndexType>
RowScatterer<LocalIndexType>::RowScatterer(std::shared_ptr<const Executor> exec,
                                           mpi::communicator comm)
    : EnablePolymorphicObject<RowScatterer>(exec),
      DistributedBase(comm),
      coll_comm_(mpi::detail::create_default_collective_communicator(comm)),
      recv_idxs_(exec),
      send_workspace_(exec),
      recv_workspace_(exec)
{}


template <typename LocalIndexType>
RowScatterer<LocalIndexType>::RowScatterer(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm,
    array<LocalIndexType> recv_idxs, dim<2> size)
    : EnablePolymorphicObject<RowScatterer>(exec),
      DistributedBase(coll_comm->get_base_communicator()),
      size_(size),
      coll_comm_(std::move(coll_comm)),
      recv_idxs_(std::move(recv_idxs)),
      send_workspace_(exec),
      recv_workspace_(exec)
{}


template <typename LocalIndexType>
RowScatterer<LocalIndexType>::RowScatterer(RowScatterer&& o) noexcept
    : EnablePolymorphicObject<RowScatterer>(o.get_executor()),
      DistributedBase(o.get_communicator()),
      recv_idxs_(o.get_executor()),
      send_workspace_(o.get_executor()),
      recv_workspace_(o.get_executor())
{
    *this = std::move(o);
}


template <typename LocalIndexType>
RowScatterer<LocalIndexType>& RowScatterer<LocalIndexType>::operator=(
    const RowScatterer& o)
{
    if (this != &o) {
        size_ = o.get_size();
        coll_comm_ = o.coll_comm_;
        recv_idxs_ = o.recv_idxs_;
    }
    return *this;
}


template <typename LocalIndexType>
RowScatterer<LocalIndexType>& RowScatterer<LocalIndexType>::operator=(
    RowScatterer&& o)
{
    if (this != &o) {
        size_ = std::exchange(o.size_, dim<2>{});
        coll_comm_ = std::exchange(
            o.coll_comm_, mpi::detail::create_default_collective_communicator(
                              o.get_communicator()));
        recv_idxs_ = std::move(o.recv_idxs_);
        send_workspace_ = std::move(o.send_workspace_);
        recv_workspace_ = std::move(o.recv_workspace_);
    }
    return *this;
}


template <typename LocalIndexType>
RowScatterer<LocalIndexType>::RowScatterer(const RowScatterer& o)
    : EnablePolymorphicObject<RowScatterer>(o.get_executor()),
      DistributedBase(o.get_communicator()),
      recv_idxs_(o.get_executor()),
      send_workspace_(o.get_executor()),
      recv_workspace_(o.get_executor())
{
    *this = o;
}


#define GKO_DECLARE_ROW_SCATTERER(IndexType) class RowScatterer<IndexType>

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ROW_SCATTERER);

#undef GKO_DECLARE_ROW_SCATTERER


#define GKO_DECLARE_ROW_SCATTERER_CONSTRUCTOR(_ltype, _gtype)         \
    RowScatterer<_ltype>::RowScatterer(                               \
        std::shared_ptr<const Executor> exec,                         \
        std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm, \
        const index_map<_ltype, _gtype>& imap)

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_ROW_SCATTERER_CONSTRUCTOR);

#undef GKO_DECLARE_ROW_SCATTERER_CONSTRUCTOR
}  // namespace distributed
}  // namespace experimental
}  // namespace gko
