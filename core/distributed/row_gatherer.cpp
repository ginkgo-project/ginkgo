// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/row_gatherer.hpp"

#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/distributed/dense_communicator.hpp>
#include <ginkgo/core/distributed/neighborhood_communicator.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/dispatch_helper.hpp"

namespace gko {
namespace experimental {
namespace distributed {


#if GINKGO_HAVE_OPENMPI_PRE_4_1_X
using DefaultCollComm = mpi::DenseCommunicator;
#else
using DefaultCollComm = mpi::NeighborhoodCommunicator;
#endif


template <typename LocalIndexType>
mpi::request RowGatherer<LocalIndexType>::apply_async(ptr_param<const LinOp> b,
                                                      ptr_param<LinOp> x) const
{
    int is_inactive;
    MPI_Status status;
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Request_get_status(req_listener_, &is_inactive, &status));
    // This is untestable. Some processes might complete the previous request
    // while others don't, so it's impossible to create a predictable behavior
    // for a test.
    GKO_THROW_IF_INVALID(is_inactive,
                         "Tried to call RowGatherer::apply_async while there "
                         "is already an active communication. Please use the "
                         "overload with a workspace to handle multiple "
                         "connections.");

    auto req = apply_async(b, x, send_workspace_);
    req_listener_ = *req.get();
    return req;
}


template <typename LocalIndexType>
mpi::request RowGatherer<LocalIndexType>::apply_async(
    ptr_param<const LinOp> b, ptr_param<LinOp> x, array<char>& workspace) const
{
    mpi::request req;

    auto exec = this->get_executor();
    auto use_host_buffer =
        mpi::requires_host_buffer(exec, coll_comm_->get_base_communicator());
    auto mpi_exec = use_host_buffer ? exec->get_master() : exec;

    GKO_THROW_IF_INVALID(
        !use_host_buffer || mpi_exec->memory_accessible(x->get_executor()),
        "The receive buffer uses device memory, but MPI support of device "
        "memory is not available or host buffer were explicitly requested. "
        "Please provide a host buffer or enable MPI support for device "
        "memory.");

    // dispatch global vector
    run<Vector,
#if GINKGO_ENABLE_HALF
        half, std::complex<half>,
#endif
        double, float, std::complex<double>, std::complex<float>>(
        make_temporary_clone(exec, b).get(), [&](const auto* b_global) {
            using ValueType =
                typename std::decay_t<decltype(*b_global)>::value_type;
            // dispatch local vector with the same precision as the global
            // vector
            ::gko::precision_dispatch<ValueType>(
                [&](auto* x_local) {
                    auto b_local = b_global->get_local_vector();

                    dim<2> send_size(coll_comm_->get_send_size(),
                                     b_local->get_size()[1]);
                    auto send_size_in_bytes =
                        sizeof(ValueType) * send_size[0] * send_size[1];
                    if (!workspace.get_executor() ||
                        !mpi_exec->memory_accessible(
                            workspace.get_executor()) ||
                        send_size_in_bytes > workspace.get_size()) {
                        workspace = array<char>(
                            mpi_exec,
                            sizeof(ValueType) * send_size[0] * send_size[1]);
                    }
                    auto send_buffer = matrix::Dense<ValueType>::create(
                        mpi_exec, send_size,
                        make_array_view(
                            mpi_exec, send_size[0] * send_size[1],
                            reinterpret_cast<ValueType*>(workspace.get_data())),
                        send_size[1]);
                    b_local->row_gather(&send_idxs_, send_buffer);

                    auto recv_ptr = x_local->get_values();
                    auto send_ptr = send_buffer->get_values();

                    b_local->get_executor()->synchronize();
                    mpi::contiguous_type type(
                        b_local->get_size()[1],
                        mpi::type_impl<ValueType>::get_type());
                    req = coll_comm_->i_all_to_all_v(
                        mpi_exec, send_ptr, type.get(), recv_ptr, type.get());
                },
                x.get());
        });
    return req;
}


template <typename LocalIndexType>
dim<2> RowGatherer<LocalIndexType>::get_size() const
{
    return size_;
}


template <typename LocalIndexType>
std::shared_ptr<const mpi::CollectiveCommunicator>
RowGatherer<LocalIndexType>::get_collective_communicator() const
{
    return coll_comm_;
}


template <typename LocalIndexType>
template <typename GlobalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm,
    const index_map<LocalIndexType, GlobalIndexType>& imap)
    : EnablePolymorphicObject<RowGatherer>(exec),
      DistributedBase(coll_comm->get_base_communicator()),
      size_(dim<2>{imap.get_non_local_size(), imap.get_global_size()}),
      coll_comm_(std::move(coll_comm)),
      send_idxs_(exec),
      send_workspace_(exec),
      req_listener_(MPI_REQUEST_NULL)
{
    // check that the coll_comm_ and imap have the same recv size
    // the same check for the send size is not possible, since the
    // imap doesn't store send indices
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

    send_idxs_.set_executor(mpi_exec);
    send_idxs_.resize_and_reset(coll_comm_->get_send_size());
    inverse_comm
        ->i_all_to_all_v(exec, temp_remote_local_idxs->get_const_flat_data(),
                         send_idxs_.get_data())
        .wait();
    send_idxs_.set_executor(exec);
}


template <typename LocalIndexType>
const LocalIndexType* RowGatherer<LocalIndexType>::get_const_send_idxs() const
{
    return send_idxs_.get_const_data();
}


template <typename LocalIndexType>
size_type RowGatherer<LocalIndexType>::get_num_send_idxs() const
{
    return send_idxs_.get_size();
}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(std::shared_ptr<const Executor> exec,
                                         mpi::communicator comm)
    : EnablePolymorphicObject<RowGatherer>(exec),
      DistributedBase(comm),
      coll_comm_(std::make_shared<DefaultCollComm>(comm)),
      send_idxs_(exec),
      send_workspace_(exec),
      req_listener_(MPI_REQUEST_NULL)
{}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(RowGatherer&& o) noexcept
    : EnablePolymorphicObject<RowGatherer>(o.get_executor()),
      DistributedBase(o.get_communicator()),
      send_idxs_(o.get_executor()),
      send_workspace_(o.get_executor()),
      req_listener_(MPI_REQUEST_NULL)
{
    *this = std::move(o);
}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>& RowGatherer<LocalIndexType>::operator=(
    const RowGatherer& o)
{
    if (this != &o) {
        size_ = o.get_size();
        coll_comm_ = o.coll_comm_;
        send_idxs_ = o.send_idxs_;
    }
    return *this;
}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>& RowGatherer<LocalIndexType>::operator=(
    RowGatherer&& o)
{
    if (this != &o) {
        size_ = std::exchange(o.size_, dim<2>{});
        coll_comm_ = std::exchange(
            o.coll_comm_,
            std::make_shared<DefaultCollComm>(o.get_communicator()));
        send_idxs_ = std::move(o.send_idxs_);
        send_workspace_ = std::move(o.send_workspace_);
        req_listener_ = std::exchange(o.req_listener_, MPI_REQUEST_NULL);
    }
    return *this;
}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(const RowGatherer& o)
    : EnablePolymorphicObject<RowGatherer>(o.get_executor()),
      DistributedBase(o.get_communicator()),
      send_idxs_(o.get_executor())
{
    *this = o;
}


#define GKO_DECLARE_ROW_GATHERER(_itype) class RowGatherer<_itype>

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ROW_GATHERER);

#undef GKO_DECLARE_ROW_GATHERER


#define GKO_DECLARE_ROW_GATHERER_CONSTRUCTOR(_ltype, _gtype)          \
    RowGatherer<_ltype>::RowGatherer(                                 \
        std::shared_ptr<const Executor> exec,                         \
        std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm, \
        const index_map<_ltype, _gtype>& imap)

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_ROW_GATHERER_CONSTRUCTOR);

#undef GKO_DECLARE_ROW_GATHERER_CONSTRUCTOR
}  // namespace distributed
}  // namespace experimental
}  // namespace gko
