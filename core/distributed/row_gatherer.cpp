// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/row_gatherer.hpp"

#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/distributed/neighborhood_communicator.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/dispatch_helper.hpp"

namespace gko {
namespace experimental {
namespace distributed {


/**
 * \brief
 * \tparam LocalIndexType index type
 * \param comm neighborhood communicator
 * \param remote_local_idxs the remote indices in their local indexing
 * \param recv_sizes the sizes that segregate remote_local_idxs
 * \param send_sizes the number of local indices per rank that are part of
 *                   remote_local_idxs on that ranks
 * \return the local indices that are part of remote_local_idxs on other ranks,
 *         ordered by the rank ordering of the communicator
 */
template <typename LocalIndexType>
array<LocalIndexType> communicate_send_gather_idxs(
    mpi::communicator comm, const array<LocalIndexType>& remote_local_idxs,
    const array<comm_index_type>& recv_ids,
    const std::vector<comm_index_type>& recv_sizes,
    const array<comm_index_type>& send_ids,
    const std::vector<comm_index_type>& send_sizes)
{
    // create temporary inverse sparse communicator
    MPI_Comm sparse_comm;
    MPI_Info info;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Info_create(&info));
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Dist_graph_create_adjacent(
        comm.get(), send_ids.get_size(), send_ids.get_const_data(),
        MPI_UNWEIGHTED, recv_ids.get_size(), recv_ids.get_const_data(),
        MPI_UNWEIGHTED, info, false, &sparse_comm));
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Info_free(&info));

    std::vector<comm_index_type> recv_offsets(recv_sizes.size() + 1);
    std::vector<comm_index_type> send_offsets(send_sizes.size() + 1);
    std::partial_sum(recv_sizes.data(), recv_sizes.data() + recv_sizes.size(),
                     recv_offsets.begin() + 1);
    std::partial_sum(send_sizes.data(), send_sizes.data() + send_sizes.size(),
                     send_offsets.begin() + 1);

    array<LocalIndexType> send_gather_idxs(remote_local_idxs.get_executor(),
                                           send_offsets.back());

    GKO_ASSERT_NO_MPI_ERRORS(MPI_Neighbor_alltoallv(
        remote_local_idxs.get_const_data(), recv_sizes.data(),
        recv_offsets.data(), mpi::type_impl<LocalIndexType>::get_type(),
        send_gather_idxs.get_data(), send_sizes.data(), send_offsets.data(),
        mpi::type_impl<LocalIndexType>::get_type(), sparse_comm));
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_free(&sparse_comm));

    return send_gather_idxs;
}


template <typename LocalIndexType>
void RowGatherer<LocalIndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    apply_async(b, x).wait();
}


template <typename LocalIndexType>
void RowGatherer<LocalIndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                             const LinOp* beta, LinOp* x) const
    GKO_NOT_IMPLEMENTED;


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

    // dispatch global vector
    run<Vector, double, float, std::complex<double>, std::complex<float>>(
        b.get(), [&](const auto* b_global) {
            using ValueType =
                typename std::decay_t<decltype(*b_global)>::value_type;
            // dispatch local vector with the same precision as the global
            // vector
            ::gko::precision_dispatch<ValueType>(
                [&](auto* x_local) {
                    auto exec = this->get_executor();

                    auto use_host_buffer = mpi::requires_host_buffer(
                        exec, coll_comm_->get_base_communicator());
                    auto mpi_exec = use_host_buffer ? exec->get_master() : exec;

                    GKO_THROW_IF_INVALID(
                        !use_host_buffer || mpi_exec->memory_accessible(
                                                x_local->get_executor()),
                        "The receive buffer uses device memory, but MPI "
                        "support of device memory is not available. Please "
                        "provide a host buffer or enable MPI support for "
                        "device memory.");

                    auto b_local = b_global->get_local_vector();

                    dim<2> send_size(coll_comm_->get_send_size(),
                                     b_local->get_size()[1]);
                    workspace.set_executor(mpi_exec);
                    workspace.resize_and_reset(sizeof(ValueType) *
                                               send_size[0] * send_size[1]);
                    auto send_buffer = matrix::Dense<ValueType>::create(
                        mpi_exec, send_size,
                        make_array_view(
                            mpi_exec, send_size[0] * send_size[1],
                            reinterpret_cast<ValueType*>(workspace.get_data())),
                        send_size[1]);
                    b_local->row_gather(&send_idxs_, send_buffer);

                    auto recv_ptr = x_local->get_values();
                    auto send_ptr = send_buffer->get_values();

                    mpi_exec->synchronize();
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
std::shared_ptr<const mpi::collective_communicator>
RowGatherer<LocalIndexType>::get_collective_communicator() const
{
    return coll_comm_;
}


template <typename LocalIndexType>
template <typename GlobalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const mpi::collective_communicator> coll_comm,
    const index_map<LocalIndexType, GlobalIndexType>& imap)
    : EnableDistributedLinOp<RowGatherer<LocalIndexType>>(
          exec, dim<2>{imap.get_non_local_size(), imap.get_global_size()}),
      DistributedBase(coll_comm->get_base_communicator()),
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

    send_idxs_.resize_and_reset(coll_comm_->get_send_size());
    inverse_comm
        ->i_all_to_all_v(exec,
                         imap.get_remote_local_idxs().get_const_flat_data(),
                         send_idxs_.get_data())
        .wait();
}


template <typename LocalIndexType>
const LocalIndexType* RowGatherer<LocalIndexType>::get_const_row_idxs() const
{
    return send_idxs_.get_const_data();
}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(std::shared_ptr<const Executor> exec,
                                         mpi::communicator comm)
    : EnableDistributedLinOp<RowGatherer<LocalIndexType>>(exec),
      DistributedBase(comm),
      coll_comm_(std::make_shared<mpi::neighborhood_communicator>(comm)),
      send_idxs_(exec),
      send_workspace_(exec),
      req_listener_(MPI_REQUEST_NULL)
{}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(RowGatherer&& o) noexcept
    : EnableDistributedLinOp<RowGatherer<LocalIndexType>>(o.get_executor()),
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
        this->set_size(o.get_size());
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
        this->set_size(o.get_size());
        o.set_size({});
        coll_comm_ = std::exchange(
            o.coll_comm_, std::make_shared<mpi::neighborhood_communicator>(
                              o.get_communicator()));
        send_idxs_ = std::move(o.send_idxs_);
        send_workspace_ = std::move(o.send_workspace_);
        req_listener_ = std::exchange(o.req_listener_, MPI_REQUEST_NULL);
    }
    return *this;
}


template <typename LocalIndexType>
RowGatherer<LocalIndexType>::RowGatherer(const RowGatherer& o)
    : EnableDistributedLinOp<RowGatherer<LocalIndexType>>(o.get_executor()),
      DistributedBase(o.get_communicator()),
      send_idxs_(o.get_executor())
{
    *this = o;
}


#define GKO_DECLARE_ROW_GATHERER(_itype) class RowGatherer<_itype>

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ROW_GATHERER);

#undef GKO_DECLARE_ROW_GATHERER


#define GKO_DECLARE_ROW_GATHERER_CONSTRUCTOR(_ltype, _gtype)           \
    RowGatherer<_ltype>::RowGatherer(                                  \
        std::shared_ptr<const Executor> exec,                          \
        std::shared_ptr<const mpi::collective_communicator> coll_comm, \
        const index_map<_ltype, _gtype>& imap)

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_ROW_GATHERER_CONSTRUCTOR);

#undef GKO_DECLARE_ROW_GATHERER_CONSTRUCTOR


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
