// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/neighborhood_communicator.hpp"

#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace experimental {
namespace mpi {


/**
 * \brief Computes the inverse envelope (target-ids, sizes) for a given
 *        one-sided communication pattern.
 *
 * \param exec the executor, this will always use the host executor
 * \param comm communicator
 * \param ids target ids of the one-sided operation
 * \param sizes number of elements send to each id
 *
 * \return the inverse envelope consisting of the target-ids and the sizes
 */
std::tuple<std::vector<comm_index_type>, std::vector<comm_index_type>>
communicate_inverse_envelope(std::shared_ptr<const Executor> exec,
                             mpi::communicator comm,
                             const std::vector<comm_index_type>& ids,
                             const std::vector<comm_index_type>& sizes)
{
    auto host_exec = exec->get_master();
    std::vector<comm_index_type> inverse_sizes_full(comm.size());
    mpi::window<comm_index_type> window(host_exec, inverse_sizes_full.data(),
                                        inverse_sizes_full.size(), comm,
                                        sizeof(comm_index_type), MPI_INFO_ENV);
    window.fence();
    for (int i = 0; i < ids.size(); ++i) {
        window.put(host_exec, sizes.data() + i, 1, ids[i], comm.rank(), 1);
    }
    window.fence();

    std::vector<comm_index_type> inverse_sizes;
    std::vector<comm_index_type> inverse_ids;
    for (int i = 0; i < inverse_sizes_full.size(); ++i) {
        if (inverse_sizes_full[i] > 0) {
            inverse_ids.push_back(i);
            inverse_sizes.push_back(inverse_sizes_full[i]);
        }
    }

    return std::make_tuple(std::move(inverse_ids), std::move(inverse_sizes));
}


/**
 * Creates a distributed graph communicator based on the input sources and
 * destinations.
 *
 * The graph is unweighted and has the same rank ordering as the input
 * communicator.
 */
mpi::communicator create_neighborhood_comm(
    mpi::communicator base, const std::vector<comm_index_type>& sources,
    const std::vector<comm_index_type>& destinations)
{
    auto in_degree = static_cast<comm_index_type>(sources.size());
    auto out_degree = static_cast<comm_index_type>(destinations.size());

    // adjacent constructor guarantees that querying sources/destinations
    // will result in the array having the same order as defined here
    MPI_Comm graph_comm;
    MPI_Info info;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Info_dup(MPI_INFO_ENV, &info));
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Dist_graph_create_adjacent(
        base.get(), in_degree, sources.data(),
        in_degree ? MPI_UNWEIGHTED : MPI_WEIGHTS_EMPTY, out_degree,
        destinations.data(), out_degree ? MPI_UNWEIGHTED : MPI_WEIGHTS_EMPTY,
        info, false, &graph_comm));
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Info_free(&info));

    return mpi::communicator::create_owning(graph_comm,
                                            base.force_host_buffer());
}


std::unique_ptr<CollectiveCommunicator>
NeighborhoodCommunicator::create_inverse() const
{
    auto base_comm = this->get_base_communicator();
    distributed::comm_index_type num_sources;
    distributed::comm_index_type num_destinations;
    distributed::comm_index_type weighted;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Dist_graph_neighbors_count(
        comm_.get(), &num_sources, &num_destinations, &weighted));

    std::vector<distributed::comm_index_type> sources(num_sources);
    std::vector<distributed::comm_index_type> destinations(num_destinations);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Dist_graph_neighbors(
        comm_.get(), num_sources, sources.data(), MPI_UNWEIGHTED,
        num_destinations, destinations.data(), MPI_UNWEIGHTED));

    return std::make_unique<NeighborhoodCommunicator>(
        base_comm, destinations, send_sizes_, send_offsets_, sources,
        recv_sizes_, recv_offsets_);
}


comm_index_type NeighborhoodCommunicator::get_recv_size() const
{
    return recv_offsets_.back();
}


comm_index_type NeighborhoodCommunicator::get_send_size() const
{
    return send_offsets_.back();
}


NeighborhoodCommunicator::NeighborhoodCommunicator(
    communicator base, const std::vector<distributed::comm_index_type>& sources,
    const std::vector<comm_index_type>& recv_sizes,
    const std::vector<comm_index_type>& recv_offsets,
    const std::vector<distributed::comm_index_type>& destinations,
    const std::vector<comm_index_type>& send_sizes,
    const std::vector<comm_index_type>& send_offsets)
    : CollectiveCommunicator(base), comm_(MPI_COMM_NULL)
{
    comm_ = create_neighborhood_comm(base, sources, destinations);
    send_sizes_ = send_sizes;
    send_offsets_ = send_offsets;
    recv_sizes_ = recv_sizes;
    recv_offsets_ = recv_offsets;
}


NeighborhoodCommunicator::NeighborhoodCommunicator(communicator base)
    : CollectiveCommunicator(std::move(base)),
      comm_(MPI_COMM_SELF),
      send_sizes_(),
      send_offsets_(1),
      recv_sizes_(),
      recv_offsets_(1)
{
    // ensure that comm_ always has the correct topology
    std::vector<comm_index_type> non_nullptr(1);
    non_nullptr.resize(0);
    comm_ = create_neighborhood_comm(this->get_base_communicator(), non_nullptr,
                                     non_nullptr);
}


request NeighborhoodCommunicator::i_all_to_all_v(
    std::shared_ptr<const Executor> exec, const void* send_buffer,
    MPI_Datatype send_type, void* recv_buffer, MPI_Datatype recv_type) const
{
    auto guard = exec->get_scoped_device_id_guard();
    request req;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Ineighbor_alltoallv(
        send_buffer, send_sizes_.data(), send_offsets_.data(), send_type,
        recv_buffer, recv_sizes_.data(), recv_offsets_.data(), recv_type,
        comm_.get(), req.get()));
    return req;
}


std::unique_ptr<CollectiveCommunicator>
NeighborhoodCommunicator::create_with_same_type(
    communicator base, const distributed::index_map_variant& imap) const
{
    return std::visit(
        [base](const auto& imap) {
            return std::unique_ptr<CollectiveCommunicator>(
                new NeighborhoodCommunicator(base, imap));
        },
        imap);
}


template <typename LocalIndexType, typename GlobalIndexType>
NeighborhoodCommunicator::NeighborhoodCommunicator(
    communicator base,
    const distributed::index_map<LocalIndexType, GlobalIndexType>& imap)
    : CollectiveCommunicator(base),
      comm_(MPI_COMM_SELF),
      recv_sizes_(imap.get_remote_target_ids().get_size()),
      recv_offsets_(recv_sizes_.size() + 1),
      send_offsets_(1)
{
    auto exec = imap.get_executor();
    if (!exec) {
        return;
    }
    auto host_exec = exec->get_master();

    auto recv_target_ids_arr =
        make_temporary_clone(host_exec, &imap.get_remote_target_ids());
    auto remote_idx_offsets_arr = make_temporary_clone(
        host_exec, &imap.get_remote_global_idxs().get_offsets());
    std::vector<comm_index_type> recv_target_ids(
        recv_target_ids_arr->get_size());
    std::copy_n(recv_target_ids_arr->get_const_data(),
                recv_target_ids_arr->get_size(), recv_target_ids.begin());
    for (size_type seg_id = 0;
         seg_id < imap.get_remote_global_idxs().get_segment_count(); ++seg_id) {
        recv_sizes_[seg_id] =
            remote_idx_offsets_arr->get_const_data()[seg_id + 1] -
            remote_idx_offsets_arr->get_const_data()[seg_id];
    }
    auto send_envelope =
        communicate_inverse_envelope(exec, base, recv_target_ids, recv_sizes_);
    const auto& send_target_ids = std::get<0>(send_envelope);
    send_sizes_ = std::move(std::get<1>(send_envelope));

    send_offsets_.resize(send_sizes_.size() + 1);
    std::partial_sum(send_sizes_.begin(), send_sizes_.end(),
                     send_offsets_.begin() + 1);
    std::partial_sum(recv_sizes_.begin(), recv_sizes_.end(),
                     recv_offsets_.begin() + 1);

    comm_ = create_neighborhood_comm(base, recv_target_ids, send_target_ids);
}


#define GKO_DECLARE_NEIGHBORHOOD_CONSTRUCTOR(LocalIndexType, GlobalIndexType) \
    NeighborhoodCommunicator::NeighborhoodCommunicator(                       \
        communicator base,                                                    \
        const distributed::index_map<LocalIndexType, GlobalIndexType>& imap)

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_NEIGHBORHOOD_CONSTRUCTOR);

#undef GKO_DECLARE_NEIGHBORHOOD_CONSTRUCTOR


}  // namespace mpi
}  // namespace experimental
}  // namespace gko
