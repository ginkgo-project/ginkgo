// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/neighborhood_communicator.hpp"

#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/allocator.hpp"


namespace gko {
namespace experimental {
namespace mpi {


/**
 * @brief Computes the inverse envelope (target-ids, sizes) for a given
 *        one-directional communication pattern.
 *
 * @param exec the executor, this will always use the host executor
 * @param comm communicator
 * @param ids target ids of the one-directional operation
 * @param sizes number of elements send to each id
 *
 * @return the inverse envelope consisting of the target-ids and the sizes
 */
std::tuple<std::vector<comm_index_type>, std::vector<comm_index_type>>
communicate_inverse_envelope(std::shared_ptr<const Executor> exec,
                             communicator comm,
                             const std::vector<comm_index_type>& ids,
                             const std::vector<comm_index_type>& sizes)
{
    auto host_exec = exec->get_master();
    vector<comm_index_type> inverse_sizes_full(comm.size(), {host_exec});
    vector<comm_index_type> send_inverse_sizes(comm.size(), {host_exec});
    for (int i = 0; i < ids.size(); ++i) {
        send_inverse_sizes[ids[i]] = sizes[i];
    }
    comm.all_to_all(host_exec, send_inverse_sizes.data(), 1,
                    inverse_sizes_full.data(), 1);

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
communicator create_neighborhood_comm(
    communicator base, const std::vector<comm_index_type>& sources,
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
        base.get(), in_degree, sources.data(), MPI_UNWEIGHTED, out_degree,
        destinations.data(), MPI_UNWEIGHTED, info, false, &graph_comm));
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Info_free(&info));

    return communicator::create_owning(graph_comm, base.force_host_buffer());
}


std::unique_ptr<CollectiveCommunicator>
NeighborhoodCommunicator::create_inverse() const
{
    auto base_comm = this->get_base_communicator();
    comm_index_type num_sources;
    comm_index_type num_destinations;
    comm_index_type weighted;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Dist_graph_neighbors_count(
        comm_.get(), &num_sources, &num_destinations, &weighted));

    std::vector<comm_index_type> sources(num_sources);
    std::vector<comm_index_type> destinations(num_destinations);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Dist_graph_neighbors(
        comm_.get(), num_sources, sources.data(), MPI_UNWEIGHTED,
        num_destinations, destinations.data(), MPI_UNWEIGHTED));

    auto inv = std::make_unique<NeighborhoodCommunicator>(base_comm);
    inv->comm_ = create_neighborhood_comm(base_comm, destinations, sources);
    inv->send_sizes_ = recv_sizes_;
    inv->send_offsets_ = recv_offsets_;
    inv->recv_sizes_ = send_sizes_;
    inv->recv_offsets_ = send_offsets_;
    return inv;
}


comm_index_type NeighborhoodCommunicator::get_recv_size() const
{
    return recv_offsets_.back();
}


comm_index_type NeighborhoodCommunicator::get_send_size() const
{
    return send_offsets_.back();
}


NeighborhoodCommunicator::NeighborhoodCommunicator(communicator base)
    : CollectiveCommunicator(std::move(base)),
      comm_(MPI_COMM_NULL),
      send_offsets_(1),
      recv_offsets_(1)
{
    if (this->get_base_communicator().get() != MPI_COMM_NULL) {
        // ensure that comm_ always has the correct topology
        std::vector<comm_index_type> non_nullptr(1);
        non_nullptr.resize(0);
        comm_ = create_neighborhood_comm(this->get_base_communicator(),
                                         non_nullptr, non_nullptr);
    }
}


request NeighborhoodCommunicator::i_all_to_all_v_impl(
    std::shared_ptr<const Executor> exec, const void* send_buffer,
    MPI_Datatype send_type, void* recv_buffer, MPI_Datatype recv_type) const
{
#if GINKGO_HAVE_OPENMPI_PRE_4_1_X
    GKO_NOT_IMPLEMENTED;
#else
    auto guard = exec->get_scoped_device_id_guard();
    request req;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Ineighbor_alltoallv(
        send_buffer, send_sizes_.data(), send_offsets_.data(), send_type,
        recv_buffer, recv_sizes_.data(), recv_offsets_.data(), recv_type,
        comm_.get(), req.get()));
    return req;
#endif
}


std::unique_ptr<CollectiveCommunicator>
NeighborhoodCommunicator::create_with_same_type(communicator base,
                                                index_map_ptr imap) const
{
    return std::visit(
        [base](const auto* imap) {
            return std::unique_ptr<CollectiveCommunicator>(
                new NeighborhoodCommunicator(base, *imap));
        },
        imap);
}


NeighborhoodCommunicator::NeighborhoodCommunicator(
    NeighborhoodCommunicator&& other)
    : NeighborhoodCommunicator(other.get_base_communicator())
{
    *this = std::move(other);
}


NeighborhoodCommunicator& NeighborhoodCommunicator::operator=(
    NeighborhoodCommunicator&& other)
{
    if (this != &other) {
        comm_ = std::exchange(other.comm_, MPI_COMM_NULL);
        send_sizes_ =
            std::exchange(other.send_sizes_, std::vector<comm_index_type>{});
        send_offsets_ =
            std::exchange(other.send_offsets_, std::vector<comm_index_type>{0});
        recv_sizes_ =
            std::exchange(other.recv_sizes_, std::vector<comm_index_type>{});
        recv_offsets_ =
            std::exchange(other.recv_offsets_, std::vector<comm_index_type>{0});
    }
    return *this;
}


bool operator==(const NeighborhoodCommunicator& a,
                const NeighborhoodCommunicator& b)
{
    return (a.comm_.is_identical(b.comm_) || a.comm_.is_congruent(b.comm_)) &&
           a.send_sizes_ == b.send_sizes_ && a.recv_sizes_ == b.recv_sizes_ &&
           a.send_offsets_ == b.send_offsets_ &&
           a.recv_offsets_ == b.recv_offsets_;
}


bool operator!=(const NeighborhoodCommunicator& a,
                const NeighborhoodCommunicator& b)
{
    return !(a == b);
}


template <typename LocalIndexType, typename GlobalIndexType>
NeighborhoodCommunicator::NeighborhoodCommunicator(
    communicator base,
    const distributed::index_map<LocalIndexType, GlobalIndexType>& imap)
    : CollectiveCommunicator(base),
      comm_(MPI_COMM_NULL),
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
