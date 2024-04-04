// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_NEIGHBORHOOD_COMMUNICATOR_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_NEIGHBORHOOD_COMMUNICATOR_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI

#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/collective_communicator.hpp>
#include <ginkgo/core/distributed/index_map.hpp>


namespace gko {
namespace experimental {
namespace mpi {


/**
 * A collective_communicator that uses a neighborhood topology.
 *
 * The neighborhood communicator is defined by a list of neighbors this
 * rank sends data to and a list of neighbors this rank receives data from.
 * No communication with any ranks that is not in one of those lists will
 * take place.
 */
class NeighborhoodCommunicator final : public CollectiveCommunicator {
public:
    using CollectiveCommunicator::i_all_to_all_v;

    /**
     * Default constructor with empty communication pattern
     * @param base  the base communicator
     */
    explicit NeighborhoodCommunicator(communicator base);

    /**
     * Create a neighborhood_communicator from an index map.
     *
     * The receive neighbors are defined by the remote indices and their
     * owning ranks of the index map. The send neighbors are deduced
     * from that through collective communication.
     *
     * @tparam LocalIndexType  the local index type of the map
     * @tparam GlobalIndexType  the global index type of the map
     * @param base  the base communicator
     * @param imap  the index map that defines the communication pattern
     */
    template <typename LocalIndexType, typename GlobalIndexType>
    NeighborhoodCommunicator(
        communicator base,
        const distributed::index_map<LocalIndexType, GlobalIndexType>& imap);

    /**
     * Create a neighborhood_communicator by explicitly defining the
     * neighborhood lists and sizes/offsets.
     *
     * @param base  the base communicator
     * @param sources  the ranks to receive from
     * @param recv_sizes  the number of elements to recv for each source
     * @param recv_offsets  the offset for each source
     * @param destinations  the ranks to send to
     * @param send_sizes  the number of elements to send for each destination
     * @param send_offsets  the offset for each destination
     */
    NeighborhoodCommunicator(
        communicator base,
        const std::vector<distributed::comm_index_type>& sources,
        const std::vector<comm_index_type>& recv_sizes,
        const std::vector<comm_index_type>& recv_offsets,
        const std::vector<distributed::comm_index_type>& destinations,
        const std::vector<comm_index_type>& send_sizes,
        const std::vector<comm_index_type>& send_offsets);

    /**
     * Communicate data from all ranks to all other ranks using the
     * neighboorhood communication MPI_Ineighbor_alltoallv. See MPI
     * documentation for more details
     *
     * @param exec  The executor, on which the message buffers are located.
     * @param send_buffer  the buffer to send
     * @param send_type  the MPI_Datatype for the send buffer
     * @param recv_buffer  the buffer to gather into
     * @param recv_type  the MPI_Datatype for the recv buffer
     *
     * @return  the request handle for the call
     */
    request i_all_to_all_v(std::shared_ptr<const Executor> exec,
                           const void* send_buffer, MPI_Datatype send_type,
                           void* recv_buffer,
                           MPI_Datatype recv_type) const override;

    std::unique_ptr<CollectiveCommunicator> create_with_same_type(
        communicator base,
        const distributed::index_map_variant& imap) const override;
    /**
     * Creates the inverse neighborhood_communicator by switching sources
     * and destinations.
     *
     * @return  collective_communicator with the inverse communication pattern
     */
    [[nodiscard]] std::unique_ptr<CollectiveCommunicator> create_inverse()
        const override;

    /**
     * @copydoc collective_communicator::get_recv_size
     */
    comm_index_type get_recv_size() const override;

    /**
     * @copydoc collective_communicator::get_recv_size
     */
    comm_index_type get_send_size() const override;

private:
    communicator comm_;

    std::vector<distributed::comm_index_type> send_sizes_;
    std::vector<distributed::comm_index_type> send_offsets_;
    std::vector<distributed::comm_index_type> recv_sizes_;
    std::vector<distributed::comm_index_type> recv_offsets_;
};


}  // namespace mpi
}  // namespace experimental
}  // namespace gko

#endif
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_NEIGHBORHOOD_COMMUNICATOR_HPP_
