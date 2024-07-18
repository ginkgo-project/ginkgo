// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_DENSE_COMMUNICATOR_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_DENSE_COMMUNICATOR_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI

#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/collective_communicator.hpp>
#include <ginkgo/core/distributed/index_map.hpp>


namespace gko {
namespace experimental {
namespace mpi {


/**
 * A collective_communicator that uses a dense communication.
 *
 * The neighborhood communicator is defined by a list of neighbors this
 * rank sends data to and a list of neighbors this rank receives data from.
 * No communication with any ranks that is not in one of those lists will
 * take place.
 */
class DenseCommunicator final : public CollectiveCommunicator {
public:
    using CollectiveCommunicator::i_all_to_all_v;

    /**
     * Default constructor with empty communication pattern
     * @param base  the base communicator
     */
    explicit DenseCommunicator(communicator base);

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
    DenseCommunicator(
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
    DenseCommunicator(communicator base,
                      const std::vector<comm_index_type>& recv_sizes,
                      const std::vector<comm_index_type>& recv_offsets,
                      const std::vector<comm_index_type>& send_sizes,
                      const std::vector<comm_index_type>& send_offsets);

    /**
     * @copydoc CollectiveCommunicator::i_all_to_all_v
     *
     * This implementation uses the neighborhood communication
     * MPI_Ineighbor_alltoallv. See MPI documentation for more details.
     */
    request i_all_to_all_v(std::shared_ptr<const Executor> exec,
                           const void* send_buffer, MPI_Datatype send_type,
                           void* recv_buffer,
                           MPI_Datatype recv_type) const override;

    [[nodiscard]] std::unique_ptr<CollectiveCommunicator> create_with_same_type(
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
    [[nodiscard]] comm_index_type get_recv_size() const override;

    /**
     * @copydoc collective_communicator::get_recv_size
     */
    [[nodiscard]] comm_index_type get_send_size() const override;

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
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_DENSE_COMMUNICATOR_HPP_
