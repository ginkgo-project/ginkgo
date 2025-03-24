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
 * A CollectiveCommunicator that uses a neighborhood topology.
 *
 * The neighborhood communicator is defined by a list of neighbors this
 * rank sends data to and a list of neighbors this rank receives data from.
 * No communication with any ranks that is not in one of those lists will
 * take place.
 */
class NeighborhoodCommunicator final : public CollectiveCommunicator {
public:
    using CollectiveCommunicator::i_all_to_all_v;

    NeighborhoodCommunicator(const NeighborhoodCommunicator& other) = default;

    NeighborhoodCommunicator(NeighborhoodCommunicator&& other);

    NeighborhoodCommunicator& operator=(const NeighborhoodCommunicator& other) =
        default;

    NeighborhoodCommunicator& operator=(NeighborhoodCommunicator&& other);

    /**
     * Default constructor with empty communication pattern
     *
     * @param base  the base communicator
     */
    explicit NeighborhoodCommunicator(communicator base);

    /**
     * Create a NeighborhoodCommunicator from an index map.
     *
     * The receiving neighbors are defined by the remote indices and their
     * owning ranks of the index map. The send neighbors are deduced
     * from that through collective communication.
     *
     * @tparam LocalIndexType  the local index type of the map
     * @tparam GlobalIndexType  the global index type of the map
     *
     * @param base  the base communicator
     * @param imap  the index map that defines the communication pattern
     */
    template <typename LocalIndexType, typename GlobalIndexType>
    NeighborhoodCommunicator(
        communicator base,
        const distributed::index_map<LocalIndexType, GlobalIndexType>& imap);

    creator_fn creator_with_same_type() const override;

    /**
     * Creates the inverse NeighborhoodCommunicator by switching sources
     * and destinations.
     *
     * @return  CollectiveCommunicator with the inverse communication pattern
     */
    [[nodiscard]] std::unique_ptr<CollectiveCommunicator> create_inverse()
        const override;

    [[nodiscard]] comm_index_type get_recv_size() const override;

    [[nodiscard]] comm_index_type get_send_size() const override;

    /**
     * Compares two communicators for equality locally.
     *
     * Equality is defined as having identical or congruent communicators and
     * their communication pattern is equal. No communication is done, i.e.
     * there is no reduction over the local equality check results.
     *
     * @return  true if both communicators are equal.
     */
    friend bool operator==(const NeighborhoodCommunicator& a,
                           const NeighborhoodCommunicator& b);

    /**
     * Compares two communicators for inequality.
     *
     * @see operator==
     */
    friend bool operator!=(const NeighborhoodCommunicator& a,
                           const NeighborhoodCommunicator& b);

protected:
    /**
     * @copydoc CollectiveCommunicator::i_all_to_all_v
     *
     * This implementation uses the neighborhood communication
     * MPI_Ineighbor_alltoallv. See MPI documentation for more details.
     */
    request i_all_to_all_v_impl(std::shared_ptr<const Executor> exec,
                                const void* send_buffer, MPI_Datatype send_type,
                                void* recv_buffer,
                                MPI_Datatype recv_type) const override;

private:
    communicator comm_;

    std::vector<comm_index_type> send_sizes_;
    std::vector<comm_index_type> send_offsets_;
    std::vector<comm_index_type> recv_sizes_;
    std::vector<comm_index_type> recv_offsets_;
};


}  // namespace mpi
}  // namespace experimental
}  // namespace gko


#endif
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_NEIGHBORHOOD_COMMUNICATOR_HPP_
