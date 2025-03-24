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
 * A CollectiveCommunicator that uses a dense communication.
 *
 * The dense communicator uses the MPI_Alltoall function for its communication.
 */
class DenseCommunicator final : public CollectiveCommunicator {
public:
    using CollectiveCommunicator::i_all_to_all_v;

    DenseCommunicator(const DenseCommunicator& other) = default;

    DenseCommunicator(DenseCommunicator&& other);

    DenseCommunicator& operator=(const DenseCommunicator& other) = default;

    DenseCommunicator& operator=(DenseCommunicator&& other);

    /**
     * Default constructor with empty communication pattern
     *
     * @param base  the base communicator
     */
    explicit DenseCommunicator(communicator base);

    /**
     * Create a DenseCommunicator from an index map.
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
    DenseCommunicator(
        communicator base,
        const distributed::index_map<LocalIndexType, GlobalIndexType>& imap);

    [[nodiscard]] creator_fn creator_with_same_type() const override;

    /**
     * Creates the inverse DenseCommunicator by switching sources
     * and destinations.
     *
     * @return  CollectiveCommunicator with the inverse communication pattern
     */
    [[nodiscard]] std::unique_ptr<CollectiveCommunicator> create_inverse()
        const override;

    [[nodiscard]] comm_index_type get_recv_size() const override;

    [[nodiscard]] comm_index_type get_send_size() const override;

    /**
     * Compares two communicators for equality.
     *
     * Equality is defined as having identical or congruent communicators and
     * their communication pattern is equal. No communication is done, i.e.
     * there is no reduction over the local equality check results.
     *
     * @return  true if both communicators are equal.
     */
    friend bool operator==(const DenseCommunicator& a,
                           const DenseCommunicator& b);

    /**
     * Compares two communicators for inequality.
     *
     * @see operator==
     */
    friend bool operator!=(const DenseCommunicator& a,
                           const DenseCommunicator& b);

protected:
    /**
     * @copydoc CollectiveCommunicator::i_all_to_all_v
     *
     * This implementation uses the dense communication
     * MPI_Alltoallv. See MPI documentation for more details.
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
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_DENSE_COMMUNICATOR_HPP_
