// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_COLLECTIVE_COMMUNICATOR_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_COLLECTIVE_COMMUNICATOR_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/index_map_fwd.hpp>


namespace gko {
namespace experimental {
namespace mpi {


/**
 * Interface for a collective communicator.
 *
 * A collective communicator only provides routines for collective
 * communications. At the moment this is restricted to the variable all-to-all.
 */
class CollectiveCommunicator {
public:
    virtual ~CollectiveCommunicator() = default;

    explicit CollectiveCommunicator(communicator base = MPI_COMM_NULL);

    [[nodiscard]] const communicator& get_base_communicator() const;

    /**
     * Non-blocking all-to-all communication.
     *
     * The send_buffer must have allocated at least get_send_size number of
     * elements, and the recv_buffer must have allocated at least get_recv_size
     * number of elements.
     *
     * @tparam SendType  the type of the elements to send
     * @tparam RecvType  the type of the elements to receive
     *
     * @param exec  the executor for the communication
     * @param send_buffer  the send buffer
     * @param recv_buffer  the receive buffer
     *
     * @return  a request handle
     */
    template <typename SendType, typename RecvType>
    [[nodiscard]] request i_all_to_all_v(std::shared_ptr<const Executor> exec,
                                         const SendType* send_buffer,
                                         RecvType* recv_buffer) const;

    /**
     * @copydoc i_all_to_all_v(std::shared_ptr<const Executor>, const SendType*
     *          send_buffer, RecvType* recv_buffer)
     */
    request i_all_to_all_v(std::shared_ptr<const Executor> exec,
                           const void* send_buffer, MPI_Datatype send_type,
                           void* recv_buffer, MPI_Datatype recv_type) const;

    /**
     * Creates a new CollectiveCommunicator with the same dynamic type.
     *
     * @param base  The base communicator
     * @param imap  The index_map that defines the communication pattern
     *
     * @return  a CollectiveCommunicator with the same dynamic type
     */
    [[nodiscard]] virtual std::unique_ptr<CollectiveCommunicator>
    create_with_same_type(communicator base,
                          const distributed::index_map_variant& imap) const = 0;

    /**
     * Creates a CollectiveCommunicator with the inverse communication pattern
     * than this object.
     *
     * @return  a CollectiveCommunicator with the inverse communication
     *          pattern.
     */
    [[nodiscard]] virtual std::unique_ptr<CollectiveCommunicator>
    create_inverse() const = 0;

    /**
     * Get the total number of received elements this communication patterns
     * expects.
     *
     * @return  number of received elements.
     */
    [[nodiscard]] virtual comm_index_type get_recv_size() const = 0;

    /**
     * Get the total number of sent elements this communication patterns
     * expects.
     *
     * @return  number of sent elements.
     */
    [[nodiscard]] virtual comm_index_type get_send_size() const = 0;

protected:
    virtual request i_all_to_all_v_impl(std::shared_ptr<const Executor> exec,
                                        const void* send_buffer,
                                        MPI_Datatype send_type,
                                        void* recv_buffer,
                                        MPI_Datatype recv_type) const = 0;

private:
    communicator base_;
};


template <typename SendType, typename RecvType>
request CollectiveCommunicator::i_all_to_all_v(
    std::shared_ptr<const Executor> exec, const SendType* send_buffer,
    RecvType* recv_buffer) const
{
    return this->i_all_to_all_v(std::move(exec), send_buffer,
                                type_impl<SendType>::get_type(), recv_buffer,
                                type_impl<RecvType>::get_type());
}


}  // namespace mpi
}  // namespace experimental
}  // namespace gko


#endif
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_COLLECTIVE_COMMUNICATOR_HPP_
