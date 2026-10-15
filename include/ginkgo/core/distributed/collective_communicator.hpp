// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_COLLECTIVE_COMMUNICATOR_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_COLLECTIVE_COMMUNICATOR_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI

#include <variant>

#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/index_map.hpp>


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
    /**
     * All allowed index_map types (as const *)
     */
    using index_map_ptr =
        std::variant<const distributed::index_map<int32, int32>*,
                     const distributed::index_map<int32, int64>*,
                     const distributed::index_map<int64, int64>*>;

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
    create_with_same_type(communicator base, index_map_ptr imap) const = 0;

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
     * Create new communicator that sends and receives a different number of
     * elements to/from each rank in its current pattern.
     *
     * The number of elements to send is deduces from the send_factors. If this
     * communicator is used to send 2 elements to rank 0, and 3 elements to
     * rank 1, then send_factors needs to have size 5, which is equal to
     * get_send_size(). An send_factors = { 4, 2, 1, 3, 2} means that
     * instead of the first element to send to rank 0, four elements are sent,
     * instead of the second element to send to rank 0, two elements are
     * sent, and so on. A visual representation of the transformation is:
     * ```
     * [ 1 1 | 1 1 1 ] -> [ 4 2 | 1 3 2] -> [ 1 1 1 1 1 1 | 1 1 1 1 1 1 ]
     * ```
     * The communication pattern will not change. Even if send_factors is zero
     * for all elements of one rank, the rank will stay in the new pattern.
     *
     * @note This is a collective call.
     *
     * @param exec  the executor to use for the new communicator
     * @param send_factors  the new number of elements to send.
     *
     * @return The new communicator and the recv_factors implied by the
     *         send_factors
     */
    [[nodiscard]] virtual std::pair<std::unique_ptr<CollectiveCommunicator>,
                                    std::vector<comm_index_type>>
    resize(std::shared_ptr<const Executor> exec,
           const std::vector<comm_index_type>& send_factors) const = 0;

    /**
     * Get the number of elements received by this process within this
     * communication pattern.
     *
     * @return  number of received elements.
     */
    [[nodiscard]] virtual comm_index_type get_recv_size() const = 0;

    /**
     * Get the number of elements sent by this process within this communication
     * pattern.
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


namespace detail {


/**
 * Creates a collective communicator with the default type.
 *
 * @note: Currently the default type is DenseCommunicator, but this is subject
 *        to change.
 *
 * @return  A collective communicator.
 */
std::shared_ptr<CollectiveCommunicator> create_default_collective_communicator(
    communicator base);


}  // namespace detail
}  // namespace mpi
}  // namespace experimental
}  // namespace gko


#endif
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_COLLECTIVE_COMMUNICATOR_HPP_
