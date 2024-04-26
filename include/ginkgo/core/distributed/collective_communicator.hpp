// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_COLLECTIVE_COMMUNICATOR_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_COLLECTIVE_COMMUNICATOR_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI

#include <ginkgo/core/base/mpi.hpp>


namespace gko {
namespace experimental {
namespace mpi {

/**
 * Interface for an collective communicator.
 *
 * A collective communicator only provides routines for collective
 * communications. At the moment this is restricted to the variable all-to-all.
 */
class collective_communicator {
public:
    virtual ~collective_communicator() = default;

    /**
     * Creates a collective_communicator.
     *
     * @param base  the underlying communicator, without any modifications, such
     *              as topology information or similar
     */
    explicit collective_communicator(communicator base) : base_(std::move(base))
    {}

    /**
     * Get the unmodified base communicator.
     *
     * @return  the unmodified base communicator.
     */
    const communicator& get_base_communicator() const { return base_; }

    /**
     * Non-blocking all-to-all communication.
     *
     * The send_buffer must have size get_send_size, and the recv_buffer
     * must have size get_recv_size.
     *
     * @tparam SendType  the type of the elements to send
     * @tparam RecvType  the type of the elements to receive
     * @param exec  the executor for the communication
     * @param send_buffer  the send buffer
     * @param recv_buffer  the receive buffer
     * @return  a request handle
     */
    template <typename SendType, typename RecvType>
    request i_all_to_all_v(std::shared_ptr<const Executor> exec,
                           const SendType* send_buffer,
                           RecvType* recv_buffer) const
    {
        return this->i_all_to_all_v(
            std::move(exec), send_buffer, type_impl<SendType>::get_type(),
            recv_buffer, type_impl<RecvType>::get_type());
    }

    /**
     * @copydoc i_all_to_all_v(std::shared_ptr<const Executor>, const SendType*
     *          send_buffer, RecvType* recv_buffer)
     */
    virtual request i_all_to_all_v(std::shared_ptr<const Executor> exec,
                                   const void* send_buffer,
                                   MPI_Datatype send_type, void* recv_buffer,
                                   MPI_Datatype recv_type) const = 0;


    /**
     * Creates a collective_communicator with the inverse communication pattern
     * than this object.
     *
     * @return  a collective_communicator with the inverse communication
     * pattern.
     */
    virtual std::unique_ptr<collective_communicator> create_inverse() const = 0;

    /**
     * Get the total number of received elements this communication patterns
     * expects.
     *
     * This is the sum of get_recv_sizes.
     *
     * @return  number of received elements.
     */
    virtual comm_index_type get_recv_size() const = 0;

    /**
     * Get the total number of sent elements this communication patterns
     * expects.
     *
     * This is the sum of get_send_sizes.
     *
     * @return  number of sent elements.
     */
    virtual comm_index_type get_send_size() const = 0;

    /**
     * Get the number of received elements per process this communication
     * pattern extpects.
     *
     * The numbering does not necessarily correlate with the natural numbering
     * of the processes. It should not be expected that the i-th element
     * denotes the number of elements received from the process with rank i.
     * Instead the i-th element denotes the number of elements received from
     * the i-th rank this process communicates with.
     *
     * @return  the nubber of received elements per process
     */
    virtual std::vector<comm_index_type> get_recv_sizes() const = 0;

    /**
     * Get the number of sent elements per process this communication
     * pattern extpects.
     *
     * The numbering does not necessarily correlate with the natural numbering
     * of the processes. It should not be expected that the i-th element
     * denotes the number of elements sent to the process with rank i.
     * Instead the i-th element denotes the number of elements sent to the i-th
     * rank this process communicates with.
     *
     * @return  the nubber of received elements per process
     */
    virtual std::vector<comm_index_type> get_send_sizes() const = 0;

private:
    communicator base_;
};


}  // namespace mpi
}  // namespace experimental
}  // namespace gko

#endif
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_COLLECTIVE_COMMUNICATOR_HPP_
