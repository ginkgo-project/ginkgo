// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_ROW_SCATTERER_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_ROW_SCATTERER_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/collective_communicator.hpp>
#include <ginkgo/core/distributed/index_map.hpp>
#include <ginkgo/core/distributed/row_gatherer.hpp>


namespace gko {
namespace experimental {
namespace distributed {


/**
 * The distributed::RowScatterer scatters local values to their owning
 * processes and accumulates received contributions into a local target.
 *
 * This is the transpose of the RowGatherer operation:
 * - RowGatherer does:  y_local = R * x_distributed  (gather remote values)
 * - RowScatterer does: x_distributed += R^T * y_local (scatter and accumulate)
 *
 * The operation is split into two phases to allow overlapping computation
 * with communication:
 *
 * 1. `apply_async` packs the send buffer and initiates the MPI
 *    communication. It returns a live `mpi::request`.
 * 2. `wait_and_accumulate` waits for the communication to complete, then
 *    accumulates the received values into the target vector.
 *
 * Example usage:
 * ```c++
 * auto rs = distributed::RowScatterer<int32>::create_from_gatherer(exec, rg);
 *
 * auto req = rs->apply_async(local_vals);
 * // ... overlap other work while communication is in flight ...
 * rs->wait_and_accumulate(req, target);
 * // target now has accumulated the scattered values
 * ```
 *
 * @tparam LocalIndexType  the index type for the stored indices
 */
template <typename LocalIndexType = int32>
class RowScatterer final
    : public EnablePolymorphicObject<RowScatterer<LocalIndexType>>,
      public EnablePolymorphicAssignment<RowScatterer<LocalIndexType>>,
      public DistributedBase {
    friend class EnablePolymorphicObject<RowScatterer, PolymorphicObject>;

public:
    /**
     * Start scattering local values to their owning ranks.
     *
     * Packs the local values into a send buffer and initiates the MPI
     * communication. The returned request can be waited on to ensure the
     * communication completes before calling wait_and_accumulate.
     *
     * @param local_values  the local values to scatter (distributed::Vector)
     *
     * @return  a live mpi::request for the in-flight communication
     */
    [[nodiscard]] mpi::request apply_async(
        ptr_param<const LinOp> local_values) const;

    /**
     * Wait for the scatter communication to complete and accumulate received
     * values into the target: target += received_contributions.
     *
     * Must be called after apply_async. The request must be the one returned
     * by apply_async.
     *
     * @param req  the mpi::request from apply_async (will be waited on)
     * @param distributed_target  the target vector to accumulate into
     */
    void wait_and_accumulate(mpi::request& req,
                             ptr_param<LinOp> distributed_target) const;

    /**
     * Returns the size of the row scatterer.
     */
    dim<2> get_size() const;

    /**
     * Get the used collective communicator.
     */
    std::shared_ptr<const mpi::CollectiveCommunicator>
    get_collective_communicator() const;

    /**
     * Creates a distributed::RowScatterer from a given collective communicator
     * and index map.
     *
     * @tparam GlobalIndexType  the global index type of the index map
     *
     * @param exec  the executor
     * @param coll_comm  the collective communicator
     * @param imap  the index map defining the scatter pattern
     *
     * @return  a unique_ptr to the created distributed::RowScatterer
     */
    template <typename GlobalIndexType = int64,
              typename = std::enable_if_t<sizeof(GlobalIndexType) >=
                                          sizeof(LocalIndexType)>>
    static std::unique_ptr<RowScatterer> create(
        std::shared_ptr<const Executor> exec,
        std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm,
        const index_map<LocalIndexType, GlobalIndexType>& imap)
    {
        return std::unique_ptr<RowScatterer>(
            new RowScatterer(std::move(exec), std::move(coll_comm), imap));
    }

    /**
     * Creates a distributed::RowScatterer from an existing RowGatherer.
     *
     * The scatterer is the transpose of the gatherer: it inverts the
     * communication pattern and accumulates received values at the positions
     * from which the gatherer originally sent.
     *
     * @param exec  the executor
     * @param gatherer  the RowGatherer to create the inverse from
     *
     * @return  a unique_ptr to the created distributed::RowScatterer
     */
    static std::unique_ptr<RowScatterer> create_from_gatherer(
        std::shared_ptr<const Executor> exec,
        ptr_param<const RowGatherer<LocalIndexType>> gatherer);

    /**
     * Create method for an empty RowScatterer.
     */
    static std::unique_ptr<RowScatterer> create(
        std::shared_ptr<const Executor> exec, mpi::communicator comm);

    RowScatterer(const RowScatterer& o);

    RowScatterer(RowScatterer&& o) noexcept;

    RowScatterer& operator=(const RowScatterer& o);

    RowScatterer& operator=(RowScatterer&& o);

private:
    template <typename GlobalIndexType>
    RowScatterer(std::shared_ptr<const Executor> exec,
                 std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm,
                 const index_map<LocalIndexType, GlobalIndexType>& imap);

    RowScatterer(std::shared_ptr<const Executor> exec, mpi::communicator comm);

    RowScatterer(std::shared_ptr<const Executor> exec,
                 std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm,
                 array<LocalIndexType> recv_idxs, dim<2> size);

    dim<2> size_;
    std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm_;
    array<LocalIndexType> recv_idxs_;  // local indices to accumulate into
    mutable gko::detail::GenericDenseCache send_cache_;
    mutable gko::detail::GenericDenseCache recv_cache_;
};


}  // namespace distributed
}  // namespace experimental
}  // namespace gko

#endif
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_ROW_SCATTERER_HPP_
