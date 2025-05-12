// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_ROW_GATHERER_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_ROW_GATHERER_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/collective_communicator.hpp>
#include <ginkgo/core/distributed/index_map.hpp>


namespace gko {
namespace experimental {
namespace distributed {


/**
 * The distributed::RowGatherer gathers the rows of distributed::Vector that
 * are located on other processes.
 *
 * Example usage:
 * ```c++
 * auto coll_comm = std::make_shared<mpi::neighborhood_communicator>(comm,
 *                                                                   imap);
 * auto rg = distributed::RowGatherer<int32>::create(exec, coll_comm, imap);
 *
 * auto b = distributed::Vector<double>::create(...);
 * auto x = matrix::Dense<double>::create(...);
 *
 * auto req = rg->apply_async(b, x);
 * // do some computation that doesn't modify b, or access x
 * req.wait();
 * // x now contains the gathered rows of b
 * ```
 * Using apply instead of apply_async will lead to a blocking communication.
 *
 * @note The output vector for the apply_async functions *must* use an executor
 *       that is compatible with the MPI implementation. In particular, if the
 *       MPI implementation is not GPU aware, then the output vector *must* use
 *       a CPU executor. Otherwise, an exception will be thrown.
 *
 * @tparam LocalIndexType  the index type for the stored indices
 */
template <typename LocalIndexType = int32>
class RowGatherer final
    : public EnablePolymorphicObject<RowGatherer<LocalIndexType>>,
      public EnablePolymorphicAssignment<RowGatherer<LocalIndexType>>,
      public DistributedBase {
    friend class EnablePolymorphicObject<RowGatherer, PolymorphicObject>;

public:
    /**
     * Asynchronous version of LinOp::apply.
     *
     * @warning Only one mpi::request can be active at any given time. This
     *          function will throw if another request is already active.
     *
     * @param b  the input distributed::Vector.
     * @param x  the output matrix::Dense with the rows gathered from b. Its
     *           executor has to be compatible with the MPI implementation, see
     *           the class documentation.
     *
     * @return  a mpi::request for this task. The task is guaranteed to
     *          be completed only after `.wait()` has been called on it.
     */
    mpi::request apply_async(ptr_param<const LinOp> b,
                             ptr_param<LinOp> x) const;

    /**
     * Asynchronous version of LinOp::apply.
     *
     * @warning Calling this multiple times with the same workspace and without
     *          waiting on each previous request will lead to incorrect
     *          data transfers.
     *
     * @param b  the input distributed::Vector.
     * @param x  the output matrix::Dense with the rows gathered from b. Its
     *           executor has to be compatible with the MPI implementation, see
     *           the class documentation.
     * @param workspace  a workspace to store temporary data for the operation.
     *                   This might not be modified before the request is
     *                   waited on.
     *
     * @return  a mpi::request for this task. The task is guaranteed to
     *          be completed only after `.wait()` has been called on it.
     */
    mpi::request apply_async(ptr_param<const LinOp> b, ptr_param<LinOp> x,
                             array<char>& workspace) const;

    /**
     * Returns the size of the row gatherer.
     */
    dim<2> get_size() const;

    /**
     * Get the used collective communicator.
     */
    std::shared_ptr<const mpi::CollectiveCommunicator>
    get_collective_communicator() const;

    /**
     * Read access to the (local) rows indices
     */
    const LocalIndexType* get_const_send_idxs() const;

    /**
     * Returns the number of (local) row indices.
     */
    size_type get_num_send_idxs() const;

    /**
     * Creates a distributed::RowGatherer from a given collective communicator
     * and index map.
     *
     * @TODO: using a segmented array instead of the imap would probably be
     *        more general
     *
     * @tparam GlobalIndexType  the global index type of the index map
     *
     * @param exec  the executor
     * @param coll_comm_creator  the collective communicator creator_fn
     * @param imap  the index map defining which rows to gather
     *
     * @return  a shared_ptr to the created distributed::RowGatherer
     */
    template <typename GlobalIndexType = int64,
              typename = std::enable_if_t<sizeof(GlobalIndexType) >=
                                          sizeof(LocalIndexType)>>
    static std::unique_ptr<RowGatherer> create(
        std::shared_ptr<const Executor> exec,
        mpi::CollectiveCommunicator::creator_fn coll_comm_creator,
        mpi::communicator base_comm,
        const index_map<LocalIndexType, GlobalIndexType>& imap)
    {
        return std::unique_ptr<RowGatherer>(new RowGatherer(
            std::move(exec), coll_comm_creator(base_comm, &imap), imap));
    }

    /*
     * Create method for an empty RowGatherer.
     */
    static std::unique_ptr<RowGatherer> create(
        std::shared_ptr<const Executor> exec, mpi::communicator comm)
    {
        return std::unique_ptr<RowGatherer>(
            new RowGatherer(std::move(exec), std::move(comm)));
    }

    RowGatherer(const RowGatherer& o);

    RowGatherer(RowGatherer&& o) noexcept;

    RowGatherer& operator=(const RowGatherer& o);

    RowGatherer& operator=(RowGatherer&& o);

private:
    /**
     * @copydoc RowGatherer::create(std::shared_ptr<const
     *          Executor>, std::shared_ptr<const mpi::collective_communicator>,
     *          const index_map<LocalIndexType, GlobalIndexType>&)
     */
    template <typename GlobalIndexType>
    RowGatherer(std::shared_ptr<const Executor> exec,
                std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm,
                const index_map<LocalIndexType, GlobalIndexType>& imap);

    /**
     * @copydoc RowGatherer::create(std::shared_ptr<const
     *          Executor>, mpi::communicator)
     */
    RowGatherer(std::shared_ptr<const Executor> exec, mpi::communicator comm);

    dim<2> size_;
    std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm_;
    array<LocalIndexType> send_idxs_;
    mutable array<char> send_workspace_;
    // This object might not hold an actual MPI request, so we can't use the
    // always owning mpi::request. It's destructor would otherwise make the
    // program crash.
    mutable MPI_Request req_listener_{MPI_REQUEST_NULL};
};
}  // namespace distributed
}  // namespace experimental
}  // namespace gko

#endif
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_ROW_GATHERER_HPP_
