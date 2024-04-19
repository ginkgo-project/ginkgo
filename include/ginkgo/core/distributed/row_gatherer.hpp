// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_ROW_GATHERER_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_ROW_GATHERER_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <future>

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
 * imap); auto rg = distributed::RowGatherer<int32>::create(exec, coll_comm,
 * imap);
 *
 * auto b = distributed::Vector<double>::create(...);
 * auto x = matrix::Dense<double>::create(...);
 *
 * auto future = rg->apply_async(b, x);
 * // do some computation that doesn't modify b, or access x
 * future.wait();
 * // x now contains the gathered rows of b
 * ```
 * Using the apply instead of the apply_async will lead to a blocking
 * communication.
 *
 * @note Objects of this class are only available as shared_ptr, since the class
 *       is derived from std::enable_shared_from_this.
 *
 * @tparam LocalIndexType  the index type for the stored indices
 */
template <typename LocalIndexType = int32>
class RowGatherer final
    : public EnableLinOp<RowGatherer<LocalIndexType>>,
      public DistributedBase,
      public std::enable_shared_from_this<RowGatherer<LocalIndexType>> {
    friend class EnablePolymorphicObject<RowGatherer, LinOp>;

public:
    /**
     * Asynchronous version of LinOp::apply.
     *
     * It is asynchronous only wrt. the calling thread. Multiple calls to this
     * function will execute in order, they are not asynchronous with each
     * other.
     *
     * @param b  the input distributed::Vector
     * @param x  the output matrix::Dense with the rows gathered from b
     * @return  a future for this task. The task is guarantueed to completed
     *          after `.wait()` has been called on the future.
     */
    std::future<void> apply_async(ptr_param<const LinOp> b,
                                  ptr_param<LinOp> x) const;

    /**
     * Get the used collective communicator.
     *
     * @return  the used collective communicator
     */
    std::shared_ptr<const mpi::CollectiveCommunicator>
    get_collective_communicator() const;

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
     * @param coll_comm  the collective communicator
     * @param imap  the index map defining which rows to gather
     *
     * @note The coll_comm and imap have to be compatible. The coll_comm must
     *       send and recv exactly as many rows as the imap defines.
     *
     * @return  a shared_ptr to the created distributed::RowGatherer
     */
    template <typename GlobalIndexType = int64,
              typename = std::enable_if_t<sizeof(GlobalIndexType) >=
                                          sizeof(LocalIndexType)>>
    static std::shared_ptr<RowGatherer> create(
        std::shared_ptr<const Executor> exec,
        std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm,
        const index_map<LocalIndexType, GlobalIndexType>& imap)
    {
        return std::shared_ptr<RowGatherer>(
            new RowGatherer(std::move(exec), std::move(coll_comm), imap));
    }

    /*
     * Create method for an empty RowGatherer.
     */
    static std::shared_ptr<RowGatherer> create(
        std::shared_ptr<const Executor> exec, mpi::communicator comm)
    {
        return std::shared_ptr<RowGatherer>(
            new RowGatherer(std::move(exec), std::move(comm)));
    }

    RowGatherer(const RowGatherer& o);

    RowGatherer(RowGatherer&& o) noexcept;

    RowGatherer& operator=(const RowGatherer& o);

    RowGatherer& operator=(RowGatherer&& o);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

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

    std::shared_ptr<const mpi::CollectiveCommunicator> coll_comm_;

    array<LocalIndexType> send_idxs_;

    detail::AnyDenseCache send_buffer_;
    detail::AnyDenseCache recv_buffer_;

    mutable int64 current_id_{0};
    mutable std::atomic<int64> active_id_{0};
};


}  // namespace distributed
}  // namespace experimental
}  // namespace gko

#endif
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_ROW_GATHERER_HPP_
