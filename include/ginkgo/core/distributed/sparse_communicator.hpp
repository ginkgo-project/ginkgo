// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_SPARSE_COMMUNICATOR_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_SPARSE_COMMUNICATOR_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <variant>


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/index_map.hpp>
#include <ginkgo/core/distributed/lin_op.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko::experimental::distributed {
// /**
//  * \brief This provides Gather implementations based on a sparse
//  * topology.
//  */
// struct SparseGather : EnableDistributedLinOp<SparseGather> {
//     template <typename LocalIndexType, typename GlobalIndexType>
//     std::shared_ptr<SparseGather> create(
//         std::shared_ptr<Executor> exec, mpi::communicator comm,
//         std::shared_ptr<const index_map<LocalIndexType, GlobalIndexType>>
//         imap)
//     {
//         return std::shared_ptr<SparseGather>(new
//         SparseGather(std::move(exec), std::move(comm), std::move(imap)));
//     }
//
// protected:
//     void apply_impl(const LinOp* b, LinOp* x) const override;
//     void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
//                     LinOp* x) const override;
//
// private:
//     using imap_i32_i32_type = index_map<int32, int32>;
//     using imap_i32_i64_type = index_map<int32, int64>;
//     using imap_i64_i64_type = index_map<int64, int64>;
//
//     SparseGather(std::shared_ptr<const Executor> exec, mpi::communicator
//     comm);
//
//     template <typename LocalIndexType, typename GlobalIndexType>
//     SparseGather(
//         std::shared_ptr<Executor> exec, mpi::communicator comm,
//         std::shared_ptr<const index_map<LocalIndexType, GlobalIndexType>>
//         imap);
//
//     mpi::communicator default_comm_;
//
//     std::vector<comm_index_type> send_sizes_;
//     std::vector<comm_index_type> send_offsets_;
//     std::vector<comm_index_type> recv_sizes_;
//     std::vector<comm_index_type> recv_offsets_;
//     std::variant<array<int32>, array<int64>> send_idxs_;
// };


/**
 * Simplified MPI communicator that handles only neighborhood all-to-all
 * communication.
 *
 * Besides the default distributed graph communicator, the inverse of that
 * graph is computed, by switching sources and destinations.
 */
class sparse_communicator
    : public std::enable_shared_from_this<sparse_communicator> {
public:
    static std::shared_ptr<sparse_communicator> create(mpi::communicator comm)
    {
        return std::shared_ptr<sparse_communicator>(
            new sparse_communicator(std::move(comm)));
    }

    template <typename LocalIndexType, typename GlobalIndexType>
    static std::shared_ptr<sparse_communicator> create(
        mpi::communicator comm,
        const index_map<LocalIndexType, GlobalIndexType>& imap)
    {
        return std::shared_ptr<sparse_communicator>{
            new sparse_communicator(std::move(comm), std::move(imap))};
    }

    /**
     * Executes non-blocking neighborhood all-to-all with the local_vector as
     * in- and output.
     *
     * This will use external send buffers, if
     * - `interleaved` recv indices are used.
     * Otherwise, it will use the local_vector buffer.
     *
     * This will use external recv buffers, if
     * - `mode != set`
     * - `mode == set` and `interleaved` recv indices are used.
     * Otherwise, it will use the local_vector buffer.
     *
     * The regardless of the used buffer, the result will be written to
     * local_vector. This is managed in the MPI request, so it is necessary to
     * wait on the request, before accessing local_vector again.
     *
     * @warning The request has to be waited on before this is destroyed.
     *
     * @todo could use raw pointer for blocking communication
     *
     * @return mpi::request, local_vector is in a valid state only after
     *         request.wait() has finished
     */
    template <typename ValueType>
    mpi::request communicate(
        const matrix::Dense<ValueType>* local_vector,
        const detail::DenseCache<ValueType>& send_buffer,
        const detail::DenseCache<ValueType>& recv_buffer) const;

    const std::vector<comm_index_type>& get_recv_sizes() const
    {
        return recv_sizes_;
    }

    const std::vector<comm_index_type>& get_recv_offsets() const
    {
        return recv_offsets_;
    }

    const std::vector<comm_index_type>& get_send_sizes() const
    {
        return send_sizes_;
    }

    const std::vector<comm_index_type>& get_send_offsets() const
    {
        return send_offsets_;
    }

    mpi::communicator get_communicator() const { return default_comm_; }

private:
    using imap_i32_i32_type = index_map<int32, int32>;
    using imap_i32_i64_type = index_map<int32, int64>;
    using imap_i64_i64_type = index_map<int64, int64>;


    explicit sparse_communicator(mpi::communicator comm) : default_comm_(comm)
    {}

    /**
     * Creates sparse communicator from overlapping_partition
     */
    template <typename LocalIndexType, typename GlobalIndexType>
    sparse_communicator(mpi::communicator comm,
                        const index_map<LocalIndexType, GlobalIndexType>& imap);

    template <typename ValueType, typename LocalIndexType>
    mpi::request communicate_impl_(
        MPI_Comm comm, const array<LocalIndexType>& send_idxs,
        const matrix::Dense<ValueType>* local_vector,
        const detail::DenseCache<ValueType>& send_buffer,
        const detail::DenseCache<ValueType>& recv_buffer) const;

    mpi::communicator default_comm_;

    std::vector<comm_index_type> send_sizes_;
    std::vector<comm_index_type> send_offsets_;
    std::vector<comm_index_type> recv_sizes_;
    std::vector<comm_index_type> recv_offsets_;
    std::variant<std::monostate, array<int32>, array<int64>> send_idxs_;
};
}  // namespace gko::experimental::distributed

#endif
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_SPARSE_COMMUNICATOR_HPP_
