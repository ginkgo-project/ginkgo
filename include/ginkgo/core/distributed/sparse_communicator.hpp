// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_SPARSE_COMMUNICATOR_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_SPARSE_COMMUNICATOR_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI && GINKGO_HAVE_CXX17


#include <variant>


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/index_map.hpp>


namespace gko {
namespace experimental {
namespace distributed {


/**
 * Simplified MPI communicator that handles only neighborhood all-to-all
 * communication.
 */
class sparse_communicator {
public:
    using hook_function = std::function<void(LinOp*)>;

    static hook_function default_hook()
    {
        return [](LinOp*) {};
    }

    sparse_communicator()
        : default_comm_(MPI_COMM_SELF), send_offsets_{0}, recv_offsets_{0}
    {}

    /**
     * Creates sparse communicator from an index map
     */
    template <typename LocalIndexType, typename GlobalIndexType>
    sparse_communicator(mpi::communicator comm,
                        const index_map<LocalIndexType, GlobalIndexType>& imap,
                        hook_function pre_hook = default_hook(),
                        hook_function post_hook = default_hook());

    /**
     * Executes non-blocking neighborhood all-to-all.
     *
     * @return mpi::request, the recv_buffer is in a valid state only after
     *         request.wait() has finished
     */
    template <typename ValueType>
    mpi::request communicate(
        const matrix::Dense<ValueType>* local_vector,
        const detail::DenseCache<ValueType>& send_buffer,
        const detail::DenseCache<ValueType>& recv_buffer) const;

    sparse_communicator(const sparse_communicator&) = default;

    sparse_communicator(sparse_communicator&&) = default;

    sparse_communicator& operator=(const sparse_communicator&) = default;

    sparse_communicator& operator=(sparse_communicator&&) = default;

private:
    template <typename ValueType, typename LocalIndexType>
    mpi::request communicate_impl_(
        mpi::communicator comm, const array<LocalIndexType>& send_idxs,
        const matrix::Dense<ValueType>* local_vector,
        const detail::DenseCache<ValueType>& send_buffer,
        const detail::DenseCache<ValueType>& recv_buffer) const;

    mpi::communicator default_comm_;

    std::vector<comm_index_type> send_sizes_;
    std::vector<comm_index_type> send_offsets_;
    std::vector<comm_index_type> recv_sizes_;
    std::vector<comm_index_type> recv_offsets_;
    std::variant<std::monostate, array<int32>, array<int64>> send_idxs_;

    hook_function pre_hook_;
    hook_function post_hook_;
};


}  // namespace distributed
}  // namespace experimental
}  // namespace gko

#endif
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_SPARSE_COMMUNICATOR_HPP_
