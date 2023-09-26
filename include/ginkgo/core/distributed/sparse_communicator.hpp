/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_SPARSE_COMMUNICATOR_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_SPARSE_COMMUNICATOR_HPP_

#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/lin_op.hpp>
#include <ginkgo/core/distributed/localized_partition.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include <variant>


namespace gko::experimental::distributed {


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
    static std::shared_ptr<sparse_communicator> create(
        mpi::communicator comm,
        std::shared_ptr<const localized_partition<int32>> part)
    {
        return std::shared_ptr<sparse_communicator>{
            new sparse_communicator(std::move(comm), std::move(part))};
    }

    static std::shared_ptr<sparse_communicator> create(
        mpi::communicator comm,
        std::shared_ptr<const localized_partition<int64>> part)
    {
        return std::shared_ptr<sparse_communicator>{
            new sparse_communicator(std::move(comm), std::move(part))};
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
        std::shared_ptr<matrix::Dense<ValueType>> local_vector) const;

    template <typename IndexType>
    std::shared_ptr<const localized_partition<IndexType>> get_partition() const
    {
        return std::get<std::shared_ptr<const localized_partition<IndexType>>>(
            part_);
    }

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
    using partition_i32_type = localized_partition<int32>;
    using partition_i64_type = localized_partition<int64>;

    /**
     * Creates sparse communicator from overlapping_partition
     */
    template <typename IndexType>
    sparse_communicator(
        mpi::communicator comm,
        std::shared_ptr<const localized_partition<IndexType>> part);

    template <typename ValueType, typename IndexType>
    mpi::request communicate_impl_(
        MPI_Comm comm, const overlap_indices<index_set<IndexType>>& send_idxs,
        const std::vector<comm_index_type>& send_sizes,
        const std::vector<comm_index_type>& send_offsets,
        const overlap_indices<index_block<IndexType>>& recv_idxs,
        const std::vector<comm_index_type>& recv_sizes,
        const std::vector<comm_index_type>& recv_offsets,
        std::shared_ptr<matrix::Dense<ValueType>> local_vector) const;

    mpi::communicator default_comm_;

    std::variant<std::shared_ptr<const partition_i32_type>,
                 std::shared_ptr<const partition_i64_type>>
        part_;

    std::vector<comm_index_type> send_sizes_;
    std::vector<comm_index_type> send_offsets_;
    std::vector<comm_index_type> recv_sizes_;
    std::vector<comm_index_type> recv_offsets_;

    // @todo can only handle one communication at a time, need to figure out how
    //       to handle multiple
    mutable std::mutex cache_mutex;
    gko::detail::AnyDenseCache recv_buffer_;
    gko::detail::AnyDenseCache send_buffer_;
    gko::detail::AnyDenseCache one_buffer_;
};


}  // namespace gko::experimental::distributed

#endif
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_SPARSE_COMMUNICATOR_HPP_
