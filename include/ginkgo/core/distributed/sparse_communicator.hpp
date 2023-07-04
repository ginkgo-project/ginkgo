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
#include <ginkgo/core/distributed/overlapping_partition.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include <variant>


namespace gko::experimental::distributed {

/**
 * Creates a distributed graph communicator based on the input sources and
 * destinations.
 *
 * The graph is unweighted and has the same rank ordering as the input
 * communicator.
 */
mpi::communicator create_neighborhood_comm(
    mpi::communicator base, const array<comm_index_type>& sources,
    const array<comm_index_type>& destinations)
{
    auto in_degree = static_cast<comm_index_type>(sources.get_num_elems());
    auto out_degree =
        static_cast<comm_index_type>(destinations.get_num_elems());

    auto sources_host =
        make_temporary_clone(sources.get_executor()->get_master(), &sources);
    auto destinations_host = make_temporary_clone(
        destinations.get_executor()->get_master(), &destinations);

    // adjacent constructor guarantees that querying sources/destinations
    // will result in the array having the same order as defined here
    MPI_Comm new_comm;
    MPI_Dist_graph_create_adjacent(
        base.get(), in_degree, sources_host->get_const_data(), MPI_UNWEIGHTED,
        out_degree, destinations_host->get_const_data(), MPI_UNWEIGHTED,
        MPI_INFO_NULL, false, &new_comm);
    mpi::communicator neighbor_comm{new_comm};  // need to make this owning

    return neighbor_comm;
}

/**
 * Creates a distributed graph communicator based on the input
 * overlapping_partition.
 */
template <typename IndexType>
mpi::communicator create_neighborhood_comm(
    mpi::communicator base, const overlapping_partition<IndexType>* part)
{
    return create_neighborhood_comm(base, part->get_recv_indices().target_ids_,
                                    part->get_send_indices().target_ids_);
}


/**
 * Mode that determines how received values and local values are combined.
 */
enum class transformation {
    set,  // sets the local values to the received values
    add   // adds the received values to the local valules
};


/**
 * Deleter that writes back received values correctly.
 */
template <typename ValueType>
struct interleaved_deleter {
    using vector_type = gko::matrix::Dense<ValueType>;

    void operator()(vector_type* ptr)
    {
        if (original.expired()) {
            GKO_INVALID_STATE(
                "Original communication object has been deleted. Please make "
                "sure that the input vector for the sparse communication has a "
                "longer lifetime than the mpi::request.");
        }
        auto shared_original = original.lock();
        if (mode == transformation::set) {
            // normal scatter
        }
        if (mode == transformation::add) {
            // scatter with add
            auto host_exec = ptr->get_executor()->get_master();
            auto host_ptr = make_temporary_clone(host_exec, ptr);
            auto offset = 0;
            for (auto cur_idxs : idxs->get_sets()) {
                auto full_idxs = cur_idxs.to_global_indices();
                full_idxs.set_executor(host_exec);
                for (int i = 0; i < full_idxs.get_num_elems(); ++i) {
                    auto row = full_idxs.get_const_data()[i];
                    for (int col = 0; col < ptr->get_size()[1]; ++col) {
                        shared_original->at(row, col) +=
                            host_ptr->at(i + offset, col);
                    }
                }
                offset += cur_idxs.get_num_local_indices();
            }
        }
        delete ptr;
    }

    interleaved_deleter(std::shared_ptr<vector_type> original,
                        const overlap_indices<int32>::interleaved* idxs,
                        transformation mode, matrix::Dense<ValueType>* one)
        : original(std::move(original)), idxs(idxs), mode(mode), one(one)
    {}

    std::weak_ptr<vector_type> original;
    const overlap_indices<int32>::interleaved* idxs;
    transformation mode;
    matrix::Dense<ValueType>* one;
};

/**
 * Deleter that writes back received values correctly.
 * Does nothing if `transformation == set`.
 */
template <typename ValueType>
struct blocked_deleter {
    using vector_type = gko::matrix::Dense<ValueType>;

    void operator()(vector_type* ptr)
    {
        if (original.expired()) {
            GKO_INVALID_STATE(
                "Original communication object has been deleted. Please make "
                "sure that the input vector for the sparse communication has a "
                "longer lifetime than the mpi::request.");
        }
        if (mode == transformation::set) {
            // do nothing
        }
        if (mode == transformation::add) {
            // need to put the 1.0 into outside storage for reuse
            // maybe store block-idxs directly and use shortcut?
            auto shared_original = original.lock();
            idxs->get_submatrix(shared_original.get())->add_scaled(one, ptr);
        }
        delete ptr;
    }

    blocked_deleter(std::shared_ptr<vector_type> original,
                    const overlap_indices<int32>::blocked* idxs,
                    transformation mode, matrix::Dense<ValueType>* one)
        : original(std::move(original)), idxs(idxs), mode(mode), one(one)
    {}

    std::weak_ptr<vector_type> original;
    const overlap_indices<int32>::blocked* idxs;
    transformation mode;
    matrix::Dense<ValueType>* one;
};

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
    using partition_type = overlapping_partition<int32>;
    using overlap_idxs_type = overlap_indices<int32>;

    static std::shared_ptr<sparse_communicator> create(
        mpi::communicator comm,
        std::shared_ptr<const overlapping_partition<int32>> part)
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
        std::shared_ptr<matrix::Dense<ValueType>> local_vector,
        transformation mode) const
    {
        return communicate_impl_(default_comm_.get(), part_->get_send_indices(),
                                 send_sizes_, send_offsets_,
                                 part_->get_recv_indices(), recv_sizes_,
                                 recv_offsets_, local_vector, mode);
    }

    /**
     * Uses the inverse communication graph to execute non-blocking neighborhood
     * all-to-all with the local_vector as in- and output.
     *
     * Compared to communicate, the recv and send indices are switched, and all
     * corresponding details.
     *
     * @copydoc communicate
     */
    template <typename ValueType>
    mpi::request communicate_inverse(
        std::shared_ptr<matrix::Dense<ValueType>> local_vector,
        transformation mode) const
    {
        return communicate_impl_(inverse_comm_.get(), part_->get_recv_indices(),
                                 recv_sizes_, recv_offsets_,
                                 part_->get_send_indices(), send_sizes_,
                                 send_offsets_, local_vector, mode);
    }

    std::shared_ptr<const partition_type> get_partition() const
    {
        return part_;
    }

private:
    /**
     * Creates sparse communicator from overlapping_partition
     */
    sparse_communicator(
        mpi::communicator comm,
        std::shared_ptr<const overlapping_partition<int32>> part)
        : default_comm_(create_neighborhood_comm(
              comm, part->get_recv_indices().get_target_ids(),
              part->get_send_indices().get_target_ids())),
          inverse_comm_(create_neighborhood_comm(
              comm, part->get_send_indices().get_target_ids(),
              part->get_recv_indices().get_target_ids())),
          part_(std::move(part)),
          send_sizes_(comm.size()),
          send_offsets_(comm.size() + 1),
          recv_sizes_(comm.size()),
          recv_offsets_(comm.size() + 1)
    {
        auto exec = part_->get_executor();  // should be exec of part_
        auto host_exec = exec->get_master();
        auto fill_size_offsets = [&](std::vector<int>& sizes,
                                     std::vector<int>& offsets,
                                     const auto& overlap) {
            std::visit(
                overloaded{
                    [&](const typename overlap_idxs_type::blocked& idxs) {
                        auto& intervals = idxs.get_intervals();
                        for (int i = 0; i < intervals.size(); ++i) {
                            sizes[i] = intervals[i].length();
                        }
                        std::partial_sum(sizes.begin(), sizes.end(),
                                         offsets.begin() + 1);
                    },
                    [&](const typename overlap_idxs_type::interleaved& idxs) {
                        auto& sets = idxs.get_sets();
                        for (int i = 0; i < sets.size(); ++i) {
                            sizes[i] = sets[i].get_num_local_indices();
                        }
                        std::partial_sum(sizes.begin(), sizes.end(),
                                         offsets.begin() + 1);
                    }},
                overlap.get_idxs());
        };
        fill_size_offsets(recv_sizes_, recv_offsets_,
                          part_->get_recv_indices());
        fill_size_offsets(send_sizes_, send_offsets_,
                          part_->get_send_indices());
    }

    template <typename ValueType>
    mpi::request communicate_impl_(
        MPI_Comm comm, const overlap_idxs_type& send_idxs,
        const std::vector<comm_index_type>& send_sizes,
        const std::vector<comm_index_type>& send_offsets,
        const overlap_idxs_type& recv_idxs,
        const std::vector<comm_index_type>& recv_sizes,
        const std::vector<comm_index_type>& recv_offsets,
        std::shared_ptr<matrix::Dense<ValueType>> local_vector,
        transformation mode) const
    {
        GKO_ASSERT(this->part_->get_size() == local_vector->get_size()[0]);

        using vector_type = matrix::Dense<ValueType>;

        auto exec = local_vector->get_executor();

        // Short-cut to get the contiguous send/recv block
        auto get_overlap_block = [&](const overlap_idxs_type& idxs) {
            GKO_ASSERT(
                std::holds_alternative<typename overlap_idxs_type::blocked>(
                    idxs.get_idxs()));
            return local_vector->create_submatrix(
                {static_cast<size_type>(idxs.get_begin()),
                 static_cast<size_type>(idxs.get_size())},
                {0, local_vector->get_size()[1]});
        };

        std::unique_lock<std::mutex> guard(cache_mutex);

        if (!one_buffer_.get<ValueType>()) {
            one_buffer_.init<ValueType>(exec, {1, 1});
            one_buffer_.get<ValueType>()->fill(one<ValueType>());
        }

        // automatically copies back/adds if necessary
        using recv_handle_t =
            std::unique_ptr<vector_type, std::function<void(vector_type*)>>;
        auto recv_handle = [&] {
            if (mode == transformation::set &&
                std::holds_alternative<typename overlap_idxs_type::blocked>(
                    recv_idxs.get_idxs())) {
                auto block_idxs = std::get<typename overlap_idxs_type::blocked>(
                    recv_idxs.get_idxs());
                return recv_handle_t{
                    block_idxs.get_submatrix(local_vector.get()).release(),
                    blocked_deleter{local_vector, &block_idxs, mode,
                                    one_buffer_.get<ValueType>()}};
            }

            recv_buffer_.init<ValueType>(
                exec, {recv_idxs.get_num_elems(), local_vector->get_size()[1]});

            if (std::holds_alternative<typename overlap_idxs_type::blocked>(
                    recv_idxs.get_idxs())) {
                return recv_handle_t{
                    make_dense_view(recv_buffer_.get<ValueType>()).release(),
                    blocked_deleter{
                        local_vector,
                        &std::get<typename overlap_idxs_type::blocked>(
                            recv_idxs.get_idxs()),
                        mode, one_buffer_.get<ValueType>()}};
            } else {
                return recv_handle_t{
                    make_dense_view(recv_buffer_.get<ValueType>()).release(),
                    interleaved_deleter{
                        local_vector,
                        &std::get<overlap_idxs_type::interleaved>(
                            recv_idxs.get_idxs()),
                        mode, one_buffer_.get<ValueType>()}};
            }
        }();
        auto send_handle = [&] {
            if (std::holds_alternative<typename overlap_idxs_type::blocked>(
                    send_idxs.get_idxs())) {
                return get_overlap_block(send_idxs);
            } else {
                send_buffer_.init<ValueType>(
                    exec,
                    {send_idxs.get_num_elems(), local_vector->get_size()[1]});

                size_type offset = 0;
                auto& sets = std::get<typename overlap_idxs_type::interleaved>(
                                 send_idxs.get_idxs())
                                 .get_sets();
                for (int i = 0; i < sets.size(); ++i) {
                    // need direct support for index_set
                    auto full_idxs = sets[i].to_global_indices();
                    local_vector->row_gather(
                        &full_idxs,
                        send_buffer_.get<ValueType>()->create_submatrix(
                            {offset, offset + full_idxs.get_num_elems()},
                            {0, local_vector->get_size()[1]}));
                    offset += full_idxs.get_num_elems();
                }

                return make_dense_view(send_buffer_.get<ValueType>());
            }
        }();
        auto recv_ptr = recv_handle->get_values();
        auto send_ptr = send_handle->get_values();

        // request deletes recv_handle on successful wait (or at destructor),
        // while keeping this alive
        mpi::request req([h = std::move(recv_handle), g = std::move(guard),
                          sp = shared_from_this()](mpi::request*) mutable {
            h.reset();
            g.release();
            sp.reset();
        });
        MPI_Ineighbor_alltoallv(send_ptr, send_sizes.data(),
                                send_offsets.data(), MPI_DOUBLE, recv_ptr,
                                recv_sizes.data(), recv_offsets.data(),
                                MPI_DOUBLE, comm, req.get());
        return req;
    }

    mpi::communicator default_comm_;
    mpi::communicator inverse_comm_;

    std::shared_ptr<const partition_type> part_;

    std::vector<comm_index_type> send_sizes_;
    std::vector<comm_index_type> send_offsets_;
    std::vector<comm_index_type> recv_sizes_;
    std::vector<comm_index_type> recv_offsets_;

    // @todo can only handle one communication at a time, need to figure out how
    //       to handle multiple
    mutable std::mutex cache_mutex;
    gko::detail::DenseCache2 recv_buffer_;
    gko::detail::DenseCache2 send_buffer_;
    gko::detail::DenseCache2 one_buffer_;
};


}  // namespace gko::experimental::distributed

#endif
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_SPARSE_COMMUNICATOR_HPP_
