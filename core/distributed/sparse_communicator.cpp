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

#include <ginkgo/core/distributed/sparse_communicator.hpp>

namespace gko {
namespace experimental {
namespace distributed {


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


template <typename ValueType>
mpi::request sparse_communicator::communicate(
    std::shared_ptr<matrix::Dense<ValueType>> local_vector,
    gko::experimental::distributed::transformation mode) const
{
    return std::visit(
        [&, this](const auto& part) {
            return communicate_impl_(
                inverse_comm_.get(), part->get_send_indices(), recv_sizes_,
                recv_offsets_, part->get_recv_indices(), send_sizes_,
                send_offsets_, local_vector, mode);
        },
        part_);
}


template <typename ValueType>
mpi::request sparse_communicator::communicate_inverse(
    std::shared_ptr<matrix::Dense<ValueType>> local_vector,
    gko::experimental::distributed::transformation mode) const
{
    return std::visit(
        [&, this](const auto& part) {
            return communicate_impl_(
                inverse_comm_.get(), part->get_recv_indices(), recv_sizes_,
                recv_offsets_, part->get_send_indices(), send_sizes_,
                send_offsets_, local_vector, mode);
        },
        part_);
}


template <typename IndexType>
sparse_communicator::sparse_communicator(
    mpi::communicator comm,
    std::shared_ptr<const overlapping_partition<IndexType>> part)
    : default_comm_(create_neighborhood_comm(
          comm, part->get_recv_indices().get_target_ids(),
          part->get_send_indices().get_target_ids())),
      inverse_comm_(create_neighborhood_comm(
          comm, part->get_send_indices().get_target_ids(),
          part->get_recv_indices().get_target_ids())),
      send_sizes_(comm.size()),
      send_offsets_(comm.size() + 1),
      recv_sizes_(comm.size()),
      recv_offsets_(comm.size() + 1)
{
    auto exec = part->get_executor();
    auto host_exec = exec->get_master();
    auto fill_size_offsets = [&](std::vector<int>& sizes,
                                 std::vector<int>& offsets,
                                 const auto& overlap) {
        using overlap_t = std::decay_t<decltype(overlap)>;
        std::visit(overloaded{[&](const typename overlap_t::blocked& idxs) {
                                  auto& intervals = idxs.get_intervals();
                                  for (int i = 0; i < intervals.size(); ++i) {
                                      sizes[i] = intervals[i].length();
                                  }
                                  std::partial_sum(sizes.begin(), sizes.end(),
                                                   offsets.begin() + 1);
                              },
                              [&](const typename overlap_t::interleaved& idxs) {
                                  auto& sets = idxs.get_sets();
                                  for (int i = 0; i < sets.size(); ++i) {
                                      sizes[i] = sets[i].get_num_elems();
                                  }
                                  std::partial_sum(sizes.begin(), sizes.end(),
                                                   offsets.begin() + 1);
                              }},
                   overlap.get_idxs());
    };
    fill_size_offsets(recv_sizes_, recv_offsets_, part->get_recv_indices());
    fill_size_offsets(send_sizes_, send_offsets_, part->get_send_indices());
    part_ = std::move(part);
}


template <typename ValueType, typename IndexType>
mpi::request sparse_communicator::communicate_impl_(
    MPI_Comm comm, const overlap_indices<IndexType>& send_idxs,
    const std::vector<comm_index_type>& send_sizes,
    const std::vector<comm_index_type>& send_offsets,
    const overlap_indices<IndexType>& recv_idxs,
    const std::vector<comm_index_type>& recv_sizes,
    const std::vector<comm_index_type>& recv_offsets,
    std::shared_ptr<matrix::Dense<ValueType>> local_vector,
    transformation mode) const
{
    using overlap_idxs_type = overlap_indices<IndexType>;
    GKO_ASSERT(std::visit([](const auto& part) { return part->get_size(); },
                          part_) == local_vector->get_size()[0]);

    using vector_type = matrix::Dense<ValueType>;

    auto exec = local_vector->get_executor();

    // Short-cut to get the contiguous send/recv block
    auto get_overlap_block = [&](const overlap_idxs_type& idxs) {
        GKO_ASSERT(std::holds_alternative<typename overlap_idxs_type::blocked>(
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
                blocked_deleter<ValueType, IndexType>{
                    local_vector, &block_idxs, mode,
                    one_buffer_.get<ValueType>()}};
        }

        recv_buffer_.init<ValueType>(
            exec, {recv_idxs.get_num_elems(), local_vector->get_size()[1]});

        if (std::holds_alternative<typename overlap_idxs_type::blocked>(
                recv_idxs.get_idxs())) {
            return recv_handle_t{
                make_dense_view(recv_buffer_.get<ValueType>()).release(),
                blocked_deleter<ValueType, IndexType>{
                    local_vector,
                    &std::get<typename overlap_idxs_type::blocked>(
                        recv_idxs.get_idxs()),
                    mode, one_buffer_.get<ValueType>()}};
        } else {
            return recv_handle_t{
                make_dense_view(recv_buffer_.get<ValueType>()).release(),
                interleaved_deleter<ValueType, IndexType>{
                    local_vector,
                    &std::get<typename overlap_idxs_type::interleaved>(
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
                exec, {send_idxs.get_num_elems(), local_vector->get_size()[1]});

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
    MPI_Ineighbor_alltoallv(send_ptr, send_sizes.data(), send_offsets.data(),
                            MPI_DOUBLE, recv_ptr, recv_sizes.data(),
                            recv_offsets.data(), MPI_DOUBLE, comm, req.get());
    return req;
}

#define GKO_DECLARE_COMMUNICATE_IMPL(ValueType, IndexType)                     \
    mpi::request sparse_communicator::communicate_impl_<ValueType, IndexType>( \
        MPI_Comm comm, const overlap_indices<IndexType>& send_idxs,            \
        const std::vector<comm_index_type>& send_sizes,                        \
        const std::vector<comm_index_type>& send_offsets,                      \
        const overlap_indices<IndexType>& recv_idxs,                           \
        const std::vector<comm_index_type>& recv_sizes,                        \
        const std::vector<comm_index_type>& recv_offsets,                      \
        std::shared_ptr<matrix::Dense<ValueType>> local_vector,                \
        transformation mode) const
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COMMUNICATE_IMPL);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko