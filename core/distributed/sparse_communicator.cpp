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
 * Get all rows of the input vector that are local to this process.
 */
template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Dense<ValueType>> get_submatrix(
    gko::matrix::Dense<ValueType>* vector,
    const overlap_indices<index_block<IndexType>>* idxs)
{
    return vector->create_submatrix(
        span{static_cast<size_type>(idxs->get_begin()),
             static_cast<size_type>(idxs->get_end())},
        span{0, vector->get_size()[1]});
}


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
    mpi::communicator base, const localized_partition<IndexType>* part)
{
    return create_neighborhood_comm(base, part->get_recv_indices().target_ids_,
                                    part->get_send_indices().target_ids_);
}


template <typename ValueType>
mpi::request sparse_communicator::communicate(
    const matrix::Dense<ValueType>* local_vector,
    const detail::DenseCache<ValueType>& send_buffer,
    const detail::DenseCache<ValueType>& recv_buffer) const
{
    return std::visit(
        [&, this](const auto& part) {
            if constexpr (std::is_same_v<std::decay_t<decltype(part)>,
                                         std::monostate>) {
                return mpi::request{};
            } else {
                return communicate_impl_(default_comm_.get(), part,
                                         local_vector, send_buffer,
                                         recv_buffer);
            }
        },
        part_);
}

#define GKO_DECLARE_COMMUNICATE(ValueType)                \
    mpi::request sparse_communicator::communicate(        \
        const matrix::Dense<ValueType>* local_vector,     \
        const detail::DenseCache<ValueType>& send_buffer, \
        const detail::DenseCache<ValueType>& recv_buffer) const

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_COMMUNICATE);

#undef GKO_DECLARE_COMMUNICATE


template <typename IndexType>
sparse_communicator::sparse_communicator(
    mpi::communicator comm,
    std::shared_ptr<const localized_partition<IndexType>> part)
    : default_comm_(create_neighborhood_comm(
          comm, part->get_recv_indices().get_target_ids(),
          part->get_send_indices().get_target_ids())),
      send_sizes_(part->get_send_indices().get_num_groups()),
      send_offsets_(part->get_send_indices().get_num_groups() + 1),
      recv_sizes_(part->get_recv_indices().get_num_groups()),
      recv_offsets_(part->get_recv_indices().get_num_groups() + 1)
{
    auto exec = part->get_executor();
    auto host_exec = exec->get_master();
    auto fill_size_offsets = [&](std::vector<int>& sizes,
                                 std::vector<int>& offsets, const auto& idxs) {
        for (int i = 0; i < idxs.get_num_groups(); ++i) {
            sizes[i] = idxs.get_indices(i).get_num_elems();
        }
        std::partial_sum(sizes.begin(), sizes.end(), offsets.begin() + 1);
    };
    fill_size_offsets(recv_sizes_, recv_offsets_, part->get_recv_indices());
    fill_size_offsets(send_sizes_, send_offsets_, part->get_send_indices());
    part_ = std::move(part);
}

#define GKO_DECLARE_SPARSE_COMMUNICATOR(IndexType) \
    sparse_communicator::sparse_communicator(      \
        mpi::communicator comm,                    \
        std::shared_ptr<const localized_partition<IndexType>> part)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_SPARSE_COMMUNICATOR);

#undef GKO_DECLARE_SPARSE_COMMUNICATOR

template <typename ValueType, typename IndexType>
mpi::request sparse_communicator::communicate_impl_(
    MPI_Comm comm, std::shared_ptr<const localized_partition<IndexType>> part,
    const matrix::Dense<ValueType>* local_vector,
    const detail::DenseCache<ValueType>& send_buffer,
    const detail::DenseCache<ValueType>& recv_buffer) const
{
    GKO_ASSERT(part->get_local_end() == local_vector->get_size()[0]);

    auto exec = local_vector->get_executor();

    auto send_idxs = part->get_send_indices();
    auto recv_idxs = part->get_recv_indices();

    recv_buffer.init(exec,
                     {recv_idxs.get_num_elems(), local_vector->get_size()[1]});

    send_buffer.init(exec,
                     {send_idxs.get_num_elems(), local_vector->get_size()[1]});
    size_type offset = 0;
    for (int i = 0; i < send_idxs.get_num_groups(); ++i) {
        // need direct support for index_set
        auto full_idxs = send_idxs.get_indices(i).to_global_indices();
        local_vector->row_gather(
            &full_idxs, send_buffer.get()->create_submatrix(
                            {offset, offset + full_idxs.get_num_elems()},
                            {0, local_vector->get_size()[1]}));
        offset += full_idxs.get_num_elems();
    }

    auto recv_ptr = recv_buffer->get_values();
    auto send_ptr = send_buffer->get_values();

    mpi::contiguous_type type(local_vector->get_size()[1],
                              mpi::type_impl<ValueType>::get_type());
    mpi::request req;
    MPI_Ineighbor_alltoallv(send_ptr, send_sizes_.data(), send_offsets_.data(),
                            type.get(), recv_ptr, recv_sizes_.data(),
                            recv_offsets_.data(), type.get(), comm, req.get());
    return req;
}

#define GKO_DECLARE_COMMUNICATE_IMPL(ValueType, IndexType)                     \
    mpi::request sparse_communicator::communicate_impl_<ValueType, IndexType>( \
        MPI_Comm comm,                                                         \
        std::shared_ptr<const localized_partition<IndexType>> part,            \
        const matrix::Dense<ValueType>* local_vector,                          \
        const detail::DenseCache<ValueType>& send_buffer,                      \
        const detail::DenseCache<ValueType>& recv_buffer) const

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COMMUNICATE_IMPL);

#undef GKO_DECLARE_COMMUNICATE_IMPL


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
