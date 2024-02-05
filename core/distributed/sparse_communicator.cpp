// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
    auto in_degree = static_cast<comm_index_type>(sources.get_size());
    auto out_degree = static_cast<comm_index_type>(destinations.get_size());

    auto sources_host =
        make_temporary_clone(sources.get_executor()->get_master(), &sources);
    auto destinations_host = make_temporary_clone(
        destinations.get_executor()->get_master(), &destinations);

    // adjacent constructor guarantees that querying sources/destinations
    // will result in the array having the same order as defined here
    MPI_Comm new_comm;
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Dist_graph_create_adjacent(
        base.get(), in_degree, sources_host->get_const_data(), MPI_UNWEIGHTED,
        out_degree, destinations_host->get_const_data(), MPI_UNWEIGHTED, info,
        false, &new_comm);
    MPI_Info_free(&info);
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
    const ::gko::detail::DenseCache<ValueType>& send_buffer,
    const ::gko::detail::DenseCache<ValueType>& recv_buffer) const
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

#define GKO_DECLARE_COMMUNICATE(ValueType)                       \
    mpi::request sparse_communicator::communicate(               \
        const matrix::Dense<ValueType>* local_vector,            \
        const ::gko::detail::DenseCache<ValueType>& send_buffer, \
        const ::gko::detail::DenseCache<ValueType>& recv_buffer) const

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_COMMUNICATE);

#undef GKO_DECLARE_COMMUNICATE


template <typename IndexType>
sparse_communicator::sparse_communicator(
    mpi::communicator comm,
    std::shared_ptr<const localized_partition<IndexType>> part)
    : default_comm_(
          create_neighborhood_comm(comm, part->get_recv_indices().target_ids,
                                   part->get_send_indices().target_ids)),
      send_sizes_(part->get_send_indices().idxs.size()),
      send_offsets_(part->get_send_indices().idxs.size() + 1),
      recv_sizes_(part->get_recv_indices().idxs.size()),
      recv_offsets_(part->get_recv_indices().idxs.size() + 1)
{
    auto exec = part->get_executor();
    auto host_exec = exec->get_master();
    auto fill_size_offsets = [&](std::vector<int>& sizes,
                                 std::vector<int>& offsets,
                                 const auto& remote_idxs) {
        for (int i = 0; i < remote_idxs.idxs.size(); ++i) {
            sizes[i] = detail::get_size(remote_idxs.idxs[i]);
        }
        std::partial_sum(sizes.begin(), sizes.end(), offsets.begin() + 1);
    };
    fill_size_offsets(recv_sizes_, recv_offsets_, part->get_recv_indices());
    fill_size_offsets(send_sizes_, send_offsets_, part->get_send_indices());

    send_idxs_ = part->get_send_indices().idxs.get_flat();

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
    const ::gko::detail::DenseCache<ValueType>& send_buffer,
    const ::gko::detail::DenseCache<ValueType>& recv_buffer) const
{
    GKO_ASSERT(part->get_local_end() == local_vector->get_size()[0]);

    auto exec = local_vector->get_executor();

    auto send_idxs = part->get_send_indices();
    auto recv_idxs = part->get_recv_indices();

    recv_buffer.init(
        exec, {detail::get_size(recv_idxs.idxs), local_vector->get_size()[1]});

    send_buffer.init(
        exec, {detail::get_size(send_idxs.idxs), local_vector->get_size()[1]});

    auto& full_send_idxs = std::get<array<IndexType>>(send_idxs_);
    local_vector->row_gather(&full_send_idxs, send_buffer.get());

    auto recv_ptr = recv_buffer->get_values();
    auto send_ptr = send_buffer->get_values();

    exec->synchronize();
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
        const ::gko::detail::DenseCache<ValueType>& send_buffer,               \
        const ::gko::detail::DenseCache<ValueType>& recv_buffer) const

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COMMUNICATE_IMPL);

#undef GKO_DECLARE_COMMUNICATE_IMPL


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
