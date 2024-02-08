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


template <typename ValueType>
mpi::request sparse_communicator::communicate(
    const matrix::Dense<ValueType>* local_vector,
    const detail::DenseCache<ValueType>& send_buffer,
    const detail::DenseCache<ValueType>& recv_buffer) const
{
    return std::visit(
        [&, this](const auto& send_idxs) {
            if constexpr (std::is_same_v<std::decay_t<decltype(send_idxs)>,
                                         std::monostate>) {
                return mpi::request{};
            } else {
                return communicate_impl_(default_comm_.get(), send_idxs,
                                         local_vector, send_buffer,
                                         recv_buffer);
            }
        },
        send_idxs_);
}

#define GKO_DECLARE_COMMUNICATE(ValueType)                \
    mpi::request sparse_communicator::communicate(        \
        const matrix::Dense<ValueType>* local_vector,     \
        const detail::DenseCache<ValueType>& send_buffer, \
        const detail::DenseCache<ValueType>& recv_buffer) const

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_COMMUNICATE);

#undef GKO_DECLARE_COMMUNICATE


template <typename LocalIndexType, typename GlobalIndexType>
sparse_communicator::sparse_communicator(
    mpi::communicator comm,
    const index_map<LocalIndexType, GlobalIndexType>& imap)
    : default_comm_(create_neighborhood_comm(comm, imap.get_recv_target_ids(),
                                             imap.get_send_target_ids())),
      send_sizes_(imap.get_local_shared_idxs().size()),
      send_offsets_(send_sizes_.size() + 1),
      recv_sizes_(imap.get_remote_local_idxs().size()),
      recv_offsets_(recv_sizes_.size() + 1)
{
    auto exec = imap.get_executor();
    auto host_exec = exec->get_master();
    auto fill_size_offsets = [&](std::vector<int>& sizes,
                                 std::vector<int>& offsets,
                                 const auto& remote_idxs) {
        for (int i = 0; i < remote_idxs.size(); ++i) {
            sizes[i] = remote_idxs[i].get_size();
        }
        std::partial_sum(sizes.begin(), sizes.end(), offsets.begin() + 1);
    };
    fill_size_offsets(recv_sizes_, recv_offsets_, imap.get_remote_local_idxs());
    fill_size_offsets(send_sizes_, send_offsets_, imap.get_local_shared_idxs());

    send_idxs_ = imap.get_local_shared_idxs().get_flat();
}

#define GKO_DECLARE_SPARSE_COMMUNICATOR(LocalIndexType, GlobalIndexType) \
    sparse_communicator::sparse_communicator(                            \
        mpi::communicator comm,                                          \
        const index_map<LocalIndexType, GlobalIndexType>& imap)

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_SPARSE_COMMUNICATOR);

#undef GKO_DECLARE_SPARSE_COMMUNICATOR

template <typename ValueType, typename LocalIndexType>
mpi::request sparse_communicator::communicate_impl_(
    MPI_Comm comm, const array<LocalIndexType>& send_idxs,
    const matrix::Dense<ValueType>* local_vector,
    const detail::DenseCache<ValueType>& send_buffer,
    const detail::DenseCache<ValueType>& recv_buffer) const
{
    auto exec = local_vector->get_executor();

    recv_buffer.init(exec, {static_cast<size_type>(recv_offsets_.back()),
                            local_vector->get_size()[1]});
    send_buffer.init(exec, {static_cast<size_type>(send_offsets_.back()),
                            local_vector->get_size()[1]});

    local_vector->row_gather(&send_idxs, send_buffer.get());

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

#define GKO_DECLARE_COMMUNICATE_IMPL(ValueType, LocalIndexType) \
    mpi::request sparse_communicator::communicate_impl_(        \
        MPI_Comm comm, const array<LocalIndexType>& send_idxs,  \
        const matrix::Dense<ValueType>* local_vector,           \
        const detail::DenseCache<ValueType>& send_buffer,       \
        const detail::DenseCache<ValueType>& recv_buffer) const

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COMMUNICATE_IMPL);

#undef GKO_DECLARE_COMMUNICATE_IMPL
}  // namespace distributed
}  // namespace experimental
}  // namespace gko
