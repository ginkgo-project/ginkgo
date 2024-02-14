// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/distributed/sparse_communicator.hpp>


#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace experimental {
namespace distributed {


/**
 * \brief Computes the inverse envelope (target-ids, sizes) for a given
 *        one-sided communication pattern.
 *
 * \param exec the executor, this will always use the host executor
 * \param comm communicator
 * \param ids target ids of the one-sided operation
 * \param sizes number of elements send to each id
 *
 * \return the inverse envelope consisting of the target-ids and the sizes
 */
std::tuple<array<comm_index_type>, std::vector<comm_index_type>>
communicate_inverse_envelope(std::shared_ptr<const Executor> exec,
                             mpi::communicator comm,
                             const array<comm_index_type>& ids,
                             const std::vector<comm_index_type>& sizes)
{
    auto host_exec = exec->get_master();
    std::vector<comm_index_type> inverse_sizes_full(comm.size());
    mpi::window<comm_index_type> window(host_exec, inverse_sizes_full.data(),
                                        inverse_sizes_full.size(), comm,
                                        sizeof(comm_index_type), MPI_INFO_ENV);
    window.fence();
    for (int i = 0; i < ids.get_size(); ++i) {
        window.put(host_exec, sizes.data() + i, 1, ids.get_const_data()[i],
                   comm.rank(), 1);
    }
    window.fence();

    std::vector<comm_index_type> inverse_sizes;
    std::vector<comm_index_type> inverse_ids;
    for (int i = 0; i < inverse_sizes_full.size(); ++i) {
        if (inverse_sizes_full[i] > 0) {
            inverse_ids.push_back(i);
            inverse_sizes.push_back(inverse_sizes_full[i]);
        }
    }

    return std::make_tuple(
        array<comm_index_type>{exec, inverse_ids.begin(), inverse_ids.end()},
        std::move(inverse_sizes));
}


/**
 * \brief
 * \tparam LocalIndexType index type
 * \param comm neighborhood communicator
 * \param remote_local_idxs the remote indices in their local indexing
 * \param recv_sizes the sizes that segregate remote_local_idxs
 * \param send_sizes the number of local indices per rank that are part of
 *                   remote_local_idxs on that ranks
 * \return the local indices that are part of remote_local_idxs on other ranks,
 *         ordered by the rank ordering of the communicator
 */
template <typename LocalIndexType>
array<LocalIndexType> communicate_send_gather_idxs(
    mpi::communicator comm, const array<LocalIndexType>& remote_local_idxs,
    const array<comm_index_type>& recv_ids,
    const std::vector<comm_index_type>& recv_sizes,
    const array<comm_index_type>& send_ids,
    const std::vector<comm_index_type>& send_sizes)
{
    // create temporary inverse sparse communicator
    MPI_Comm sparse_comm;
    MPI_Info info;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Info_create(&info));
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Dist_graph_create_adjacent(
        comm.get(), send_ids.get_size(), send_ids.get_const_data(),
        MPI_UNWEIGHTED, recv_ids.get_size(), recv_ids.get_const_data(),
        MPI_UNWEIGHTED, info, false, &sparse_comm));
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Info_free(&info));

    std::vector<comm_index_type> recv_offsets(recv_sizes.size() + 1);
    std::vector<comm_index_type> send_offsets(send_sizes.size() + 1);
    std::partial_sum(recv_sizes.data(), recv_sizes.data() + recv_sizes.size(),
                     recv_offsets.begin() + 1);
    std::partial_sum(send_sizes.data(), send_sizes.data() + send_sizes.size(),
                     send_offsets.begin() + 1);

    array<LocalIndexType> send_gather_idxs(remote_local_idxs.get_executor(),
                                           send_offsets.back());

    GKO_ASSERT_NO_MPI_ERRORS(MPI_Neighbor_alltoallv(
        remote_local_idxs.get_const_data(), recv_sizes.data(),
        recv_offsets.data(), mpi::type_impl<LocalIndexType>::get_type(),
        send_gather_idxs.get_data(), send_sizes.data(), send_offsets.data(),
        mpi::type_impl<LocalIndexType>::get_type(), sparse_comm));
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_free(&sparse_comm));

    return send_gather_idxs;
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
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Info_dup(MPI_INFO_ENV, &info));
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Dist_graph_create_adjacent(
        base.get(), in_degree, sources_host->get_const_data(), MPI_UNWEIGHTED,
        out_degree, destinations_host->get_const_data(), MPI_UNWEIGHTED, info,
        false, &new_comm));
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Info_free(&info));

    return mpi::communicator::create_owning(new_comm, base.force_host_buffer());
}


template <typename LocalIndexType, typename GlobalIndexType>
sparse_communicator::sparse_communicator(
    mpi::communicator comm,
    const index_map<LocalIndexType, GlobalIndexType>& imap)
    : default_comm_(MPI_COMM_NULL),
      recv_sizes_(imap.get_remote_local_idxs().size()),
      recv_offsets_(recv_sizes_.size() + 1)
{
    auto exec = imap.get_executor();

    auto& recv_target_ids = imap.get_remote_target_ids();
    std::transform(imap.get_remote_global_idxs().begin(),
                   imap.get_remote_global_idxs().end(), recv_sizes_.begin(),
                   [](const auto& a) { return a.get_size(); });
    auto send_envelope =
        communicate_inverse_envelope(exec, comm, recv_target_ids, recv_sizes_);
    auto& send_target_ids = std::get<0>(send_envelope);
    send_sizes_ = std::move(std::get<1>(send_envelope));
    send_offsets_.resize(send_sizes_.size() + 1);

    std::partial_sum(recv_sizes_.begin(), recv_sizes_.end(),
                     recv_offsets_.begin() + 1);
    std::partial_sum(send_sizes_.begin(), send_sizes_.end(),
                     send_offsets_.begin() + 1);

    send_idxs_ = communicate_send_gather_idxs(
        comm, imap.get_remote_local_idxs().get_flat(), recv_target_ids,
        recv_sizes_, send_target_ids, send_sizes_);

    default_comm_ =
        create_neighborhood_comm(comm, recv_target_ids, send_target_ids);
}

#define GKO_DECLARE_SPARSE_COMMUNICATOR(LocalIndexType, GlobalIndexType) \
    sparse_communicator::sparse_communicator(                            \
        mpi::communicator comm,                                          \
        const index_map<LocalIndexType, GlobalIndexType>& imap)

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_SPARSE_COMMUNICATOR);

#undef GKO_DECLARE_SPARSE_COMMUNICATOR


std::shared_ptr<const Executor> get_mpi_exec(
    std::shared_ptr<const Executor> exec, mpi::communicator comm)
{
    bool use_host_buffer = mpi::requires_host_buffer(exec, comm);
    return use_host_buffer ? exec->get_master() : exec;
}


template <typename ValueType>
mpi::request sparse_communicator::communicate(
    const matrix::Dense<ValueType>* local_vector,
    const detail::DenseCache<ValueType>& send_buffer,
    const detail::DenseCache<ValueType>& recv_buffer) const
{
    return std::visit(
        [&, this](const auto& send_idxs) {
            auto mpi_exec =
                get_mpi_exec(local_vector->get_executor(), default_comm_);
            recv_buffer.init(mpi_exec,
                             {static_cast<size_type>(recv_offsets_.back()),
                              local_vector->get_size()[1]});
            send_buffer.init(mpi_exec,
                             {static_cast<size_type>(send_offsets_.back()),
                              local_vector->get_size()[1]});
            if constexpr (std::is_same_v<std::decay_t<decltype(send_idxs)>,
                                         std::monostate>) {
                return mpi::request{};
            } else {
                return communicate_impl_(default_comm_, send_idxs, local_vector,
                                         send_buffer, recv_buffer);
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


template <typename ValueType, typename LocalIndexType>
mpi::request sparse_communicator::communicate_impl_(
    mpi::communicator comm, const array<LocalIndexType>& send_idxs,
    const matrix::Dense<ValueType>* local_vector,
    const detail::DenseCache<ValueType>& send_buffer,
    const detail::DenseCache<ValueType>& recv_buffer) const
{
    auto exec = local_vector->get_executor();

    auto mpi_exec = get_mpi_exec(exec, default_comm_);

    local_vector->row_gather(&send_idxs, send_buffer.get());

    auto recv_ptr = recv_buffer->get_values();
    auto send_ptr = send_buffer->get_values();

    exec->synchronize();
    mpi::contiguous_type type(local_vector->get_size()[1],
                              mpi::type_impl<ValueType>::get_type());
    mpi::request req;
    auto g = mpi_exec->get_scoped_device_id_guard();
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Ineighbor_alltoallv(
        send_ptr, send_sizes_.data(), send_offsets_.data(), type.get(),
        recv_ptr, recv_sizes_.data(), recv_offsets_.data(), type.get(),
        comm.get(), req.get()));
    return req;
}

#define GKO_DECLARE_COMMUNICATE_IMPL(ValueType, LocalIndexType)         \
    mpi::request sparse_communicator::communicate_impl_(                \
        mpi::communicator comm, const array<LocalIndexType>& send_idxs, \
        const matrix::Dense<ValueType>* local_vector,                   \
        const detail::DenseCache<ValueType>& send_buffer,               \
        const detail::DenseCache<ValueType>& recv_buffer) const

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COMMUNICATE_IMPL);

#undef GKO_DECLARE_COMMUNICATE_IMPL


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
