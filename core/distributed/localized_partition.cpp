// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/distributed/localized_partition.hpp>


#include "core/components/min_max_array_kernels.hpp"


namespace gko::experimental::distributed {
namespace array_kernels {


GKO_REGISTER_OPERATION(max, components::max_array);
GKO_REGISTER_OPERATION(min, components::min_array);


}  // namespace array_kernels


namespace detail {


template <typename IndexType>
IndexType get_max(const array<IndexType>& arr)
{
    auto exec = arr.get_executor();
    IndexType max;
    exec->run(array_kernels::make_max(arr, max));
    return max;
}


template <typename IndexType>
IndexType get_min(const array<IndexType>& arr)
{
    auto exec = arr.get_executor();
    IndexType min;
    exec->run(array_kernels::make_min(arr, min));
    return min;
}


template <typename IndexType>
IndexType get_max(const array_collection<IndexType>& arrs)
{
    return get_max(arrs.get_flat());
}

#define GKO_DECLARE_GET_MAX_ARRS(IndexType) \
    IndexType get_max(const array_collection<IndexType>& arrs)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_GET_MAX_ARRS);


template <typename IndexType>
IndexType get_min(const array_collection<IndexType>& arrs)
{
    return get_min(arrs.get_flat());
}

#define GKO_DECLARE_GET_MIN_ARRS(IndexType) \
    IndexType get_min(const array_collection<IndexType>& arrs)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_GET_MIN_ARRS);


}  // namespace detail


template <typename IndexType>
span_collection<IndexType> compute_span_collection(
    size_type local_size, const std::vector<comm_index_type> sizes)
{
    span_collection<IndexType> blocks;
    auto offset = static_cast<IndexType>(local_size);
    for (IndexType current_size : sizes) {
        blocks.emplace_back(offset, offset + current_size);
        offset += current_size;
    }
    return blocks;
}


template <typename IndexType>
std::shared_ptr<localized_partition<IndexType>>
localized_partition<IndexType>::build_from_blocked_recv(
    std::shared_ptr<const Executor> exec, size_type local_size,
    const std::vector<std::pair<array<index_type>, comm_index_type>>& send_idxs,
    const array<comm_index_type>& recv_ids,
    const std::vector<comm_index_type>& recv_sizes)
{
    // make sure shared indices are a subset of local indices
    GKO_ASSERT(send_idxs.empty() || send_idxs.size() == 0 ||
               local_size >=
                   detail::get_max(
                       std::max_element(send_idxs.begin(), send_idxs.end(),
                                        [](const auto& a, const auto& b) {
                                            return detail::get_max(a.first) <
                                                   detail::get_max(b.first);
                                        })
                           ->first));
    GKO_ASSERT(recv_ids.get_size() == recv_sizes.size());

    std::vector<size_type> num_send_idxs(send_idxs.size());
    std::transform(send_idxs.begin(), send_idxs.end(), num_send_idxs.begin(),
                   [](const auto& a) { return a.first.get_size(); });

    array_collection<IndexType> send_idxs_arrs(exec, num_send_idxs);
    array<comm_index_type> send_target_ids(exec->get_master(),
                                           send_idxs.size());
    for (int i = 0; i < send_idxs.size(); ++i) {
        send_idxs_arrs[i] = send_idxs[i].first;
        send_target_ids.get_data()[i] = send_idxs[i].second;
    }

    send_target_ids.set_executor(exec);

    auto intervals = compute_span_collection<IndexType>(local_size, recv_sizes);

    return std::shared_ptr<localized_partition>{new localized_partition{
        exec, local_size,
        send_indices_type{std::move(send_target_ids),
                          std::move(send_idxs_arrs)},
        recv_indices_type{recv_ids, std::move(intervals)}}};
}


std::tuple<array<comm_index_type>, std::vector<comm_index_type>>
communicate_inverse_envelope(std::shared_ptr<const Executor> exec,
                             mpi::communicator comm,
                             const array<comm_index_type>& ids,
                             const std::vector<comm_index_type>& sizes)
{
    auto host_exec = exec->get_master();
    std::vector<comm_index_type> inverse_sizes_full(comm.size());
    mpi::window<comm_index_type> window(host_exec, inverse_sizes_full.data(),
                                        inverse_sizes_full.size(), comm);
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

template <typename LocalIndexType>
array<LocalIndexType> communicate_send_gather_idxs(
    mpi::communicator comm, const array<LocalIndexType>& recv_gather_idxs,
    const array<comm_index_type>& recv_ids,
    const std::vector<comm_index_type>& recv_sizes,
    const array<comm_index_type>& send_ids,
    const std::vector<comm_index_type>& send_sizes)
{
    MPI_Comm sparse_comm;
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Dist_graph_create_adjacent(
        comm.get(), send_ids.get_size(), send_ids.get_const_data(),
        MPI_UNWEIGHTED, recv_ids.get_size(), recv_ids.get_const_data(),
        MPI_UNWEIGHTED, info, false, &sparse_comm);
    MPI_Info_free(&info);

    std::vector<comm_index_type> recv_offsets(recv_sizes.size() + 1);
    std::vector<comm_index_type> send_offsets(send_sizes.size() + 1);
    std::partial_sum(recv_sizes.data(), recv_sizes.data() + recv_sizes.size(),
                     recv_offsets.begin() + 1);
    std::partial_sum(send_sizes.data(), send_sizes.data() + send_sizes.size(),
                     send_offsets.begin() + 1);

    array<LocalIndexType> send_gather_idxs(recv_gather_idxs.get_executor(),
                                           send_offsets.back());

    MPI_Neighbor_alltoallv(
        recv_gather_idxs.get_const_data(), recv_sizes.data(),
        recv_offsets.data(), mpi::type_impl<LocalIndexType>::get_type(),
        send_gather_idxs.get_data(), send_sizes.data(), send_offsets.data(),
        mpi::type_impl<LocalIndexType>::get_type(), sparse_comm);

    return send_gather_idxs;
}

template <typename IndexType>
std::shared_ptr<localized_partition<IndexType>>
localized_partition<IndexType>::build_from_remote_send_indices(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    size_type local_size, const array<comm_index_type>& recv_ids,
    const std::vector<comm_index_type>& recv_sizes,
    const array<IndexType>& remote_send_indices)
{
    GKO_ASSERT(recv_ids.get_size() == recv_sizes.size());

    auto send_envelope =
        communicate_inverse_envelope(exec, comm, recv_ids, recv_sizes);
    auto send_ids = std::move(std::get<0>(send_envelope));
    auto send_sizes = std::move(std::get<1>(send_envelope));

    array_collection<IndexType> send_idxs(
        communicate_send_gather_idxs(comm, remote_send_indices, recv_ids,
                                     recv_sizes, send_ids, send_sizes),
        send_sizes);

    std::vector<comm_index_type> send_offsets(send_sizes.size() + 1, 0);
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     send_offsets.begin() + 1);

    auto intervals = compute_span_collection<IndexType>(local_size, recv_sizes);

    return std::shared_ptr<localized_partition<IndexType>>(
        new localized_partition(
            exec, local_size,
            send_indices_type{std::move(send_ids), std::move(send_idxs)},
            recv_indices_type{recv_ids, std::move(intervals)}));
}

#define GKO_DECLARE_LOCALIZED_PARTITION(_type) class localized_partition<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_LOCALIZED_PARTITION);


}  // namespace gko::experimental::distributed
