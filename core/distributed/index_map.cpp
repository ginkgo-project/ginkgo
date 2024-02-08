// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/distributed/index_map.hpp>


#include "core/components/min_max_array_kernels.hpp"
#include "core/distributed/index_map_kernels.hpp"


namespace gko {
namespace array_kernels {


GKO_REGISTER_OPERATION(max, components::max_array);
GKO_REGISTER_OPERATION(min, components::min_array);


}  // namespace array_kernels


namespace index_map_kernels {


GKO_REGISTER_OPERATION(build_mapping, index_map::build_mapping);


}


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


}  // namespace detail


namespace collection {


template <typename IndexType>
IndexType get_max(const array<IndexType>& arrs)
{
    return detail::get_max(arrs.get_flat());
}

#define GKO_DECLARE_GET_MAX_ARRS(IndexType) \
    IndexType get_max(const collection::array<IndexType>& arrs)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_GET_MAX_ARRS);


template <typename IndexType>
IndexType get_min(const array<IndexType>& arrs)
{
    return detail::get_min(arrs.get_flat());
}

#define GKO_DECLARE_GET_MIN_ARRS(IndexType) \
    IndexType get_min(const collection::array<IndexType>& arrs)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_GET_MIN_ARRS);


}  // namespace collection


namespace experimental::distributed {


template <typename IndexType>
collection::span<IndexType> compute_span_collection(
    size_type local_size, const std::vector<comm_index_type> sizes)
{
    collection::span<IndexType> blocks;
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

    collection::array<IndexType> send_idxs_arrs(exec, num_send_idxs);
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

    collection::array<IndexType> send_idxs(
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


template <typename LocalIndexType, typename GlobalIndexType>
semi_global_index<LocalIndexType>
index_map<LocalIndexType, GlobalIndexType>::get_semi_global(
    GlobalIndexType id) const
{
    auto host_partition = make_temporary_clone(exec_->get_master(), partition_);

    // find range that contains id
    auto ranges_bounds = host_partition->get_range_bounds();
    auto range_id = std::distance(
        ranges_bounds,
        std::lower_bound(ranges_bounds,
                         ranges_bounds + host_partition->get_num_ranges(), id));

    // find part that contains range
    auto part_id = host_partition->get_part_ids()[range_id];

    // return local id wrt. to the range id
    auto local_id = static_cast<LocalIndexType>(
        id - host_partition->get_range_starting_indices()[range_id]);

    // store global-to-local for faster semi-global lookup
    return {part_id, local_id};
}


template <typename LocalIndexType, typename GlobalIndexType>
LocalIndexType index_map<LocalIndexType, GlobalIndexType>::get_local(
    comm_index_type process_id, LocalIndexType semi_global_id) const
{
    auto exec = target_ids_.get_executor();
    auto host_process_ids =
        make_temporary_clone(exec->get_master(), &target_ids_);
    auto set_id =
        std::distance(target_ids_.get_const_data(),
                      std::lower_bound(host_process_ids->get_const_data(),
                                       host_process_ids->get_const_data() +
                                           host_process_ids->get_size(),
                                       process_id));

    auto& remote_idxs = remote_local_idxs_[set_id];

    return id_set_offsets_[set_id] +
           std::distance(remote_idxs.get_const_data(),
                         std::lower_bound(remote_idxs.get_const_data(),
                                          remote_idxs.get_const_data() +
                                              remote_idxs.get_size(),
                                          semi_global_id));
}


template <typename LocalIndexType, typename GlobalIndexType>
array<LocalIndexType> index_map<LocalIndexType, GlobalIndexType>::get_local(
    comm_index_type process_id,
    const array<LocalIndexType>& semi_global_ids) const
{
    auto exec = semi_global_ids.get_executor();
    auto host_semi_global_ids =
        make_temporary_clone(exec->get_master(), &semi_global_ids);

    array<LocalIndexType> local_ids{exec->get_master(),
                                    semi_global_ids.get_size()};

    auto set_id = std::distance(
        target_ids_.get_const_data(),
        std::lower_bound(target_ids_.get_const_data(),
                         target_ids_.get_const_data() + target_ids_.get_size(),
                         process_id));

    for (size_type i = 0; i < host_semi_global_ids->get_size(); ++i) {
        auto current_set = remote_local_idxs_[set_id];
        local_ids.get_data()[i] =
            id_set_offsets_[set_id] +
            std::distance(current_set.get_data(),
                          std::lower_bound(
                              current_set.get_data(),
                              current_set.get_data() + current_set.get_size(),
                              host_semi_global_ids->get_const_data()[i]));
    }

    return local_ids;
}


template <typename LocalIndexType, typename GlobalIndexType>
array<LocalIndexType> index_map<LocalIndexType, GlobalIndexType>::get_local(
    const array<comm_index_type>& process_ids,
    const collection::array<LocalIndexType>& semi_global_ids) const
{
    auto exec = process_ids.get_executor();
    auto host_process_ids =
        make_temporary_clone(exec->get_master(), &process_ids);

    array<LocalIndexType> local_ids{exec->get_master(),
                                    semi_global_ids.get_flat().get_size()};

    std::vector<size_type> query_size_offsets(semi_global_ids.size());
    std::exclusive_scan(
        semi_global_ids.begin(), semi_global_ids.end(),
        query_size_offsets.begin(), 0,
        [](const auto& acc, const auto& a) { return acc + a.get_size(); });

    for (size_type i = 0; i < host_process_ids->get_size(); ++i) {
        auto pid = host_process_ids->get_const_data()[i];

        auto current_view =
            make_array_view(exec->get_master(), semi_global_ids[i].get_size(),
                            local_ids.get_data() + query_size_offsets[i]);
        auto current_result = get_local(pid, semi_global_ids[i]);
        current_view = current_result;
    }

    return local_ids;
}
template <typename LocalIndexType, typename GlobalIndexType>
array<LocalIndexType> index_map<LocalIndexType, GlobalIndexType>::get_local(
    const GlobalIndexType global_ids) const GKO_NOT_IMPLEMENTED;


template <typename LocalIndexType, typename GlobalIndexType>
index_map<LocalIndexType, GlobalIndexType>::index_map(
    std::shared_ptr<const Executor> exec, std::shared_ptr<const part_type> part,
    const array<GlobalIndexType>& recv_connections,
    const collection::array<LocalIndexType>& send_connections)
    : exec_(std::move(exec)),
      target_ids_(exec_),
      remote_local_idxs_(exec_),
      remote_global_idxs_(exec_)
{
    exec_->run(index_map_kernels::make_build_mapping(
        part.get(), recv_connections, target_ids_, remote_local_idxs_,
        remote_global_idxs_));

    std::vector<comm_index_type> recv_sizes(remote_local_idxs_.size());
    std::transform(remote_local_idxs_.begin(), remote_local_idxs_.end(),
                   recv_sizes.begin(),
                   [](const auto& a) { return a.get_size(); });

    // auto send_envelope =
    //     communicate_inverse_envelope(exec, comm, target_ids_, recv_sizes);
    // auto send_ids = std::move(std::get<0>(send_envelope));
    // auto send_sizes = std::move(std::get<1>(send_envelope));
    //
    // local_idxs_ =
    //     collection::array(communicate_send_gather_idxs(
    //                           comm, remote_local_idxs_.get_flat(),
    //                           target_ids_, recv_sizes, send_ids, send_sizes),
    //                       send_sizes);
}

#define GKO_DECLARE_INDEX_MAP(_ltype, _gtype) class index_map<_ltype, _gtype>
GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_INDEX_MAP);

#define GKO_DECLARE_LOCALIZED_PARTITION(_type) class localized_partition<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_LOCALIZED_PARTITION);


}  // namespace experimental::distributed
}  // namespace gko
