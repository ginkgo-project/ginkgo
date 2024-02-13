// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/distributed/index_map.hpp>


#include <sys/socket.h>


#include "core/components/min_max_array_kernels.hpp"
#include "core/distributed/index_map_kernels.hpp"


namespace gko {
namespace array_kernels {


GKO_REGISTER_OPERATION(max, components::max_array);
GKO_REGISTER_OPERATION(min, components::min_array);


}  // namespace array_kernels


namespace index_map_kernels {


GKO_REGISTER_OPERATION(build_mapping, index_map::build_mapping);
GKO_REGISTER_OPERATION(get_local, index_map::get_local);

}  // namespace index_map_kernels


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

template <typename LocalIndexType, typename GlobalIndexType>
array<LocalIndexType> index_map<LocalIndexType, GlobalIndexType>::get_local(
    const array<GlobalIndexType>& global_ids, index_space is) const
{
    array<LocalIndexType> local_ids(exec_);

    exec_->run(index_map_kernels::make_get_local(
        partition_.get(), recv_target_ids_, remote_global_idxs_, rank_,
        global_ids, is, local_ids));

    return local_ids;
}


template <typename LocalIndexType, typename GlobalIndexType>
index_map<LocalIndexType, GlobalIndexType>::index_map(
    std::shared_ptr<const Executor> exec, std::shared_ptr<const part_type> part,
    comm_index_type rank, const array<GlobalIndexType>& recv_connections,
    const array<comm_index_type>& send_target_ids,
    const collection::array<LocalIndexType>& send_connections)
    : exec_(std::move(exec)),
      partition_(std::move(part)),
      rank_(rank),
      recv_target_ids_(exec_),
      remote_local_idxs_(exec_),
      remote_global_idxs_(exec_),
      recv_set_offsets_(exec_),
      send_target_ids_(send_target_ids),
      local_idxs_(send_connections)
{
    GKO_THROW_IF_INVALID(send_target_ids_.get_size() == local_idxs_.size(),
                         "The size of the send_targets_ids and "
                         "send_connections doesn't match up");

    exec_->run(index_map_kernels::make_build_mapping(
        partition_.get(), recv_connections, recv_target_ids_,
        remote_local_idxs_, remote_global_idxs_));

    std::vector<comm_index_type> recv_sizes(remote_local_idxs_.size());
    std::transform(remote_local_idxs_.begin(), remote_local_idxs_.end(),
                   recv_sizes.begin(),
                   [](const auto& a) { return a.get_size(); });
    recv_set_offsets_.set_executor(exec_->get_master());
    recv_set_offsets_.resize_and_reset(recv_sizes.size() + 1);
    recv_set_offsets_.fill(0);
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     recv_set_offsets_.get_data() + 1);
}

template <typename LocalIndexType, typename GlobalIndexType>
index_map<LocalIndexType, GlobalIndexType>::index_map(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    std::shared_ptr<const part_type> part,
    const array<GlobalIndexType>& recv_connections)
    : exec_(std::move(exec)),
      partition_(std::move(part)),
      rank_(comm.rank()),
      recv_target_ids_(exec_),
      remote_local_idxs_(exec_),
      remote_global_idxs_(exec_),
      recv_set_offsets_(exec_),
      send_target_ids_(exec_),
      local_idxs_(exec_)
{
    exec_->run(index_map_kernels::make_build_mapping(
        partition_.get(), recv_connections, recv_target_ids_,
        remote_local_idxs_, remote_global_idxs_));

    std::vector<comm_index_type> recv_sizes(remote_local_idxs_.size());
    std::transform(remote_local_idxs_.begin(), remote_local_idxs_.end(),
                   recv_sizes.begin(),
                   [](const auto& a) { return a.get_size(); });
    recv_set_offsets_.set_executor(exec_->get_master());
    recv_set_offsets_.resize_and_reset(recv_sizes.size() + 1);
    recv_set_offsets_.fill(0);
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     recv_set_offsets_.get_data() + 1);
    recv_set_offsets_.set_executor(exec_);

    auto send_envelope =
        communicate_inverse_envelope(exec_, comm, recv_target_ids_, recv_sizes);
    send_target_ids_ = std::move(std::get<0>(send_envelope));
    auto send_sizes = std::move(std::get<1>(send_envelope));

    local_idxs_ = collection::array(
        communicate_send_gather_idxs(comm, remote_local_idxs_.get_flat(),
                                     recv_target_ids_, recv_sizes,
                                     send_target_ids_, send_sizes),
        send_sizes);
}

#define GKO_DECLARE_INDEX_MAP(_ltype, _gtype) class index_map<_ltype, _gtype>
GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_INDEX_MAP);


}  // namespace experimental::distributed
}  // namespace gko
