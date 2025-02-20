// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/index_map.hpp"

#include "core/distributed/index_map_kernels.hpp"


namespace gko {
namespace index_map_kernels {


GKO_REGISTER_OPERATION(build_mapping, index_map::build_mapping);
GKO_REGISTER_OPERATION(map_to_local, index_map::map_to_local);
GKO_REGISTER_OPERATION(map_to_global, index_map::map_to_global);


}  // namespace index_map_kernels


namespace experimental {
namespace distributed {


template <typename LocalIndexType, typename GlobalIndexType>
size_type index_map<LocalIndexType, GlobalIndexType>::get_local_size() const
{
    return partition_ ? partition_->get_part_size(rank_) : 0;
}


template <typename LocalIndexType, typename GlobalIndexType>
size_type index_map<LocalIndexType, GlobalIndexType>::get_non_local_size() const
{
    return remote_global_idxs_.get_size();
}


template <typename LocalIndexType, typename GlobalIndexType>
const segmented_array<GlobalIndexType>&
index_map<LocalIndexType, GlobalIndexType>::get_remote_global_idxs() const
{
    return remote_global_idxs_;
}


template <typename LocalIndexType, typename GlobalIndexType>
const segmented_array<LocalIndexType>&
index_map<LocalIndexType, GlobalIndexType>::get_remote_local_idxs() const
{
    return remote_local_idxs_;
}


template <typename LocalIndexType, typename GlobalIndexType>
const array<comm_index_type>&
index_map<LocalIndexType, GlobalIndexType>::get_remote_target_ids() const
{
    return remote_target_ids_;
}


template <typename LocalIndexType, typename GlobalIndexType>
std::shared_ptr<const Executor>
index_map<LocalIndexType, GlobalIndexType>::get_executor() const
{
    return exec_;
}


template <typename LocalIndexType, typename GlobalIndexType>
size_type index_map<LocalIndexType, GlobalIndexType>::get_global_size() const
{
    return partition_ ? partition_->get_size() : 0;
}


template <typename LocalIndexType, typename GlobalIndexType>
array<LocalIndexType> index_map<LocalIndexType, GlobalIndexType>::map_to_local(
    const array<GlobalIndexType>& global_ids, index_space index_space_v) const
{
    array<LocalIndexType> local_ids(exec_);

    exec_->run(index_map_kernels::make_map_to_local(
        partition_.get(), remote_target_ids_, to_device(remote_global_idxs_),
        rank_, global_ids, index_space_v, local_ids));

    return local_ids;
}


template <typename LocalIndexType, typename GlobalIndexType>
array<GlobalIndexType>
index_map<LocalIndexType, GlobalIndexType>::map_to_global(
    const array<LocalIndexType>& local_idxs, index_space index_space_v) const
{
    array<GlobalIndexType> global_idxs(exec_);

    exec_->run(index_map_kernels::make_map_to_global(
        to_device_const(partition_.get()), to_device(remote_global_idxs_),
        rank_, local_idxs, index_space_v, global_idxs));

    return global_idxs;
}


template <typename LocalIndexType, typename GlobalIndexType>
index_map<LocalIndexType, GlobalIndexType>::index_map(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const partition_type> partition, comm_index_type rank,
    const array<GlobalIndexType>& recv_connections)
    : exec_(std::move(exec)),
      partition_(std::move(partition)),
      rank_(rank),
      remote_target_ids_(exec_),
      remote_local_idxs_(exec_),
      remote_global_idxs_(exec_)
{
    array<LocalIndexType> flat_remote_local_idxs(exec_);
    array<GlobalIndexType> flat_remote_global_idxs(exec_);
    array<int64> remote_sizes(exec_);
    exec_->run(index_map_kernels::make_build_mapping(
        partition_.get(), recv_connections, remote_target_ids_,
        flat_remote_local_idxs, flat_remote_global_idxs, remote_sizes));
    remote_local_idxs_ = segmented_array<LocalIndexType>::create_from_sizes(
        std::move(flat_remote_local_idxs), remote_sizes);
    remote_global_idxs_ = segmented_array<GlobalIndexType>::create_from_sizes(
        std::move(flat_remote_global_idxs), remote_sizes);
}


template <typename LocalIndexType, typename GlobalIndexType>
index_map<LocalIndexType, GlobalIndexType>::index_map(
    std::shared_ptr<const Executor> exec)
    : exec_(exec),
      partition_(nullptr),
      remote_target_ids_(exec),
      remote_local_idxs_(exec),
      remote_global_idxs_(exec)
{}


template <typename LocalIndexType, typename GlobalIndexType>
index_map<LocalIndexType, GlobalIndexType>&
index_map<LocalIndexType, GlobalIndexType>::operator=(const index_map& other)
{
    if (this != &other) {
        partition_ = other.partition_;
        rank_ = other.rank_;
        remote_target_ids_ = other.remote_target_ids_;
        remote_local_idxs_ = other.remote_local_idxs_;
        remote_global_idxs_ = other.remote_global_idxs_;
    }
    return *this;
}


template <typename LocalIndexType, typename GlobalIndexType>
index_map<LocalIndexType, GlobalIndexType>&
index_map<LocalIndexType, GlobalIndexType>::operator=(index_map&& other)
{
    if (this != &other) {
        partition_ = std::move(other.partition_);
        rank_ = other.rank_;
        remote_target_ids_ = std::move(other.remote_target_ids_);
        remote_local_idxs_ = std::move(other.remote_local_idxs_);
        remote_global_idxs_ = std::move(other.remote_global_idxs_);
    }
    return *this;
}


template <typename LocalIndexType, typename GlobalIndexType>
index_map<LocalIndexType, GlobalIndexType>::index_map(const index_map& other)
    : exec_(other.get_executor()),
      remote_local_idxs_(other.get_executor()),
      remote_global_idxs_(other.get_executor())
{
    *this = other;
}


template <typename LocalIndexType, typename GlobalIndexType>
index_map<LocalIndexType, GlobalIndexType>::index_map(
    index_map&& other) noexcept
    : exec_(other.exec_),
      remote_local_idxs_(other.get_executor()),
      remote_global_idxs_(other.get_executor())
{
    *this = std::move(other);
}


#define GKO_DECLARE_INDEX_MAP(_ltype, _gtype) struct index_map<_ltype, _gtype>

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_INDEX_MAP);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
