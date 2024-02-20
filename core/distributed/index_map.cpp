// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/distributed/index_map.hpp>


#include "core/distributed/index_map_kernels.hpp"


namespace gko {
namespace index_map_kernels {


GKO_REGISTER_OPERATION(build_mapping, index_map::build_mapping);
GKO_REGISTER_OPERATION(get_local, index_map::get_local);

}  // namespace index_map_kernels


namespace experimental {
namespace distributed {


template <typename LocalIndexType, typename GlobalIndexType>
GlobalIndexType index_map<LocalIndexType, GlobalIndexType>::get_global(
    const LocalIndexType& local_id, index_space is) const
{
    if (is == index_space::local) {
        auto host_part = make_temporary_clone(exec_->get_master(), partition_);
        auto host_local_ranges =
            make_temporary_clone(exec_->get_master(), &local_ranges_);
        auto ranges = host_part->get_range_bounds();
        auto range_start_idxs = host_part->get_range_starting_indices();

        auto local_gid = static_cast<GlobalIndexType>(local_id);

        // find-last
        auto range_id = static_cast<size_type>(-1);
        for (size_type i = 0; i < host_local_ranges->get_size(); ++i) {
            if (range_start_idxs[host_local_ranges->get_const_data()[i]] <=
                local_gid) {
                range_id = host_local_ranges->get_const_data()[i];
            }
        }
        GKO_THROW_IF_INVALID(range_id != static_cast<size_type>(-1),
                             "Index not part of local index space");

        return ranges[range_id] + (local_gid - range_start_idxs[range_id]);
    }
    if (is == index_space::non_local) {
        GKO_THROW_IF_INVALID(
            local_id < remote_global_idxs_.get_flat().get_size(),
            "Index not part of non-local index space");

        auto host_remote_global_idxs = make_temporary_clone(
            exec_->get_master(), &remote_global_idxs_.get_flat());
        return host_remote_global_idxs->get_const_data()[local_id];
    }
    GKO_NOT_IMPLEMENTED;
}


template <typename LocalIndexType, typename GlobalIndexType>
array<LocalIndexType> index_map<LocalIndexType, GlobalIndexType>::get_local(
    const array<GlobalIndexType>& global_ids, index_space is) const
{
    array<LocalIndexType> local_ids(exec_);

    exec_->run(index_map_kernels::make_get_local(
        partition_.get(), remote_target_ids_, remote_global_idxs_, rank_,
        global_ids, is, local_ids));

    return local_ids;
}


template <typename LocalIndexType, typename GlobalIndexType>
index_map<LocalIndexType, GlobalIndexType>::index_map(
    std::shared_ptr<const Executor> exec, std::shared_ptr<const part_type> part,
    comm_index_type rank, const array<GlobalIndexType>& recv_connections)
    : exec_(std::move(exec)),
      partition_(std::move(part)),
      local_ranges_(exec_),
      rank_(rank),
      remote_target_ids_(exec_),
      remote_local_idxs_(exec_),
      remote_global_idxs_(exec_)
{
    exec_->run(index_map_kernels::make_build_mapping(
        partition_.get(), recv_connections, remote_target_ids_,
        remote_local_idxs_, remote_global_idxs_));

    auto host_part = make_temporary_clone(exec_->get_master(), partition_);
    auto part_ids = host_part->get_part_ids();

    std::vector<size_type> host_local_ranges;
    for (size_type i = 0; i < partition_->get_num_ranges(); ++i) {
        if (part_ids[i] == rank_) {
            host_local_ranges.push_back(i);
        }
    }
    local_ranges_ = array<size_type>(exec_, host_local_ranges.begin(),
                                     host_local_ranges.end());
}


#define GKO_DECLARE_INDEX_MAP(_ltype, _gtype) class index_map<_ltype, _gtype>

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_INDEX_MAP);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
