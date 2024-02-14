// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/distributed/index_map.hpp>


#include <sys/socket.h>


#include "core/distributed/index_map_kernels.hpp"


namespace gko {
namespace index_map_kernels {


GKO_REGISTER_OPERATION(build_mapping, index_map::build_mapping);
GKO_REGISTER_OPERATION(get_local, index_map::get_local);

}  // namespace index_map_kernels


namespace experimental {
namespace distributed {


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
      rank_(rank),
      remote_target_ids_(exec_),
      remote_local_idxs_(exec_),
      remote_global_idxs_(exec_)
{
    exec_->run(index_map_kernels::make_build_mapping(
        partition_.get(), recv_connections, remote_target_ids_,
        remote_local_idxs_, remote_global_idxs_));
}


#define GKO_DECLARE_INDEX_MAP(_ltype, _gtype) class index_map<_ltype, _gtype>

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_INDEX_MAP);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
