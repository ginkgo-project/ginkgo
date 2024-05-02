// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/index_map_kernels.hpp"


#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace index_map {


template <typename LocalIndexType, typename GlobalIndexType>
void build_mapping(
    std::shared_ptr<const DefaultExecutor> exec,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        part,
    const array<GlobalIndexType>& recv_connections,
    array<experimental::distributed::comm_index_type>& remote_part_ids,
    array<LocalIndexType>& remote_local_idxs,
    array<GlobalIndexType>& remote_global_idxs,
    array<int64>& remote_sizes) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_INDEX_MAP_BUILD_MAPPING);


template <typename LocalIndexType, typename GlobalIndexType>
void map_to_local(
    std::shared_ptr<const DefaultExecutor> exec,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        partition,
    const array<experimental::distributed::comm_index_type>& remote_target_ids,
    device_segmented_array<const GlobalIndexType> remote_global_idxs,
    experimental::distributed::comm_index_type rank,
    const array<GlobalIndexType>& global_ids,
    experimental::distributed::index_space is,
    array<LocalIndexType>& local_ids) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_INDEX_MAP_MAP_TO_LOCAL);


}  // namespace index_map
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
