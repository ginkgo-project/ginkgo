// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_DISTRIBUTED_INDEX_MAP_KERNELS_HPP_
#define GKO_CORE_DISTRIBUTED_INDEX_MAP_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/segmented_array.hpp>
#include <ginkgo/core/distributed/index_map.hpp>
#include <ginkgo/core/distributed/partition.hpp>

#include "core/base/kernel_declaration.hpp"
#include "core/base/segmented_array.hpp"
#include "core/distributed/device_partition.hpp"


namespace gko {
namespace kernels {


/**
 * This kernel creates an index map from a partition and global remote indices.
 *
 * The index map is defined by the output parameters remote_local_idxs,
 * remote_global_idxs, and remote_sizes. After this functions:
 *
 * - remote_global_idxs contains the unique indices from recv_connections,
 *   sorted first by the owning part (as defined in the partition) and then by
 *   the global index
 * - remote_local_idxs contains the indices of remote_global_idxs but mapped
 *   into the local index spaces of their owning parts
 * - remote_sizes contains the number of remote indices (either in
 *   remote_global_idxs, or remote_local_indices) per owning part.
 *
 * The sizes array is used to create segmented arrays for both output index
 * arrays.
 */
#define GKO_DECLARE_INDEX_MAP_BUILD_MAPPING(_ltype, _gtype)                  \
    void build_mapping(                                                      \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        const experimental::distributed::Partition<_ltype, _gtype>* part,    \
        const array<_gtype>& recv_connections,                               \
        array<experimental::distributed::comm_index_type>& part_ids,         \
        array<_ltype>& remote_local_idxs, array<_gtype>& remote_global_idxs, \
        array<int64>& remote_sizes)


/**
 * This kernel maps global indices to local indices.
 *
 * The global indices in remote_global_idxs are mapped into the local index
 * space defined by is. The resulting indices are stored in local_ids.
 * The index map is defined by the input parameters:
 *
 * - partition:  the global partition
 * - remote_target_ids: the owning part ids of each segment of
 *                      remote_global_idxs
 * - remote_global_idxs: the remote global indices, segmented by the owning part
 *   ids
 * - rank: the part id of this process
 *
 * Any global index that is not in the specified local index space is mapped
 * to invalid_index.
 */
#define GKO_DECLARE_INDEX_MAP_MAP_TO_LOCAL(_ltype, _gtype)                     \
    void map_to_local(                                                         \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const experimental::distributed::Partition<_ltype, _gtype>* partition, \
        const array<experimental::distributed::comm_index_type>&               \
            remote_target_ids,                                                 \
        device_segmented_array<const _gtype> remote_global_idxs,               \
        experimental::distributed::comm_index_type rank,                       \
        const array<_gtype>& global_ids,                                       \
        experimental::distributed::index_space is, array<_ltype>& local_ids)


/**
 * This kernels maps local indices to global indices.
 *
 * The relevant input parameter from the index map are:
 *
 * - partition:  the global partition
 * - remote_global_idxs: the remote global indices, segmented by the owning part
 *                       ids
 * - rank: the part id of this process
 *
 * Any local index that is not part of the specified index space is mapped to
 * invalid_index.
 */
#define GKO_DECLARE_INDEX_MAP_MAP_TO_GLOBAL(_ltype, _gtype)      \
    void map_to_global(                                          \
        std::shared_ptr<const DefaultExecutor> exec,             \
        device_partition<const _ltype, const _gtype> partition,  \
        device_segmented_array<const _gtype> remote_global_idxs, \
        experimental::distributed::comm_index_type rank,         \
        const array<_ltype>& local_ids,                          \
        experimental::distributed::index_space is, array<_gtype>& global_ids)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                      \
    template <typename LocalIndexType, typename GlobalIndexType>          \
    GKO_DECLARE_INDEX_MAP_BUILD_MAPPING(LocalIndexType, GlobalIndexType); \
    template <typename LocalIndexType, typename GlobalIndexType>          \
    GKO_DECLARE_INDEX_MAP_MAP_TO_LOCAL(LocalIndexType, GlobalIndexType);  \
    template <typename LocalIndexType, typename GlobalIndexType>          \
    GKO_DECLARE_INDEX_MAP_MAP_TO_GLOBAL(LocalIndexType, GlobalIndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(index_map,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_DISTRIBUTED_INDEX_MAP_KERNELS_HPP_
