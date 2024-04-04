// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef INDEX_MAP_KERNELS_HPP
#define INDEX_MAP_KERNELS_HPP


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/segmented_array.hpp>
#include <ginkgo/core/distributed/index_map.hpp>
#include <ginkgo/core/distributed/partition.hpp>


#include "core/base/kernel_declaration.hpp"
#include "core/base/segmented_array.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_INDEX_MAP_BUILD_MAPPING(_ltype, _gtype)                  \
    void build_mapping(                                                      \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        const experimental::distributed::Partition<_ltype, _gtype>* part,    \
        const array<_gtype>& recv_connections,                               \
        array<experimental::distributed::comm_index_type>& part_ids,         \
        array<_ltype>& remote_local_idxs, array<_gtype>& remote_global_idxs, \
        array<int64>& remote_sizes)


#define GKO_DECLARE_INDEX_MAP_GET_LOCAL_FROM_GLOBAL_ARRAY(_ltype, _gtype)      \
    void get_local(                                                            \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const experimental::distributed::Partition<_ltype, _gtype>* partition, \
        const array<experimental::distributed::comm_index_type>&               \
            remote_targed_ids,                                                 \
        device_segmented_array<const _gtype> remote_global_idxs,               \
        experimental::distributed::comm_index_type rank,                       \
        const array<_gtype>& global_ids,                                       \
        experimental::distributed::index_space is, array<_ltype>& local_ids)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                      \
    template <typename LocalIndexType, typename GlobalIndexType>          \
    GKO_DECLARE_INDEX_MAP_BUILD_MAPPING(LocalIndexType, GlobalIndexType); \
    template <typename LocalIndexType, typename GlobalIndexType>          \
    GKO_DECLARE_INDEX_MAP_GET_LOCAL_FROM_GLOBAL_ARRAY(LocalIndexType,     \
                                                      GlobalIndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(index_map,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko

#endif  // INDEX_MAP_KERNELS_HPP
