// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef INDEX_MAP_KERNELS_HPP
#define INDEX_MAP_KERNELS_HPP


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/collection.hpp>
#include <ginkgo/core/distributed/partition.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_INDEX_MAP_BUILD_MAPPING(_ltype, _gtype)               \
    void build_mapping(                                                   \
        std::shared_ptr<const DefaultExecutor> exec,                      \
        const experimental::distributed::Partition<_ltype, _gtype>* part, \
        const array<_gtype>& recv_connections,                            \
        array<experimental::distributed::comm_index_type>& part_ids,      \
        collection::array<_ltype>& remote_local_idxs,                     \
        collection::array<_gtype>& remote_global_idxs)


#define GKO_DECLARE_ALL_AS_TEMPLATES                             \
    template <typename LocalIndexType, typename GlobalIndexType> \
    GKO_DECLARE_INDEX_MAP_BUILD_MAPPING(LocalIndexType, GlobalIndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(index_map,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko

#endif  // INDEX_MAP_KERNELS_HPP
