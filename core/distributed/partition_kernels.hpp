// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_DISTRIBUTED_PARTITION_KERNELS_HPP_
#define GKO_CORE_DISTRIBUTED_PARTITION_KERNELS_HPP_


#include <ginkgo/core/distributed/partition.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_PARTITION_COUNT_RANGES                                 \
    void count_ranges(std::shared_ptr<const DefaultExecutor> exec, \
                      const array<comm_index_type>& mapping,       \
                      size_type& num_ranges)

#define GKO_PARTITION_BUILD_FROM_CONTIGUOUS(GlobalIndexType)                  \
    void build_from_contiguous(std::shared_ptr<const DefaultExecutor> exec,   \
                               const array<GlobalIndexType>& ranges,          \
                               const array<comm_index_type>& part_id_mapping, \
                               GlobalIndexType* range_bounds,                 \
                               comm_index_type* part_ids)

#define GKO_PARTITION_BUILD_FROM_MAPPING(GlobalIndexType)                \
    void build_from_mapping(std::shared_ptr<const DefaultExecutor> exec, \
                            const array<comm_index_type>& mapping,       \
                            GlobalIndexType* range_bounds,               \
                            comm_index_type* part_ids)

#define GKO_PARTITION_BUILD_FROM_GLOBAL_SIZE(GlobalIndexType)   \
    void build_ranges_from_global_size(                         \
        std::shared_ptr<const DefaultExecutor> exec,            \
        comm_index_type num_parts, GlobalIndexType global_size, \
        array<GlobalIndexType>& ranges)

#define GKO_DECLARE_PARTITION_BUILD_STARTING_INDICES(LocalIndexType,          \
                                                     GlobalIndexType)         \
    void build_starting_indices(std::shared_ptr<const DefaultExecutor> exec,  \
                                const GlobalIndexType* range_offsets,         \
                                const int* range_parts, size_type num_ranges, \
                                comm_index_type num_parts,                    \
                                comm_index_type& num_empty_parts,             \
                                LocalIndexType* ranks, LocalIndexType* sizes)

#define GKO_DECLARE_PARTITION_IS_ORDERED(LocalIndexType, GlobalIndexType)   \
    void has_ordered_parts(std::shared_ptr<const DefaultExecutor> exec,     \
                           const experimental::distributed::Partition<      \
                               LocalIndexType, GlobalIndexType>* partition, \
                           bool* result)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                    \
    using comm_index_type = experimental::distributed::comm_index_type; \
    GKO_PARTITION_COUNT_RANGES;                                         \
    template <typename GlobalIndexType>                                 \
    GKO_PARTITION_BUILD_FROM_CONTIGUOUS(GlobalIndexType);               \
    template <typename GlobalIndexType>                                 \
    GKO_PARTITION_BUILD_FROM_MAPPING(GlobalIndexType);                  \
    template <typename GlobalIndexType>                                 \
    GKO_PARTITION_BUILD_FROM_GLOBAL_SIZE(GlobalIndexType);              \
    template <typename LocalIndexType, typename GlobalIndexType>        \
    GKO_DECLARE_PARTITION_BUILD_STARTING_INDICES(LocalIndexType,        \
                                                 GlobalIndexType);      \
    template <typename LocalIndexType, typename GlobalIndexType>        \
    GKO_DECLARE_PARTITION_IS_ORDERED(LocalIndexType, GlobalIndexType)

GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(partition,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_DISTRIBUTED_PARTITION_KERNELS_HPP_
