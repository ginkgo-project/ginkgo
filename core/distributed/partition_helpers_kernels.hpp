// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_PARTITION_HELPERS_KERNELS_HPP
#define GINKGO_PARTITION_HELPERS_KERNELS_HPP


#include <ginkgo/core/base/array.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_PARTITION_HELPERS_SORT_BY_RANGE_START(IndexType) \
    void sort_by_range_start(                                        \
        std::shared_ptr<const DefaultExecutor> exec,                 \
        array<IndexType>& range_start_ends,                          \
        array<experimental::distributed::comm_index_type>& part_ids)


#define GKO_DECLARE_PARTITION_HELPERS_CHECK_CONSECUTIVE_RANGES(IndexType)      \
    void check_consecutive_ranges(std::shared_ptr<const DefaultExecutor> exec, \
                                  const array<IndexType>& range_start_ends,    \
                                  bool& result)


#define GKO_DECLARE_PARTITION_HELPERS_COMPRESS_RANGES(IndexType)      \
    void compress_ranges(std::shared_ptr<const DefaultExecutor> exec, \
                         const array<IndexType>& range_start_ends,    \
                         array<IndexType>& range_offsets)


#define GKO_DECLARE_ALL_AS_TEMPLATES(_export_macro)                       \
    template <typename GlobalIndexType>                                   \
    _export_macro GKO_DECLARE_PARTITION_HELPERS_SORT_BY_RANGE_START(      \
        GlobalIndexType);                                                 \
    template <typename GlobalIndexType>                                   \
    _export_macro GKO_DECLARE_PARTITION_HELPERS_CHECK_CONSECUTIVE_RANGES( \
        GlobalIndexType);                                                 \
    template <typename GlobalIndexType>                                   \
    _export_macro GKO_DECLARE_PARTITION_HELPERS_COMPRESS_RANGES(          \
        GlobalIndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(partition_helpers,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GINKGO_PARTITION_HELPERS_KERNELS_HPP
