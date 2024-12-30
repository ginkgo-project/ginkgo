// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_COMPONENTS_RANGE_MINIMUM_QUERY_KERNELS_HPP_
#define GKO_CORE_COMPONENTS_RANGE_MINIMUM_QUERY_KERNELS_HPP_


#include "core/components/range_minimum_query.hpp"

#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_RANGE_MINIMUM_QUERY_COMPUTE_LOOKUP_SMALL_KERNEL(IndexType) \
    void compute_lookup_small(std::shared_ptr<const DefaultExecutor> exec,     \
                              const IndexType* values, IndexType size,         \
                              block_argmin_storage_type& block_argmin,         \
                              uint16* block_type)


#define GKO_DECLARE_RANGE_MINIMUM_QUERY_COMPUTE_LOOKUP_LARGE_KERNEL(IndexType) \
    void compute_lookup_large(                                                 \
        std::shared_ptr<const DefaultExecutor> exec, const IndexType* values,  \
        const block_argmin_storage_type& block_argmin, IndexType size,         \
        range_minimum_query_superblocks<IndexType>& superblocks)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                        \
    constexpr int small_block_size = 8;                                     \
    using block_argmin_storage_type =                                       \
        detail::bit_packed_span<uint32, detail::ceil_log2_constexpr(        \
                                            small_block_size)>;             \
    template <typename IndexType>                                           \
    GKO_DECLARE_RANGE_MINIMUM_QUERY_COMPUTE_LOOKUP_SMALL_KERNEL(IndexType); \
    template <typename IndexType>                                           \
    GKO_DECLARE_RANGE_MINIMUM_QUERY_COMPUTE_LOOKUP_LARGE_KERNEL(IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(range_minimum_query,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_COMPONENTS_PRECISION_CONVERSION_KERNELS_HPP_
