// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_COMPONENTS_RANGE_MINIMUM_QUERY_KERNELS_HPP_
#define GKO_CORE_COMPONENTS_RANGE_MINIMUM_QUERY_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_COMPUTE_RMQ_LOOKUP_SMALL_KERNEL(IndexType)                 \
    void compute_rmq_lookup_small(std::shared_ptr<const DefaultExecutor> exec, \
                                  const IndexType* values, IndexType size,     \
                                  IndexType* block_argmin, uint16* block_type)


#define GKO_DECLARE_COMPUTE_RMQ_LOOKUP_LARGE_KERNEL(IndexType)                 \
    void compute_rmq_lookup_large(std::shared_ptr<const DefaultExecutor> exec, \
                                  const IndexType* block_values,               \
                                  IndexType size,                              \
                                  IndexType* superblock_argmin)


#define GKO_DECLARE_ALL_AS_TEMPLATES                        \
    template <typename IndexType>                           \
    GKO_DECLARE_COMPUTE_RMQ_LOOKUP_SMALL_KERNEL(IndexType); \
    template <typename IndexType>                           \
    GKO_DECLARE_COMPUTE_RMQ_LOOKUP_LARGE_KERNEL(IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(components,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_COMPONENTS_PRECISION_CONVERSION_KERNELS_HPP_
