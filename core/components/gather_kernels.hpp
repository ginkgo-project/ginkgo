// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_COMPONENTS_GATHER_KERNELS_HPP_
#define GKO_CORE_COMPONENTS_GATHER_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


/**
 * result[i] = orig[gather_map[i]].
 *
 * The map may repeat entries and may differ in length from the source, so this
 * is a general gather rather than a permutation.
 */
#define GKO_DECLARE_GATHER_KERNEL(ValueType, IndexType)      \
    void gather(std::shared_ptr<const DefaultExecutor> exec, \
                size_type num_res, const ValueType* orig,    \
                const IndexType* gather_map, ValueType* result)


#define GKO_DECLARE_ALL_AS_TEMPLATES                  \
    template <typename ValueType, typename IndexType> \
    GKO_DECLARE_GATHER_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(components,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_GATHER_KERNELS_HPP_
