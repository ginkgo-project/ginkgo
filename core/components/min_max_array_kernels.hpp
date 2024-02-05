// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_COMPONENTS_MIN_MAX_ARRAY_KERNELS_HPP_
#define GKO_CORE_COMPONENTS_MIN_MAX_ARRAY_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_MAX_ARRAY_KERNEL(IndexType)                 \
    void max_array(std::shared_ptr<const DefaultExecutor> exec, \
                   const array<IndexType>& data, IndexType& result)


#define GKO_DECLARE_MIN_ARRAY_KERNEL(IndexType)                 \
    void min_array(std::shared_ptr<const DefaultExecutor> exec, \
                   const array<IndexType>& data, IndexType& result)


#define GKO_DECLARE_ALL_AS_TEMPLATES         \
    template <typename IndexType>            \
    GKO_DECLARE_MAX_ARRAY_KERNEL(IndexType); \
    template <typename IndexType>            \
    GKO_DECLARE_MIN_ARRAY_KERNEL(IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(components,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_MIN_MAX_ARRAY_KERNELS_HPP_
