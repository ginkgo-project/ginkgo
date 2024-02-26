// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_COMPONENTS_REDUCE_ARRAY_KERNELS_HPP_
#define GKO_CORE_COMPONENTS_REDUCE_ARRAY_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_REDUCE_ADD_ARRAY_KERNEL(ValueType)                 \
    void reduce_add_array(std::shared_ptr<const DefaultExecutor> exec, \
                          const array<ValueType>& data,                \
                          array<ValueType>& result)


#define GKO_DECLARE_ALL_AS_TEMPLATES \
    template <typename ValueType>    \
    GKO_DECLARE_REDUCE_ADD_ARRAY_KERNEL(ValueType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(components,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_REDUCE_ARRAY_KERNELS_HPP_
