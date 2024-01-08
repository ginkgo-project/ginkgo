// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_COMPONENTS_FILL_ARRAY_KERNELS_HPP_
#define GKO_CORE_COMPONENTS_FILL_ARRAY_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_FILL_ARRAY_KERNEL(ValueType)                 \
    void fill_array(std::shared_ptr<const DefaultExecutor> exec, \
                    ValueType* data, size_type num_entries, ValueType val)

#define GKO_DECLARE_FILL_SEQ_ARRAY_KERNEL(ValueType)                 \
    void fill_seq_array(std::shared_ptr<const DefaultExecutor> exec, \
                        ValueType* data, size_type num_entries)


#define GKO_DECLARE_ALL_AS_TEMPLATES          \
    template <typename ValueType>             \
    GKO_DECLARE_FILL_ARRAY_KERNEL(ValueType); \
    template <typename ValueType>             \
    GKO_DECLARE_FILL_SEQ_ARRAY_KERNEL(ValueType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(components,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_FILL_ARRAY_KERNELS_HPP_
