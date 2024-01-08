// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_COMPONENTS_PRECISION_CONVERSION_KERNELS_HPP_
#define GKO_CORE_COMPONENTS_PRECISION_CONVERSION_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_CONVERT_PRECISION_KERNEL(SourceType, TargetType)    \
    void convert_precision(std::shared_ptr<const DefaultExecutor> exec, \
                           size_type size, const SourceType* in,        \
                           TargetType* out)


#define GKO_DECLARE_ALL_AS_TEMPLATES                    \
    template <typename SourceType, typename TargetType> \
    GKO_DECLARE_CONVERT_PRECISION_KERNEL(SourceType, TargetType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(components,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_COMPONENTS_PRECISION_CONVERSION_KERNELS_HPP_
