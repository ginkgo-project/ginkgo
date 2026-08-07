// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#include <memory>

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/device_views.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/multivector.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL(ValueType)           \
    void simple_apply(std::shared_ptr<const DefaultExecutor> exec, \
                      matrix::view::dense<const ValueType> a,      \
                      matrix::view::dense<const ValueType> b,      \
                      matrix::view::dense<ValueType> c)

#define GKO_DECLARE_DENSE_APPLY_KERNEL(ValueType)           \
    void apply(std::shared_ptr<const DefaultExecutor> exec, \
               matrix::view::dense<const ValueType> alpha,  \
               matrix::view::dense<const ValueType> a,      \
               matrix::view::dense<const ValueType> b,      \
               matrix::view::dense<const ValueType> beta,   \
               matrix::view::dense<ValueType> c)


#define GKO_DECLARE_ALL_AS_TEMPLATES                  \
    template <typename ValueType>                     \
    GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL(ValueType); \
    template <typename ValueType>                     \
    GKO_DECLARE_DENSE_APPLY_KERNEL(ValueType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(dense, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko
