// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_NULL_SPACE_KERNELS_HPP_
#define GKO_CORE_SOLVER_NULL_SPACE_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace null_space {


#define GKO_DECLARE_NULL_SPACE_COMPUTE_SCALED_COLUMN_SUMS_KERNEL(ValueType) \
    void compute_scaled_column_sums(                                        \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        matrix::view::dense<const ValueType> x,                             \
        remove_complex<ValueType> scale,                                    \
        matrix::view::dense<ValueType> result, array<char>& tmp)


#define GKO_DECLARE_NULL_SPACE_REMOVE_CONSTANT_KERNEL(ValueType)      \
    void remove_constant(std::shared_ptr<const DefaultExecutor> exec, \
                         matrix::view::dense<const ValueType> mean,   \
                         matrix::view::dense<ValueType> x)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                     \
    template <typename ValueType>                                        \
    GKO_DECLARE_NULL_SPACE_COMPUTE_SCALED_COLUMN_SUMS_KERNEL(ValueType); \
    template <typename ValueType>                                        \
    GKO_DECLARE_NULL_SPACE_REMOVE_CONSTANT_KERNEL(ValueType)


}  // namespace null_space


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(null_space,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_SOLVER_NULL_SPACE_KERNELS_HPP_
