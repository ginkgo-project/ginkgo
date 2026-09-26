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


/**
 * Computes the coefficients of x with respect to the nullspace: if
 * `has_constant`, row 0 holds the means, sum_i x(i, j) * inv_size, followed by
 * the rows basis^H x. `coefficients` must be contiguous (stride equal to the
 * number of columns of x).
 */
#define GKO_DECLARE_NULL_SPACE_COMPUTE_COEFFICIENTS_KERNEL(ValueType)  \
    void compute_coefficients(                                         \
        std::shared_ptr<const DefaultExecutor> exec,                   \
        matrix::view::dense<const ValueType> x,                        \
        matrix::view::dense<const ValueType> basis, bool has_constant, \
        remove_complex<ValueType> inv_size,                            \
        matrix::view::dense<ValueType> coefficients, array<char>& tmp)


/**
 * Removes the nullspace components given by the coefficients from x:
 * x(i, j) -= coefficients(0, j) (if `has_constant`) + basis(i, :) *
 * coefficients(:, j).
 */
#define GKO_DECLARE_NULL_SPACE_SUBTRACT_PROJECTION_KERNEL(ValueType)   \
    void subtract_projection(                                          \
        std::shared_ptr<const DefaultExecutor> exec,                   \
        matrix::view::dense<const ValueType> basis, bool has_constant, \
        matrix::view::dense<const ValueType> coefficients,             \
        matrix::view::dense<ValueType> x)


#define GKO_DECLARE_ALL_AS_TEMPLATES                               \
    template <typename ValueType>                                  \
    GKO_DECLARE_NULL_SPACE_COMPUTE_COEFFICIENTS_KERNEL(ValueType); \
    template <typename ValueType>                                  \
    GKO_DECLARE_NULL_SPACE_SUBTRACT_PROJECTION_KERNEL(ValueType)


}  // namespace null_space


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(null_space,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_SOLVER_NULL_SPACE_KERNELS_HPP_
