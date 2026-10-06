// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/nullspace_kernels.hpp"

#include <ginkgo/core/base/math.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Nullspace namespace.
 *
 */
namespace nullspace {


template <typename ValueType>
void compute_coefficients(std::shared_ptr<const DefaultExecutor> exec,
                          matrix::view::dense<const ValueType> x,
                          matrix::view::dense<const ValueType> basis,
                          bool has_constant, remove_complex<ValueType> inv_size,
                          matrix::view::dense<ValueType> coefficients,
                          array<char>& tmp)
{
    const auto num_rhs = static_cast<int64>(x.size[1]);
    const int64 offset = has_constant ? 1 : 0;
    // one reduction over the rows for all coefficients, with the column index
    // coefficient_row * num_rhs + rhs, so x and basis are read only once
    run_kernel_col_reduction_cached(
        exec,
        [] GKO_KERNEL(auto i, auto col, auto x, auto basis, auto num_rhs,
                      auto offset, auto inv_size) {
            const auto l = col / num_rhs;
            const auto j = col % num_rhs;
            return l < offset ? x(i, j) * inv_size
                              : conj(basis(i, l - offset)) * x(i, j);
        },
        GKO_KERNEL_REDUCE_SUM(ValueType), coefficients.values,
        dim<2>{x.size[0], coefficients.size[0] * x.size[1]}, tmp, x, basis,
        num_rhs, offset, inv_size);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_NULLSPACE_COMPUTE_COEFFICIENTS_KERNEL);


template <typename ValueType>
void subtract_projection(std::shared_ptr<const DefaultExecutor> exec,
                         matrix::view::dense<const ValueType> basis,
                         bool has_constant,
                         matrix::view::dense<const ValueType> coefficients,
                         matrix::view::dense<const ValueType> x,
                         matrix::view::dense<ValueType> output)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto basis, auto num_basis, auto offset,
                      auto coefficients, auto x, auto output) {
            auto value = offset > 0 ? coefficients(0, j) : zero(x(i, j));
            for (int64 l = 0; l < num_basis; ++l) {
                value += basis(i, l) * coefficients(l + offset, j);
            }
            output(i, j) = x(i, j) - value;
        },
        x.size, basis, static_cast<int64>(basis.size[1]),
        static_cast<int64>(has_constant ? 1 : 0), coefficients, x, output);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_NULLSPACE_SUBTRACT_PROJECTION_KERNEL);


}  // namespace nullspace
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
