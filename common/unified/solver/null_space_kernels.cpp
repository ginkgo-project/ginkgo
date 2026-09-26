// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/null_space_kernels.hpp"

#include <ginkgo/core/base/math.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The NullSpace namespace.
 *
 */
namespace null_space {


template <typename ValueType>
void compute_scaled_column_sums(std::shared_ptr<const DefaultExecutor> exec,
                                matrix::view::dense<const ValueType> x,
                                remove_complex<ValueType> scale,
                                matrix::view::dense<ValueType> result,
                                array<char>& tmp)
{
    run_kernel_col_reduction_cached(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto x, auto scale) {
            return x(i, j) * scale;
        },
        GKO_KERNEL_REDUCE_SUM(ValueType), result.values, x.size, tmp, x,
        scale);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_NULL_SPACE_COMPUTE_SCALED_COLUMN_SUMS_KERNEL);


template <typename ValueType>
void remove_constant(std::shared_ptr<const DefaultExecutor> exec,
                     matrix::view::dense<const ValueType> mean,
                     matrix::view::dense<ValueType> x)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto mean, auto x) {
            x(row, col) -= mean[col];
        },
        x.size, mean.values, x);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_NULL_SPACE_REMOVE_CONSTANT_KERNEL);


}  // namespace null_space
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
