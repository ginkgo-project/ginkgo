// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/chebyshev_kernels.hpp"

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "core/base/mixed_precision_types.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace chebyshev {


template <typename ValueType, typename ScalarType>
void init_update(std::shared_ptr<const DefaultExecutor> exec,
                 const ScalarType alpha,
                 const matrix::Dense<ValueType>* inner_sol,
                 matrix::Dense<ValueType>* update_sol,
                 matrix::Dense<ValueType>* output)
{
    using type = device_type<highest_precision<ValueType, ScalarType>>;
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto alpha, auto inner_sol,
                      auto update_sol, auto output) {
            const auto inner_val = static_cast<type>(inner_sol(row, col));
            update_sol(row, col) =
                static_cast<device_type<ValueType>>(inner_val);
            output(row, col) = static_cast<device_type<ValueType>>(
                static_cast<type>(output(row, col)) +
                static_cast<type>(alpha) * inner_val);
        },
        output->get_size(), alpha, inner_sol, update_sol, output);
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2(
    GKO_DECLARE_CHEBYSHEV_INIT_UPDATE_KERNEL);


template <typename ValueType, typename ScalarType>
void update(std::shared_ptr<const DefaultExecutor> exec, const ScalarType alpha,
            const ScalarType beta, matrix::Dense<ValueType>* inner_sol,
            matrix::Dense<ValueType>* update_sol,
            matrix::Dense<ValueType>* output)
{
    using type = device_type<highest_precision<ValueType, ScalarType>>;
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto alpha, auto beta, auto inner_sol,
                      auto update_sol, auto output) {
            const auto val = static_cast<type>(inner_sol(row, col)) +
                             static_cast<type>(beta) *
                                 static_cast<type>(update_sol(row, col));
            inner_sol(row, col) = static_cast<device_type<ValueType>>(val);
            update_sol(row, col) = static_cast<device_type<ValueType>>(val);
            output(row, col) = static_cast<device_type<ValueType>>(
                static_cast<type>(output(row, col)) +
                static_cast<type>(alpha) * val);
        },
        output->get_size(), alpha, beta, inner_sol, update_sol, output);
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_TYPE_2(
    GKO_DECLARE_CHEBYSHEV_UPDATE_KERNEL);


}  // namespace chebyshev
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
