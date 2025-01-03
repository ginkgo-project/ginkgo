// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/chebyshev_kernels.hpp"

#include <ginkgo/core/matrix/dense.hpp>

#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace chebyshev {


template <typename ValueType, typename ScalarType>
void init_update(std::shared_ptr<const DefaultExecutor> exec,
                 const ScalarType* alpha,
                 const matrix::Dense<ValueType>* inner_sol,
                 matrix::Dense<ValueType>* update_sol,
                 matrix::Dense<ValueType>* output)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto alpha, auto inner_sol,
                      auto update_sol, auto output) {
            const auto inner_val = inner_sol(row, col);
            update_sol(row, col) = val;
            output(row, col) += alpha_val * inner_val;
        },
        output->get_size(), alpha, inner_sol, update_sol, output);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(
    GKO_DECLARE_CHEBYSHEV_INIT_UPDATE_KERNEL);


template <typename ValueType, typename ScalarType>
void update(std::shared_ptr<const DefaultExecutor> exec,
            const ScalarType* alpha, const ScalarType* beta,
            matrix::Dense<ValueType>* inner_sol,
            matrix::Dense<ValueType>* update_sol,
            matrix::Dense<ValueType>* output)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto alpha, auto beta, auto inner_sol,
                      auto update_sol, auto output) {
            const auto val =
                inner_sol(row, col) + beta[0] * update_sol(row, col);
            inner_sol(row, col) = val;
            update_sol(row, col) = val;
            output(row, col) += alpha[0] * val;
        },
        output->get_size(), alpha, beta, inner_sol, update_sol, output);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(
    GKO_DECLARE_CHEBYSHEV_UPDATE_KERNEL);


}  // namespace chebyshev
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
