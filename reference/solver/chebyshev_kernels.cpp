// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/chebyshev_kernels.hpp"

#include <ginkgo/core/matrix/dense.hpp>

namespace gko {
namespace kernels {
namespace reference {
namespace chebyshev {


template <typename ValueType, typename ScalarType>
void init_update(std::shared_ptr<const DefaultExecutor> exec,
                 const ScalarType alpha,
                 const matrix::Dense<ValueType>* inner_sol,
                 matrix::Dense<ValueType>* update_sol,
                 matrix::Dense<ValueType>* output)
{
    for (size_t row = 0; row < output->get_size()[0]; row++) {
        for (size_t col = 0; col < output->get_size()[1]; col++) {
            const auto inner_val = inner_sol->at(row, col);
            update_sol->at(row, col) = inner_val;
            output->at(row, col) += alpha * inner_val;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(
    GKO_DECLARE_CHEBYSHEV_INIT_UPDATE_KERNEL);


template <typename ValueType, typename ScalarType>
void update(std::shared_ptr<const DefaultExecutor> exec, const ScalarType alpha,
            const ScalarType beta, matrix::Dense<ValueType>* inner_sol,
            matrix::Dense<ValueType>* update_sol,
            matrix::Dense<ValueType>* output)
{
    for (size_t row = 0; row < output->get_size()[0]; row++) {
        for (size_t col = 0; col < output->get_size()[1]; col++) {
            const auto val =
                inner_sol->at(row, col) + beta * update_sol->at(row, col);
            inner_sol->at(row, col) = val;
            update_sol->at(row, col) = val;
            output->at(row, col) += alpha * val;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(
    GKO_DECLARE_CHEBYSHEV_UPDATE_KERNEL);


}  // namespace chebyshev
}  // namespace reference
}  // namespace kernels
}  // namespace gko
