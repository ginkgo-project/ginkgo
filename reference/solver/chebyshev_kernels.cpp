// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/chebyshev_kernels.hpp"

#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/chebyshev.hpp>

namespace gko {
namespace kernels {
namespace reference {
namespace chebyshev {


template <typename ValueType>
void init_update(std::shared_ptr<const DefaultExecutor> exec,
                 const solver::detail::coeff_type<ValueType> alpha,
                 const matrix::Dense<ValueType>* inner_sol,
                 matrix::Dense<ValueType>* update_sol,
                 matrix::Dense<ValueType>* output)
{
    // the coeff_type always be the highest precision, so we need
    // to cast the others from ValueType to this precision.
    using arithmetic_type = solver::detail::coeff_type<ValueType>;
    for (size_t row = 0; row < output->get_size()[0]; row++) {
        for (size_t col = 0; col < output->get_size()[1]; col++) {
            const auto inner_val =
                static_cast<arithmetic_type>(inner_sol->at(row, col));
            update_sol->at(row, col) = static_cast<ValueType>(inner_val);
            output->at(row, col) = static_cast<ValueType>(
                static_cast<arithmetic_type>(output->at(row, col)) +
                alpha * inner_val);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CHEBYSHEV_INIT_UPDATE_KERNEL);


template <typename ValueType>
void update(std::shared_ptr<const DefaultExecutor> exec,
            const solver::detail::coeff_type<ValueType> alpha,
            const solver::detail::coeff_type<ValueType> beta,
            matrix::Dense<ValueType>* inner_sol,
            matrix::Dense<ValueType>* update_sol,
            matrix::Dense<ValueType>* output)
{
    // the coeff_type always be the highest precision, so we need
    // to cast the others from ValueType to this precision.
    using arithmetic_type = solver::detail::coeff_type<ValueType>;
    for (size_t row = 0; row < output->get_size()[0]; row++) {
        for (size_t col = 0; col < output->get_size()[1]; col++) {
            const auto val =
                static_cast<arithmetic_type>(inner_sol->at(row, col)) +
                beta * static_cast<arithmetic_type>(update_sol->at(row, col));
            inner_sol->at(row, col) = static_cast<ValueType>(val);
            update_sol->at(row, col) = static_cast<ValueType>(val);
            output->at(row, col) = static_cast<ValueType>(
                static_cast<arithmetic_type>(output->at(row, col)) +
                alpha * val);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CHEBYSHEV_UPDATE_KERNEL);


}  // namespace chebyshev
}  // namespace reference
}  // namespace kernels
}  // namespace gko
