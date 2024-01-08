// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/multigrid_kernels.hpp"


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The MULTIGRID solver namespace.
 *
 * @ingroup multigrid
 */
namespace multigrid {


template <typename ValueType>
void kcycle_step_1(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Dense<ValueType>* rho,
                   const matrix::Dense<ValueType>* v,
                   matrix::Dense<ValueType>* g, matrix::Dense<ValueType>* d,
                   matrix::Dense<ValueType>* e)
{
    const auto nrows = g->get_size()[0];
    const auto nrhs = g->get_size()[1];
    for (size_type i = 0; i < nrhs; i++) {
        auto temp = alpha->at(0, i) / rho->at(0, i);
        for (size_type j = 0; j < nrows; j++) {
            if (is_finite(temp)) {
                g->at(j, i) -= temp * v->at(j, i);
                e->at(j, i) *= temp;
            }
            d->at(j, i) = e->at(j, i);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_1_KERNEL);


template <typename ValueType>
void kcycle_step_2(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Dense<ValueType>* rho,
                   const matrix::Dense<ValueType>* gamma,
                   const matrix::Dense<ValueType>* beta,
                   const matrix::Dense<ValueType>* zeta,
                   const matrix::Dense<ValueType>* d,
                   matrix::Dense<ValueType>* e)
{
    const auto nrows = e->get_size()[0];
    const auto nrhs = e->get_size()[1];
    for (size_type i = 0; i < nrhs; i++) {
        auto scalar_d = zeta->at(0, i) /
                        (beta->at(0, i) -
                         gamma->at(0, i) * gamma->at(0, i) / rho->at(0, i));
        auto scalar_e =
            one<ValueType>() - gamma->at(0, i) / alpha->at(0, i) * scalar_d;
        if (is_finite(scalar_d) && is_finite(scalar_e)) {
            for (size_type j = 0; j < nrows; j++) {
                e->at(j, i) = scalar_e * e->at(j, i) + scalar_d * d->at(j, i);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_2_KERNEL);


template <typename ValueType>
void kcycle_check_stop(std::shared_ptr<const DefaultExecutor> exec,
                       const matrix::Dense<ValueType>* old_norm,
                       const matrix::Dense<ValueType>* new_norm,
                       const ValueType rel_tol, bool& is_stop)
{
    is_stop = true;
    for (size_type i = 0; i < old_norm->get_size()[1]; i++) {
        if (new_norm->at(0, i) > rel_tol * old_norm->at(0, i)) {
            is_stop = false;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(
    GKO_DECLARE_MULTIGRID_KCYCLE_CHECK_STOP_KERNEL);


}  // namespace multigrid
}  // namespace reference
}  // namespace kernels
}  // namespace gko
