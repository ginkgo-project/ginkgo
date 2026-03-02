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
                   matrix::view::dense<const ValueType> alpha,
                   matrix::view::dense<const ValueType> rho,
                   matrix::view::dense<const ValueType> v,
                   matrix::view::dense<ValueType> g,
                   matrix::view::dense<ValueType> d,
                   matrix::view::dense<ValueType> e)
{
    const auto nrows = g.size[0];
    const auto nrhs = g.size[1];
    for (size_type i = 0; i < nrhs; i++) {
        auto temp = alpha(0, i) / rho(0, i);
        for (size_type j = 0; j < nrows; j++) {
            if (is_finite(temp)) {
                g(j, i) -= temp * v(j, i);
                e(j, i) *= temp;
            }
            d(j, i) = e(j, i);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_1_KERNEL);


template <typename ValueType>
void kcycle_step_2(std::shared_ptr<const DefaultExecutor> exec,
                   matrix::view::dense<const ValueType> alpha,
                   matrix::view::dense<const ValueType> rho,
                   matrix::view::dense<const ValueType> gamma,
                   matrix::view::dense<const ValueType> beta,
                   matrix::view::dense<const ValueType> zeta,
                   matrix::view::dense<const ValueType> d,
                   matrix::view::dense<ValueType> e)
{
    const auto nrows = e.size[0];
    const auto nrhs = e.size[1];
    for (size_type i = 0; i < nrhs; i++) {
        auto scalar_d =
            zeta(0, i) / (beta(0, i) - gamma(0, i) * gamma(0, i) / rho(0, i));
        auto scalar_e = one<ValueType>() - gamma(0, i) / alpha(0, i) * scalar_d;
        if (is_finite(scalar_d) && is_finite(scalar_e)) {
            for (size_type j = 0; j < nrows; j++) {
                e(j, i) = scalar_e * e(j, i) + scalar_d * d(j, i);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_2_KERNEL);


template <typename ValueType>
void kcycle_check_stop(std::shared_ptr<const DefaultExecutor> exec,
                       matrix::view::dense<const ValueType> old_norm,
                       matrix::view::dense<const ValueType> new_norm,
                       const ValueType rel_tol, bool& is_stop)
{
    is_stop = true;
    for (size_type i = 0; i < old_norm.size[1]; i++) {
        if (new_norm(0, i) > rel_tol * old_norm(0, i)) {
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
