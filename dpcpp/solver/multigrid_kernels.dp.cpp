// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/multigrid_kernels.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>

#include "core/components/fill_array_kernels.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
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
                   matrix::view::dense<ValueType> e) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_1_KERNEL);


template <typename ValueType>
void kcycle_step_2(std::shared_ptr<const DefaultExecutor> exec,
                   matrix::view::dense<const ValueType> alpha,
                   matrix::view::dense<const ValueType> rho,
                   matrix::view::dense<const ValueType> gamma,
                   matrix::view::dense<const ValueType> beta,
                   matrix::view::dense<const ValueType> zeta,
                   matrix::view::dense<const ValueType> d,
                   matrix::view::dense<ValueType> e) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_2_KERNEL);


template <typename ValueType>
void kcycle_check_stop(std::shared_ptr<const DefaultExecutor> exec,
                       matrix::view::dense<const ValueType> old_norm,
                       matrix::view::dense<const ValueType> new_norm,
                       const ValueType rel_tol,
                       bool& is_stop) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(
    GKO_DECLARE_MULTIGRID_KCYCLE_CHECK_STOP_KERNEL);


}  // namespace multigrid
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
