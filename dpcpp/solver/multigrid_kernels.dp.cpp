// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Dense<ValueType>* rho,
                   const matrix::Dense<ValueType>* v,
                   matrix::Dense<ValueType>* g, matrix::Dense<ValueType>* d,
                   matrix::Dense<ValueType>* e) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_1_KERNEL);


template <typename ValueType>
void kcycle_step_2(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Dense<ValueType>* rho,
                   const matrix::Dense<ValueType>* gamma,
                   const matrix::Dense<ValueType>* beta,
                   const matrix::Dense<ValueType>* zeta,
                   const matrix::Dense<ValueType>* d,
                   matrix::Dense<ValueType>* e) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_2_KERNEL);


template <typename ValueType>
void kcycle_check_stop(std::shared_ptr<const DefaultExecutor> exec,
                       const matrix::Dense<ValueType>* old_norm,
                       const matrix::Dense<ValueType>* new_norm,
                       const ValueType rel_tol,
                       bool& is_stop) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(
    GKO_DECLARE_MULTIGRID_KCYCLE_CHECK_STOP_KERNEL);


}  // namespace multigrid
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
