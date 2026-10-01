// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_MULTIGRID_KERNELS_HPP_
#define GKO_CORE_SOLVER_MULTIGRID_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/device_views.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace multigrid {


#define GKO_DECLARE_MULTIGRID_KCYCLE_STEP_1_KERNEL(ValueType)       \
    void kcycle_step_1(std::shared_ptr<const DefaultExecutor> exec, \
                       matrix::view::dense<const ValueType> alpha,  \
                       matrix::view::dense<const ValueType> rho,    \
                       matrix::view::dense<const ValueType> v,      \
                       matrix::view::dense<ValueType> g,            \
                       matrix::view::dense<ValueType> d,            \
                       matrix::view::dense<ValueType> e)

#define GKO_DECLARE_MULTIGRID_KCYCLE_STEP_2_KERNEL(ValueType)       \
    void kcycle_step_2(std::shared_ptr<const DefaultExecutor> exec, \
                       matrix::view::dense<const ValueType> alpha,  \
                       matrix::view::dense<const ValueType> rho,    \
                       matrix::view::dense<const ValueType> gamma,  \
                       matrix::view::dense<const ValueType> beta,   \
                       matrix::view::dense<const ValueType> zeta,   \
                       matrix::view::dense<const ValueType> d,      \
                       matrix::view::dense<ValueType> e)

#define GKO_DECLARE_MULTIGRID_KCYCLE_CHECK_STOP_KERNEL(ValueType)         \
    void kcycle_check_stop(std::shared_ptr<const DefaultExecutor> exec,   \
                           matrix::view::dense<const ValueType> old_norm, \
                           matrix::view::dense<const ValueType> new_norm, \
                           const ValueType rel_tol, bool& is_stop)


#define GKO_DECLARE_ALL_AS_TEMPLATES                       \
    template <typename ValueType>                          \
    GKO_DECLARE_MULTIGRID_KCYCLE_STEP_1_KERNEL(ValueType); \
    template <typename ValueType>                          \
    GKO_DECLARE_MULTIGRID_KCYCLE_STEP_2_KERNEL(ValueType); \
    template <typename ValueType>                          \
    GKO_DECLARE_MULTIGRID_KCYCLE_CHECK_STOP_KERNEL(ValueType)


}  // namespace multigrid


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(multigrid,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_MULTIGRID_KERNELS_HPP_
