// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_MULTIGRID_KERNELS_HPP_
#define GKO_CORE_SOLVER_MULTIGRID_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace multigrid {


#define GKO_DECLARE_MULTIGRID_KCYCLE_STEP_1_KERNEL(_type)                      \
    void kcycle_step_1(std::shared_ptr<const DefaultExecutor> exec,            \
                       const matrix::Dense<_type>* alpha,                      \
                       const matrix::Dense<_type>* rho,                        \
                       const matrix::Dense<_type>* v, matrix::Dense<_type>* g, \
                       matrix::Dense<_type>* d, matrix::Dense<_type>* e)

#define GKO_DECLARE_MULTIGRID_KCYCLE_STEP_2_KERNEL(_type)                    \
    void kcycle_step_2(                                                      \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        const matrix::Dense<_type>* alpha, const matrix::Dense<_type>* rho,  \
        const matrix::Dense<_type>* gamma, const matrix::Dense<_type>* beta, \
        const matrix::Dense<_type>* zeta, const matrix::Dense<_type>* d,     \
        matrix::Dense<_type>* e)

#define GKO_DECLARE_MULTIGRID_KCYCLE_CHECK_STOP_KERNEL(_type)           \
    void kcycle_check_stop(std::shared_ptr<const DefaultExecutor> exec, \
                           const matrix::Dense<_type>* old_norm,        \
                           const matrix::Dense<_type>* new_norm,        \
                           const _type rel_tol, bool& is_stop)


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
