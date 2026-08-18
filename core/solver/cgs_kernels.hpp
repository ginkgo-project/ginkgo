// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_CGS_KERNELS_HPP_
#define GKO_CORE_SOLVER_CGS_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/device_views.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace cgs {


#define GKO_DECLARE_CGS_INITIALIZE_KERNEL(ValueType)             \
    void initialize(std::shared_ptr<const DefaultExecutor> exec, \
                    matrix::view::dense<const ValueType> b,      \
                    matrix::view::dense<ValueType> r,            \
                    matrix::view::dense<ValueType> r_tld,        \
                    matrix::view::dense<ValueType> p,            \
                    matrix::view::dense<ValueType> q,            \
                    matrix::view::dense<ValueType> u,            \
                    matrix::view::dense<ValueType> u_hat,        \
                    matrix::view::dense<ValueType> v_hat,        \
                    matrix::view::dense<ValueType> t,            \
                    matrix::view::dense<ValueType> alpha,        \
                    matrix::view::dense<ValueType> beta,         \
                    matrix::view::dense<ValueType> gamma,        \
                    matrix::view::dense<ValueType> prev_rho,     \
                    matrix::view::dense<ValueType> rho,          \
                    array<stopping_status>& stop_status)


#define GKO_DECLARE_CGS_STEP_1_KERNEL(ValueType)               \
    void step_1(std::shared_ptr<const DefaultExecutor> exec,   \
                matrix::view::dense<const ValueType> r,        \
                matrix::view::dense<ValueType> u,              \
                matrix::view::dense<ValueType> p,              \
                matrix::view::dense<const ValueType> q,        \
                matrix::view::dense<ValueType> beta,           \
                matrix::view::dense<const ValueType> rho,      \
                matrix::view::dense<const ValueType> rho_prev, \
                const array<stopping_status>& stop_status)


#define GKO_DECLARE_CGS_STEP_2_KERNEL(ValueType)             \
    void step_2(std::shared_ptr<const DefaultExecutor> exec, \
                matrix::view::dense<const ValueType> u,      \
                matrix::view::dense<const ValueType> v_hat,  \
                matrix::view::dense<ValueType> q,            \
                matrix::view::dense<ValueType> t,            \
                matrix::view::dense<ValueType> alpha,        \
                matrix::view::dense<const ValueType> rho,    \
                matrix::view::dense<const ValueType> gamma,  \
                const array<stopping_status>& stop_status)


#define GKO_DECLARE_CGS_STEP_3_KERNEL(ValueType)             \
    void step_3(std::shared_ptr<const DefaultExecutor> exec, \
                matrix::view::dense<const ValueType> t,      \
                matrix::view::dense<const ValueType> u_hat,  \
                matrix::view::dense<ValueType> r,            \
                matrix::view::dense<ValueType> x,            \
                matrix::view::dense<const ValueType> alpha,  \
                const array<stopping_status>& stop_status)


#define GKO_DECLARE_ALL_AS_TEMPLATES              \
    template <typename ValueType>                 \
    GKO_DECLARE_CGS_INITIALIZE_KERNEL(ValueType); \
    template <typename ValueType>                 \
    GKO_DECLARE_CGS_STEP_1_KERNEL(ValueType);     \
    template <typename ValueType>                 \
    GKO_DECLARE_CGS_STEP_2_KERNEL(ValueType);     \
    template <typename ValueType>                 \
    GKO_DECLARE_CGS_STEP_3_KERNEL(ValueType)


}  // namespace cgs


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(cgs, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_CGS_KERNELS_HPP_
