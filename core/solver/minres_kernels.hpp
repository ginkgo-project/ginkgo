// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_MINRES_KERNELS_HPP_
#define GKO_CORE_SOLVER_MINRES_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace minres {


#define GKO_DECLARE_MINRES_INITIALIZE_KERNEL(ValueType)          \
    void initialize(std::shared_ptr<const DefaultExecutor> exec, \
                    matrix::view::dense<const ValueType> r,      \
                    matrix::view::dense<ValueType> z,            \
                    matrix::view::dense<ValueType> p,            \
                    matrix::view::dense<ValueType> p_prev,       \
                    matrix::view::dense<ValueType> q,            \
                    matrix::view::dense<ValueType> q_prev,       \
                    matrix::view::dense<ValueType> q_tilde,      \
                    matrix::view::dense<ValueType> beta,         \
                    matrix::view::dense<ValueType> gamma,        \
                    matrix::view::dense<ValueType> delta,        \
                    matrix::view::dense<ValueType> cos_prev,     \
                    matrix::view::dense<ValueType> cos,          \
                    matrix::view::dense<ValueType> sin_prev,     \
                    matrix::view::dense<ValueType> sin,          \
                    matrix::view::dense<ValueType> eta_next,     \
                    matrix::view::dense<ValueType> eta,          \
                    array<stopping_status>& stop_status)


#define GKO_DECLARE_MINRES_STEP_1_KERNEL(ValueType)          \
    void step_1(std::shared_ptr<const DefaultExecutor> exec, \
                matrix::view::dense<ValueType> alpha,        \
                matrix::view::dense<ValueType> beta,         \
                matrix::view::dense<ValueType> gamma,        \
                matrix::view::dense<ValueType> delta,        \
                matrix::view::dense<ValueType> cos_prev,     \
                matrix::view::dense<ValueType> cos,          \
                matrix::view::dense<ValueType> sin_prev,     \
                matrix::view::dense<ValueType> sin,          \
                matrix::view::dense<ValueType> eta,          \
                matrix::view::dense<ValueType> eta_next,     \
                matrix::view::dense<ValueType> tau,          \
                const array<stopping_status>& stop_status)

#define GKO_DECLARE_MINRES_STEP_2_KERNEL(ValueType)           \
    void step_2(std::shared_ptr<const DefaultExecutor> exec,  \
                matrix::view::dense<ValueType> x,             \
                matrix::view::dense<ValueType> p,             \
                matrix::view::dense<const ValueType> p_prev,  \
                matrix::view::dense<ValueType> z,             \
                matrix::view::dense<const ValueType> z_tilde, \
                matrix::view::dense<ValueType> q,             \
                matrix::view::dense<ValueType> q_prev,        \
                matrix::view::dense<ValueType> v,             \
                matrix::view::dense<const ValueType> alpha,   \
                matrix::view::dense<const ValueType> beta,    \
                matrix::view::dense<const ValueType> gamma,   \
                matrix::view::dense<const ValueType> delta,   \
                matrix::view::dense<const ValueType> cos,     \
                matrix::view::dense<const ValueType> eta,     \
                const array<stopping_status>& stop_status)


#define GKO_DECLARE_ALL_AS_TEMPLATES                 \
    template <typename ValueType>                    \
    GKO_DECLARE_MINRES_INITIALIZE_KERNEL(ValueType); \
    template <typename ValueType>                    \
    GKO_DECLARE_MINRES_STEP_1_KERNEL(ValueType);     \
    template <typename ValueType>                    \
    GKO_DECLARE_MINRES_STEP_2_KERNEL(ValueType)


}  // namespace minres


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(minres, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_MINRES_KERNELS_HPP_
