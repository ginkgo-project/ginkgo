// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_PIPE_CG_KERNELS_HPP_
#define GKO_CORE_SOLVER_PIPE_CG_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/device_views.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace pipe_cg {


#define GKO_DECLARE_PIPE_CG_INITIALIZE_1_KERNEL(ValueType)         \
    void initialize_1(std::shared_ptr<const DefaultExecutor> exec, \
                      matrix::view::dense<const ValueType> b,      \
                      matrix::view::dense<ValueType> r,            \
                      matrix::view::dense<ValueType> prev_rho,     \
                      array<stopping_status>& stop_status)

#define GKO_DECLARE_PIPE_CG_INITIALIZE_2_KERNEL(ValueType)                  \
    void initialize_2(                                                      \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        matrix::view::dense<ValueType> p, matrix::view::dense<ValueType> q, \
        matrix::view::dense<ValueType> f, matrix::view::dense<ValueType> g, \
        matrix::view::dense<ValueType> beta,                                \
        matrix::view::dense<const ValueType> z,                             \
        matrix::view::dense<const ValueType> w,                             \
        matrix::view::dense<const ValueType> m,                             \
        matrix::view::dense<const ValueType> n,                             \
        matrix::view::dense<const ValueType> delta)


#define GKO_DECLARE_PIPE_CG_STEP_1_KERNEL(ValueType)                          \
    void step_1(                                                              \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        matrix::view::dense<ValueType> x, matrix::view::dense<ValueType> r,   \
        matrix::view::dense<ValueType> z1, matrix::view::dense<ValueType> z2, \
        matrix::view::dense<ValueType> w,                                     \
        matrix::view::dense<const ValueType> p,                               \
        matrix::view::dense<const ValueType> q,                               \
        matrix::view::dense<const ValueType> f,                               \
        matrix::view::dense<const ValueType> g,                               \
        matrix::view::dense<const ValueType> rho,                             \
        matrix::view::dense<const ValueType> beta,                            \
        const array<stopping_status>& stop_status)


#define GKO_DECLARE_PIPE_CG_STEP_2_KERNEL(ValueType)                           \
    void step_2(                                                               \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        matrix::view::dense<ValueType> beta, matrix::view::dense<ValueType> p, \
        matrix::view::dense<ValueType> q, matrix::view::dense<ValueType> f,    \
        matrix::view::dense<ValueType> g,                                      \
        matrix::view::dense<const ValueType> z,                                \
        matrix::view::dense<const ValueType> w,                                \
        matrix::view::dense<const ValueType> m,                                \
        matrix::view::dense<const ValueType> n,                                \
        matrix::view::dense<const ValueType> prev_rho,                         \
        matrix::view::dense<const ValueType> rho,                              \
        matrix::view::dense<const ValueType> delta,                            \
        const array<stopping_status>& stop_status)


#define GKO_DECLARE_ALL_AS_TEMPLATES                    \
    template <typename ValueType>                       \
    GKO_DECLARE_PIPE_CG_INITIALIZE_1_KERNEL(ValueType); \
    template <typename ValueType>                       \
    GKO_DECLARE_PIPE_CG_INITIALIZE_2_KERNEL(ValueType); \
    template <typename ValueType>                       \
    GKO_DECLARE_PIPE_CG_STEP_1_KERNEL(ValueType);       \
    template <typename ValueType>                       \
    GKO_DECLARE_PIPE_CG_STEP_2_KERNEL(ValueType)


}  // namespace pipe_cg


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(pipe_cg, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_PIPE_CG_KERNELS_HPP_
