// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_PIPE_CG_KERNELS_HPP_
#define GKO_CORE_SOLVER_PIPE_CG_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace pipe_cg {


#define GKO_DECLARE_PIPE_CG_INITIALIZE_1_KERNEL(_type)             \
    void initialize_1(std::shared_ptr<const DefaultExecutor> exec, \
                      matrix::view::dense<const _type> b,          \
                      matrix::view::dense<_type> r,                \
                      matrix::view::dense<_type> prev_rho,         \
                      array<stopping_status>* stop_status)

#define GKO_DECLARE_PIPE_CG_INITIALIZE_2_KERNEL(_type)                       \
    void initialize_2(                                                       \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        matrix::view::dense<_type> p, matrix::view::dense<_type> q,          \
        matrix::view::dense<_type> f, matrix::view::dense<_type> g,          \
        matrix::view::dense<_type> beta, matrix::view::dense<const _type> z, \
        matrix::view::dense<const _type> w,                                  \
        matrix::view::dense<const _type> m,                                  \
        matrix::view::dense<const _type> n,                                  \
        matrix::view::dense<const _type> delta)


#define GKO_DECLARE_PIPE_CG_STEP_1_KERNEL(_type)                              \
    void step_1(std::shared_ptr<const DefaultExecutor> exec,                  \
                matrix::view::dense<_type> x, matrix::view::dense<_type> r,   \
                matrix::view::dense<_type> z1, matrix::view::dense<_type> z2, \
                matrix::view::dense<_type> w,                                 \
                matrix::view::dense<const _type> p,                           \
                matrix::view::dense<const _type> q,                           \
                matrix::view::dense<const _type> f,                           \
                matrix::view::dense<const _type> g,                           \
                matrix::view::dense<const _type> rho,                         \
                matrix::view::dense<const _type> beta,                        \
                const array<stopping_status>* stop_status)


#define GKO_DECLARE_PIPE_CG_STEP_2_KERNEL(_type)                               \
    void step_2(std::shared_ptr<const DefaultExecutor> exec,                   \
                matrix::view::dense<_type> beta, matrix::view::dense<_type> p, \
                matrix::view::dense<_type> q, matrix::view::dense<_type> f,    \
                matrix::view::dense<_type> g,                                  \
                matrix::view::dense<const _type> z,                            \
                matrix::view::dense<const _type> w,                            \
                matrix::view::dense<const _type> m,                            \
                matrix::view::dense<const _type> n,                            \
                matrix::view::dense<const _type> prev_rho,                     \
                matrix::view::dense<const _type> rho,                          \
                matrix::view::dense<const _type> delta,                        \
                const array<stopping_status>* stop_status)


#define GKO_DECLARE_ALL_AS_TEMPLATES                \
    template <typename _type>                       \
    GKO_DECLARE_PIPE_CG_INITIALIZE_1_KERNEL(_type); \
    template <typename _type>                       \
    GKO_DECLARE_PIPE_CG_INITIALIZE_2_KERNEL(_type); \
    template <typename _type>                       \
    GKO_DECLARE_PIPE_CG_STEP_1_KERNEL(_type);       \
    template <typename _type>                       \
    GKO_DECLARE_PIPE_CG_STEP_2_KERNEL(_type)


}  // namespace pipe_cg


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(pipe_cg, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_PIPE_CG_KERNELS_HPP_
