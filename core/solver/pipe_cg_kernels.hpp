// SPDX-FileCopyrightText: 2025 The Ginkgo authors
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


#define GKO_DECLARE_PIPE_CG_INITIALIZE_1_KERNEL(_type)                        \
    void initialize_1(std::shared_ptr<const DefaultExecutor> exec,            \
                      const matrix::Dense<_type>* b, matrix::Dense<_type>* r, \
                      matrix::Dense<_type>* prev_rho,                         \
                      array<stopping_status>* stop_status)

#define GKO_DECLARE_PIPE_CG_INITIALIZE_2_KERNEL(_type)                        \
    void initialize_2(                                                        \
        std::shared_ptr<const DefaultExecutor> exec, matrix::Dense<_type>* p, \
        matrix::Dense<_type>* q, matrix::Dense<_type>* f,                     \
        matrix::Dense<_type>* g, matrix::Dense<_type>* beta,                  \
        const matrix::Dense<_type>* z, const matrix::Dense<_type>* w,         \
        const matrix::Dense<_type>* m, const matrix::Dense<_type>* n,         \
        const matrix::Dense<_type>* delta)


#define GKO_DECLARE_PIPE_CG_STEP_1_KERNEL(_type)                              \
    void step_1(                                                              \
        std::shared_ptr<const DefaultExecutor> exec, matrix::Dense<_type>* x, \
        matrix::Dense<_type>* r, matrix::Dense<_type>* z1,                    \
        matrix::Dense<_type>* z2, matrix::Dense<_type>* w,                    \
        const matrix::Dense<_type>* p, const matrix::Dense<_type>* q,         \
        const matrix::Dense<_type>* f, const matrix::Dense<_type>* g,         \
        const matrix::Dense<_type>* rho, const matrix::Dense<_type>* beta,    \
        const array<stopping_status>* stop_status)


#define GKO_DECLARE_PIPE_CG_STEP_2_KERNEL(_type)                             \
    void step_2(                                                             \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        matrix::Dense<_type>* beta, matrix::Dense<_type>* p,                 \
        matrix::Dense<_type>* q, matrix::Dense<_type>* f,                    \
        matrix::Dense<_type>* g, const matrix::Dense<_type>* z,              \
        const matrix::Dense<_type>* w, const matrix::Dense<_type>* m,        \
        const matrix::Dense<_type>* n, const matrix::Dense<_type>* prev_rho, \
        const matrix::Dense<_type>* rho, const matrix::Dense<_type>* delta,  \
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
