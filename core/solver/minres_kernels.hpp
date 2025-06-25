// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
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


#define GKO_DECLARE_MINRES_INITIALIZE_KERNEL(_type)                            \
    void initialize(std::shared_ptr<const DefaultExecutor> exec,               \
                    const matrix::Dense<_type>* r, matrix::Dense<_type>* z,    \
                    matrix::Dense<_type>* p, matrix::Dense<_type>* p_prev,     \
                    matrix::Dense<_type>* q, matrix::Dense<_type>* q_prev,     \
                    matrix::Dense<_type>* q_tilde, matrix::Dense<_type>* beta, \
                    matrix::Dense<_type>* gamma, matrix::Dense<_type>* delta,  \
                    matrix::Dense<_type>* cos_prev, matrix::Dense<_type>* cos, \
                    matrix::Dense<_type>* sin_prev, matrix::Dense<_type>* sin, \
                    matrix::Dense<_type>* eta_next, matrix::Dense<_type>* eta, \
                    array<stopping_status>* stop_status)


#define GKO_DECLARE_MINRES_STEP_1_KERNEL(_type)                            \
    void step_1(std::shared_ptr<const DefaultExecutor> exec,               \
                matrix::Dense<_type>* alpha, matrix::Dense<_type>* beta,   \
                matrix::Dense<_type>* gamma, matrix::Dense<_type>* delta,  \
                matrix::Dense<_type>* cos_prev, matrix::Dense<_type>* cos, \
                matrix::Dense<_type>* sin_prev, matrix::Dense<_type>* sin, \
                matrix::Dense<_type>* eta, matrix::Dense<_type>* eta_next, \
                matrix::Dense<_type>* tau,                                 \
                const array<stopping_status>* stop_status)

#define GKO_DECLARE_MINRES_STEP_2_KERNEL(_type)                               \
    void step_2(                                                              \
        std::shared_ptr<const DefaultExecutor> exec, matrix::Dense<_type>* x, \
        matrix::Dense<_type>* p, const matrix::Dense<_type>* p_prev,          \
        matrix::Dense<_type>* z, const matrix::Dense<_type>* z_tilde,         \
        matrix::Dense<_type>* q, matrix::Dense<_type>* q_prev,                \
        matrix::Dense<_type>* v, const matrix::Dense<_type>* alpha,           \
        const matrix::Dense<_type>* beta, const matrix::Dense<_type>* gamma,  \
        const matrix::Dense<_type>* delta, const matrix::Dense<_type>* cos,   \
        const matrix::Dense<_type>* eta,                                      \
        const array<stopping_status>* stop_status)


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
