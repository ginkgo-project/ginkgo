// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_CGS_KERNELS_HPP_
#define GKO_CORE_SOLVER_CGS_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace cgs {


#define GKO_DECLARE_CGS_INITIALIZE_KERNEL(_type)                               \
    void initialize(std::shared_ptr<const DefaultExecutor> exec,               \
                    const matrix::Dense<_type>* b, matrix::Dense<_type>* r,    \
                    matrix::Dense<_type>* r_tld, matrix::Dense<_type>* p,      \
                    matrix::Dense<_type>* q, matrix::Dense<_type>* u,          \
                    matrix::Dense<_type>* u_hat, matrix::Dense<_type>* v_hat,  \
                    matrix::Dense<_type>* t, matrix::Dense<_type>* alpha,      \
                    matrix::Dense<_type>* beta, matrix::Dense<_type>* gamma,   \
                    matrix::Dense<_type>* prev_rho, matrix::Dense<_type>* rho, \
                    array<stopping_status>* stop_status)


#define GKO_DECLARE_CGS_STEP_1_KERNEL(_type)                                 \
    void step_1(std::shared_ptr<const DefaultExecutor> exec,                 \
                const matrix::Dense<_type>* r, matrix::Dense<_type>* u,      \
                matrix::Dense<_type>* p, const matrix::Dense<_type>* q,      \
                matrix::Dense<_type>* beta, const matrix::Dense<_type>* rho, \
                const matrix::Dense<_type>* rho_prev,                        \
                const array<stopping_status>* stop_status)


#define GKO_DECLARE_CGS_STEP_2_KERNEL(_type)                                \
    void step_2(std::shared_ptr<const DefaultExecutor> exec,                \
                const matrix::Dense<_type>* u,                              \
                const matrix::Dense<_type>* v_hat, matrix::Dense<_type>* q, \
                matrix::Dense<_type>* t, matrix::Dense<_type>* alpha,       \
                const matrix::Dense<_type>* rho,                            \
                const matrix::Dense<_type>* gamma,                          \
                const array<stopping_status>* stop_status)


#define GKO_DECLARE_CGS_STEP_3_KERNEL(_type)                                \
    void step_3(std::shared_ptr<const DefaultExecutor> exec,                \
                const matrix::Dense<_type>* t,                              \
                const matrix::Dense<_type>* u_hat, matrix::Dense<_type>* r, \
                matrix::Dense<_type>* x, const matrix::Dense<_type>* alpha, \
                const array<stopping_status>* stop_status)


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
