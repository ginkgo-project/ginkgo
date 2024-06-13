// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_IDR_KERNELS_HPP_
#define GKO_CORE_SOLVER_IDR_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace idr {


#define GKO_DECLARE_IDR_INITIALIZE_KERNEL(_type)                   \
    void initialize(std::shared_ptr<const DefaultExecutor> exec,   \
                    const size_type nrhs, matrix::Dense<_type>* m, \
                    matrix::Dense<_type>* subspace_vectors,        \
                    bool deterministic, array<stopping_status>* stop_status)


#define GKO_DECLARE_IDR_STEP_1_KERNEL(_type)                                 \
    void step_1(                                                             \
        std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,   \
        const size_type k, const matrix::Dense<_type>* m,                    \
        const matrix::Dense<_type>* f, const matrix::Dense<_type>* residual, \
        const matrix::Dense<_type>* g, matrix::Dense<_type>* c,              \
        matrix::Dense<_type>* v, const array<stopping_status>* stop_status)


#define GKO_DECLARE_IDR_STEP_2_KERNEL(_type)                            \
    void step_2(std::shared_ptr<const DefaultExecutor> exec,            \
                const size_type nrhs, const size_type k,                \
                const matrix::Dense<_type>* omega,                      \
                const matrix::Dense<_type>* preconditioned_vector,      \
                const matrix::Dense<_type>* c, matrix::Dense<_type>* u, \
                const array<stopping_status>* stop_status)


#define GKO_DECLARE_IDR_STEP_3_KERNEL(_type)                                 \
    void step_3(std::shared_ptr<const DefaultExecutor> exec,                 \
                const size_type nrhs, const size_type k,                     \
                const matrix::Dense<_type>* p, matrix::Dense<_type>* g,      \
                matrix::Dense<_type>* g_k, matrix::Dense<_type>* u,          \
                matrix::Dense<_type>* m, matrix::Dense<_type>* f,            \
                matrix::Dense<_type>* alpha, matrix::Dense<_type>* residual, \
                matrix::Dense<_type>* x,                                     \
                const array<stopping_status>* stop_status)


#define GKO_DECLARE_IDR_COMPUTE_OMEGA_KERNEL(_type)                         \
    void compute_omega(                                                     \
        std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,  \
        const remove_complex<_type> kappa, const matrix::Dense<_type>* tht, \
        const matrix::Dense<remove_complex<_type>>* residual_norm,          \
        matrix::Dense<_type>* omega,                                        \
        const array<stopping_status>* stop_status)


#define GKO_DECLARE_ALL_AS_TEMPLATES              \
    template <typename ValueType>                 \
    GKO_DECLARE_IDR_INITIALIZE_KERNEL(ValueType); \
    template <typename ValueType>                 \
    GKO_DECLARE_IDR_STEP_1_KERNEL(ValueType);     \
    template <typename ValueType>                 \
    GKO_DECLARE_IDR_STEP_2_KERNEL(ValueType);     \
    template <typename ValueType>                 \
    GKO_DECLARE_IDR_STEP_3_KERNEL(ValueType);     \
    template <typename ValueType>                 \
    GKO_DECLARE_IDR_COMPUTE_OMEGA_KERNEL(ValueType)


}  // namespace idr


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(idr, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_IDR_KERNELS_HPP_
