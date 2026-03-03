// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
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


#define GKO_DECLARE_IDR_INITIALIZE_KERNEL(ValueType)                        \
    void initialize(std::shared_ptr<const DefaultExecutor> exec,            \
                    const size_type nrhs, matrix::view::dense<ValueType> m, \
                    matrix::view::dense<ValueType> subspace_vectors,        \
                    bool deterministic, array<stopping_status>* stop_status)


#define GKO_DECLARE_IDR_STEP_1_KERNEL(ValueType)                            \
    void step_1(                                                            \
        std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,  \
        const size_type k, matrix::view::dense<const ValueType> m,          \
        matrix::view::dense<const ValueType> f,                             \
        matrix::view::dense<const ValueType> residual,                      \
        matrix::view::dense<const ValueType> g,                             \
        matrix::view::dense<ValueType> c, matrix::view::dense<ValueType> v, \
        const array<stopping_status>* stop_status)


#define GKO_DECLARE_IDR_STEP_2_KERNEL(ValueType)                            \
    void step_2(std::shared_ptr<const DefaultExecutor> exec,                \
                const size_type nrhs, const size_type k,                    \
                matrix::view::dense<const ValueType> omega,                 \
                matrix::view::dense<const ValueType> preconditioned_vector, \
                matrix::view::dense<const ValueType> c,                     \
                matrix::view::dense<ValueType> u,                           \
                const array<stopping_status>* stop_status)


#define GKO_DECLARE_IDR_STEP_3_KERNEL(ValueType)                              \
    void step_3(                                                              \
        std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,    \
        const size_type k, matrix::view::dense<const ValueType> p,            \
        matrix::view::dense<ValueType> g, matrix::view::dense<ValueType> g_k, \
        matrix::view::dense<ValueType> u, matrix::view::dense<ValueType> m,   \
        matrix::view::dense<ValueType> f,                                     \
        matrix::view::dense<ValueType> alpha,                                 \
        matrix::view::dense<ValueType> residual,                              \
        matrix::view::dense<ValueType> x,                                     \
        const array<stopping_status>* stop_status)


#define GKO_DECLARE_IDR_COMPUTE_OMEGA_KERNEL(ValueType)                     \
    void compute_omega(                                                     \
        std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,  \
        const remove_complex<ValueType> kappa,                              \
        matrix::view::dense<const ValueType> tht,                           \
        matrix::view::dense<const remove_complex<ValueType>> residual_norm, \
        matrix::view::dense<ValueType> omega,                               \
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
