// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_GCR_KERNELS_HPP_
#define GKO_CORE_SOLVER_GCR_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace gcr {


#define GKO_DECLARE_GCR_INITIALIZE_KERNEL(_type)                 \
    void initialize(std::shared_ptr<const DefaultExecutor> exec, \
                    const matrix::Dense<_type>* b,               \
                    matrix::Dense<_type>* residual,              \
                    stopping_status* stop_status)


#define GKO_DECLARE_GCR_RESTART_KERNEL(_type)                 \
    void restart(std::shared_ptr<const DefaultExecutor> exec, \
                 const matrix::Dense<_type>* residual,        \
                 const matrix::Dense<_type>* A_residual,      \
                 matrix::Dense<_type>* p_bases,               \
                 matrix::Dense<_type>* Ap_bases, size_type* final_iter_nums)


#define GKO_DECLARE_GCR_STEP_1_KERNEL(_type)                                   \
    void step_1(std::shared_ptr<const DefaultExecutor> exec,                   \
                matrix::Dense<_type>* x, matrix::Dense<_type>* residual,       \
                const matrix::Dense<_type>* p, const matrix::Dense<_type>* Ap, \
                const matrix::Dense<remove_complex<_type>>* Ap_norm,           \
                const matrix::Dense<_type>* rAp,                               \
                const stopping_status* stop_status)


#define GKO_DECLARE_ALL_AS_TEMPLATES              \
    template <typename ValueType>                 \
    GKO_DECLARE_GCR_INITIALIZE_KERNEL(ValueType); \
    template <typename ValueType>                 \
    GKO_DECLARE_GCR_RESTART_KERNEL(ValueType);    \
    template <typename ValueType>                 \
    GKO_DECLARE_GCR_STEP_1_KERNEL(ValueType)


}  // namespace gcr


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(gcr, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_GCR_KERNELS_HPP_
