// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
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


#define GKO_DECLARE_GCR_INITIALIZE_KERNEL(ValueType)             \
    void initialize(std::shared_ptr<const DefaultExecutor> exec, \
                    matrix::view::dense<const ValueType> b,      \
                    matrix::view::dense<ValueType> residual,     \
                    stopping_status* stop_status)


#define GKO_DECLARE_GCR_RESTART_KERNEL(ValueType)                 \
    void restart(std::shared_ptr<const DefaultExecutor> exec,     \
                 matrix::view::dense<const ValueType> residual,   \
                 matrix::view::dense<const ValueType> A_residual, \
                 matrix::view::dense<ValueType> p_bases,          \
                 matrix::view::dense<ValueType> Ap_bases,         \
                 size_type* final_iter_nums)


#define GKO_DECLARE_GCR_STEP_1_KERNEL(ValueType)                              \
    void step_1(std::shared_ptr<const DefaultExecutor> exec,                  \
                matrix::view::dense<ValueType> x,                             \
                matrix::view::dense<ValueType> residual,                      \
                matrix::view::dense<const ValueType> p,                       \
                matrix::view::dense<const ValueType> Ap,                      \
                matrix::view::dense<const remove_complex<ValueType>> Ap_norm, \
                matrix::view::dense<const ValueType> rAp,                     \
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
