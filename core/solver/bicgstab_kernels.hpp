// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_BICGSTAB_KERNELS_HPP_
#define GKO_CORE_SOLVER_BICGSTAB_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/device_views.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace bicgstab {


#define GKO_DECLARE_BICGSTAB_INITIALIZE_KERNEL(ValueType)                    \
    void initialize(                                                         \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        matrix::view::dense<const ValueType> b,                              \
        matrix::view::dense<ValueType> r, matrix::view::dense<ValueType> rr, \
        matrix::view::dense<ValueType> y, matrix::view::dense<ValueType> s,  \
        matrix::view::dense<ValueType> t, matrix::view::dense<ValueType> z,  \
        matrix::view::dense<ValueType> v, matrix::view::dense<ValueType> p,  \
        matrix::view::dense<ValueType> prev_rho,                             \
        matrix::view::dense<ValueType> rho,                                  \
        matrix::view::dense<ValueType> alpha,                                \
        matrix::view::dense<ValueType> beta,                                 \
        matrix::view::dense<ValueType> gamma,                                \
        matrix::view::dense<ValueType> omega,                                \
        array<stopping_status>& stop_status)


#define GKO_DECLARE_BICGSTAB_STEP_1_KERNEL(ValueType)          \
    void step_1(std::shared_ptr<const DefaultExecutor> exec,   \
                matrix::view::dense<const ValueType> r,        \
                matrix::view::dense<ValueType> p,              \
                matrix::view::dense<const ValueType> v,        \
                matrix::view::dense<const ValueType> rho,      \
                matrix::view::dense<const ValueType> prev_rho, \
                matrix::view::dense<const ValueType> alpha,    \
                matrix::view::dense<const ValueType> omega,    \
                const array<stopping_status>& stop_status)


#define GKO_DECLARE_BICGSTAB_STEP_2_KERNEL(ValueType)        \
    void step_2(std::shared_ptr<const DefaultExecutor> exec, \
                matrix::view::dense<const ValueType> r,      \
                matrix::view::dense<ValueType> s,            \
                matrix::view::dense<const ValueType> v,      \
                matrix::view::dense<const ValueType> rho,    \
                matrix::view::dense<ValueType> alpha,        \
                matrix::view::dense<const ValueType> beta,   \
                const array<stopping_status>& stop_status)


#define GKO_DECLARE_BICGSTAB_STEP_3_KERNEL(ValueType)        \
    void step_3(std::shared_ptr<const DefaultExecutor> exec, \
                matrix::view::dense<ValueType> x,            \
                matrix::view::dense<ValueType> r,            \
                matrix::view::dense<const ValueType> s,      \
                matrix::view::dense<const ValueType> t,      \
                matrix::view::dense<const ValueType> y,      \
                matrix::view::dense<const ValueType> z,      \
                matrix::view::dense<const ValueType> alpha,  \
                matrix::view::dense<const ValueType> beta,   \
                matrix::view::dense<const ValueType> gamma,  \
                matrix::view::dense<ValueType> omega,        \
                const array<stopping_status>& stop_status)


#define GKO_DECLARE_BICGSTAB_FINALIZE_KERNEL(ValueType)        \
    void finalize(std::shared_ptr<const DefaultExecutor> exec, \
                  matrix::view::dense<ValueType> x,            \
                  matrix::view::dense<const ValueType> y,      \
                  matrix::view::dense<const ValueType> alpha,  \
                  array<stopping_status>& stop_status)


#define GKO_DECLARE_ALL_AS_TEMPLATES                   \
    template <typename ValueType>                      \
    GKO_DECLARE_BICGSTAB_INITIALIZE_KERNEL(ValueType); \
    template <typename ValueType>                      \
    GKO_DECLARE_BICGSTAB_STEP_1_KERNEL(ValueType);     \
    template <typename ValueType>                      \
    GKO_DECLARE_BICGSTAB_STEP_2_KERNEL(ValueType);     \
    template <typename ValueType>                      \
    GKO_DECLARE_BICGSTAB_STEP_3_KERNEL(ValueType);     \
    template <typename ValueType>                      \
    GKO_DECLARE_BICGSTAB_FINALIZE_KERNEL(ValueType)


}  // namespace bicgstab


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(bicgstab, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_BICGSTAB_KERNELS_HPP_
