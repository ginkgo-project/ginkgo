// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_BICG_KERNELS_HPP_
#define GKO_CORE_SOLVER_BICG_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace bicg {


#define GKO_DECLARE_BICG_INITIALIZE_KERNEL(ValueType)                          \
    void initialize(                                                           \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        matrix::view::dense<const ValueType> b,                                \
        matrix::view::dense<ValueType> r, matrix::view::dense<ValueType> z,    \
        matrix::view::dense<ValueType> p, matrix::view::dense<ValueType> q,    \
        matrix::view::dense<ValueType> prev_rho,                               \
        matrix::view::dense<ValueType> rho, matrix::view::dense<ValueType> r2, \
        matrix::view::dense<ValueType> z2, matrix::view::dense<ValueType> p2,  \
        matrix::view::dense<ValueType> q2,                                     \
        array<stopping_status>* stop_status)


#define GKO_DECLARE_BICG_STEP_1_KERNEL(ValueType)              \
    void step_1(std::shared_ptr<const DefaultExecutor> exec,   \
                matrix::view::dense<ValueType> p,              \
                matrix::view::dense<const ValueType> z,        \
                matrix::view::dense<ValueType> p2,             \
                matrix::view::dense<const ValueType> z2,       \
                matrix::view::dense<const ValueType> rho,      \
                matrix::view::dense<const ValueType> prev_rho, \
                const array<stopping_status>* stop_status)


#define GKO_DECLARE_BICG_STEP_2_KERNEL(ValueType)            \
    void step_2(std::shared_ptr<const DefaultExecutor> exec, \
                matrix::view::dense<ValueType> x,            \
                matrix::view::dense<ValueType> r,            \
                matrix::view::dense<ValueType> r2,           \
                matrix::view::dense<const ValueType> p,      \
                matrix::view::dense<const ValueType> q,      \
                matrix::view::dense<const ValueType> q2,     \
                matrix::view::dense<const ValueType> beta,   \
                matrix::view::dense<const ValueType> rho,    \
                const array<stopping_status>* stop_status)


#define GKO_DECLARE_ALL_AS_TEMPLATES               \
    template <typename ValueType>                  \
    GKO_DECLARE_BICG_INITIALIZE_KERNEL(ValueType); \
    template <typename ValueType>                  \
    GKO_DECLARE_BICG_STEP_1_KERNEL(ValueType);     \
    template <typename ValueType>                  \
    GKO_DECLARE_BICG_STEP_2_KERNEL(ValueType)


}  // namespace bicg


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(bicg, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_BICG_KERNELS_HPP_
