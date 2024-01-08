// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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


#define GKO_DECLARE_BICG_INITIALIZE_KERNEL(_type)                            \
    void initialize(std::shared_ptr<const DefaultExecutor> exec,             \
                    const matrix::Dense<_type>* b, matrix::Dense<_type>* r,  \
                    matrix::Dense<_type>* z, matrix::Dense<_type>* p,        \
                    matrix::Dense<_type>* q, matrix::Dense<_type>* prev_rho, \
                    matrix::Dense<_type>* rho, matrix::Dense<_type>* r2,     \
                    matrix::Dense<_type>* z2, matrix::Dense<_type>* p2,      \
                    matrix::Dense<_type>* q2,                                \
                    array<stopping_status>* stop_status)


#define GKO_DECLARE_BICG_STEP_1_KERNEL(_type)                             \
    void step_1(std::shared_ptr<const DefaultExecutor> exec,              \
                matrix::Dense<_type>* p, const matrix::Dense<_type>* z,   \
                matrix::Dense<_type>* p2, const matrix::Dense<_type>* z2, \
                const matrix::Dense<_type>* rho,                          \
                const matrix::Dense<_type>* prev_rho,                     \
                const array<stopping_status>* stop_status)


#define GKO_DECLARE_BICG_STEP_2_KERNEL(_type)                                  \
    void step_2(std::shared_ptr<const DefaultExecutor> exec,                   \
                matrix::Dense<_type>* x, matrix::Dense<_type>* r,              \
                matrix::Dense<_type>* r2, const matrix::Dense<_type>* p,       \
                const matrix::Dense<_type>* q, const matrix::Dense<_type>* q2, \
                const matrix::Dense<_type>* beta,                              \
                const matrix::Dense<_type>* rho,                               \
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
