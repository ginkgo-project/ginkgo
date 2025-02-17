// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_CG_KERNELS_HPP_
#define GKO_CORE_SOLVER_CG_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/base/kernel_declaration.hpp"
#include "ginkgo/core/base/work_estimate.hpp"


namespace gko {
namespace kernels {
namespace cg {


#define GKO_DECLARE_CG_INITIALIZE_KERNEL(_type)                              \
    void initialize(std::shared_ptr<const DefaultExecutor> exec,             \
                    const matrix::Dense<_type>* b, matrix::Dense<_type>* r,  \
                    matrix::Dense<_type>* z, matrix::Dense<_type>* p,        \
                    matrix::Dense<_type>* q, matrix::Dense<_type>* prev_rho, \
                    matrix::Dense<_type>* rho,                               \
                    array<stopping_status>* stop_status)


#define GKO_DECLARE_CG_STEP_1_KERNEL(_type)                             \
    void step_1(std::shared_ptr<const DefaultExecutor> exec,            \
                matrix::Dense<_type>* p, const matrix::Dense<_type>* z, \
                const matrix::Dense<_type>* rho,                        \
                const matrix::Dense<_type>* prev_rho,                   \
                const array<stopping_status>* stop_status)


#define GKO_DECLARE_CG_STEP_2_KERNEL(_type)                                   \
    void step_2(std::shared_ptr<const DefaultExecutor> exec,                  \
                matrix::Dense<_type>* x, matrix::Dense<_type>* r,             \
                const matrix::Dense<_type>* p, const matrix::Dense<_type>* q, \
                const matrix::Dense<_type>* beta,                             \
                const matrix::Dense<_type>* rho,                              \
                const array<stopping_status>* stop_status)


#define GKO_DECLARE_ALL_AS_TEMPLATES             \
    template <typename ValueType>                \
    GKO_DECLARE_CG_INITIALIZE_KERNEL(ValueType); \
    template <typename ValueType>                \
    GKO_DECLARE_CG_STEP_1_KERNEL(ValueType);     \
    template <typename ValueType>                \
    GKO_DECLARE_CG_STEP_2_KERNEL(ValueType)


}  // namespace cg


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(cg, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


namespace work_estimate::cg {


template <typename ValueType>
memory_bound_work_estimate initialize(
    const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* r,
    matrix::Dense<ValueType>* z, matrix::Dense<ValueType>* p,
    matrix::Dense<ValueType>* q, matrix::Dense<ValueType>* prev_rho,
    matrix::Dense<ValueType>* rho, array<stopping_status>* stop_status)
{
    const auto num_values = b->get_size()[0] * b->get_size()[1];
    return memory_bound_work_estimate{num_values * sizeof(ValueType),
                                      6 * num_values * sizeof(ValueType)};
}


template <typename ValueType>
memory_bound_work_estimate step_1(matrix::Dense<ValueType>* p,
                                  const matrix::Dense<ValueType>* z,
                                  const matrix::Dense<ValueType>* rho,
                                  const matrix::Dense<ValueType>* prev_rho,
                                  const array<stopping_status>* stop_status)
{
    const auto num_values = p->get_size()[0] * p->get_size()[1];
    return memory_bound_work_estimate{2 * num_values * sizeof(ValueType),
                                      num_values * sizeof(ValueType)};
}


template <typename ValueType>
memory_bound_work_estimate step_2(matrix::Dense<ValueType>* x,
                                  matrix::Dense<ValueType>* r,
                                  const matrix::Dense<ValueType>* p,
                                  const matrix::Dense<ValueType>* q,
                                  const matrix::Dense<ValueType>* beta,
                                  const matrix::Dense<ValueType>* rho,
                                  const array<stopping_status>* stop_status)
{
    const auto num_values = x->get_size()[0] * x->get_size()[1];
    return memory_bound_work_estimate{4 * num_values * sizeof(ValueType),
                                      2 * num_values * sizeof(ValueType)};
}


}  // namespace work_estimate::cg
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_CG_KERNELS_HPP_
