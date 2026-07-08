// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_BICGSTAB_KERNELS_HPP_
#define GKO_CORE_SOLVER_BICGSTAB_KERNELS_HPP_


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
namespace bicgstab {


#define GKO_DECLARE_BICGSTAB_INITIALIZE_KERNEL(_type)                        \
    void initialize(std::shared_ptr<const DefaultExecutor> exec,             \
                    const matrix::Dense<_type>* b, matrix::Dense<_type>* r,  \
                    matrix::Dense<_type>* rr, matrix::Dense<_type>* y,       \
                    matrix::Dense<_type>* s, matrix::Dense<_type>* t,        \
                    matrix::Dense<_type>* z, matrix::Dense<_type>* v,        \
                    matrix::Dense<_type>* p, matrix::Dense<_type>* prev_rho, \
                    matrix::Dense<_type>* rho, matrix::Dense<_type>* alpha,  \
                    matrix::Dense<_type>* beta, matrix::Dense<_type>* gamma, \
                    matrix::Dense<_type>* omega,                             \
                    array<stopping_status>* stop_status)


#define GKO_DECLARE_BICGSTAB_STEP_1_KERNEL(_type)                             \
    void step_1(                                                              \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const matrix::Dense<_type>* r, matrix::Dense<_type>* p,               \
        const matrix::Dense<_type>* v, const matrix::Dense<_type>* rho,       \
        const matrix::Dense<_type>* prev_rho,                                 \
        const matrix::Dense<_type>* alpha, const matrix::Dense<_type>* omega, \
        const array<stopping_status>* stop_status)


#define GKO_DECLARE_BICGSTAB_STEP_2_KERNEL(_type)                             \
    void step_2(std::shared_ptr<const DefaultExecutor> exec,                  \
                const matrix::Dense<_type>* r, matrix::Dense<_type>* s,       \
                const matrix::Dense<_type>* v,                                \
                const matrix::Dense<_type>* rho, matrix::Dense<_type>* alpha, \
                const matrix::Dense<_type>* beta,                             \
                const array<stopping_status>* stop_status)


#define GKO_DECLARE_BICGSTAB_STEP_3_KERNEL(_type)                             \
    void step_3(                                                              \
        std::shared_ptr<const DefaultExecutor> exec, matrix::Dense<_type>* x, \
        matrix::Dense<_type>* r, const matrix::Dense<_type>* s,               \
        const matrix::Dense<_type>* t, const matrix::Dense<_type>* y,         \
        const matrix::Dense<_type>* z, const matrix::Dense<_type>* alpha,     \
        const matrix::Dense<_type>* beta, const matrix::Dense<_type>* gamma,  \
        matrix::Dense<_type>* omega,                                          \
        const array<stopping_status>* stop_status)


#define GKO_DECLARE_BICGSTAB_FINALIZE_KERNEL(_type)                       \
    void finalize(std::shared_ptr<const DefaultExecutor> exec,            \
                  matrix::Dense<_type>* x, const matrix::Dense<_type>* y, \
                  const matrix::Dense<_type>* alpha,                      \
                  array<stopping_status>* stop_status)


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


namespace work_estimate::bicgstab {


template <typename ValueType>
memory_bound_work_estimate initialize(
    const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* r,
    matrix::Dense<ValueType>* rr, matrix::Dense<ValueType>* y,
    matrix::Dense<ValueType>* s, matrix::Dense<ValueType>* t,
    matrix::Dense<ValueType>* z, matrix::Dense<ValueType>* v,
    matrix::Dense<ValueType>* p, matrix::Dense<ValueType>* prev_rho,
    matrix::Dense<ValueType>* rho, matrix::Dense<ValueType>* alpha,
    matrix::Dense<ValueType>* beta, matrix::Dense<ValueType>* gamma,
    matrix::Dense<ValueType>* omega, array<stopping_status>* stop_status)
{
    const auto num_rows = b->get_size()[0] * b->get_size()[1];
    return memory_bound_work_estimate{num_rows * sizeof(ValueType),
                                      14 * num_rows * sizeof(ValueType)};
}


template <typename ValueType>
memory_bound_work_estimate step_1(const matrix::Dense<ValueType>* r,
                                  matrix::Dense<ValueType>* p,
                                  const matrix::Dense<ValueType>* v,
                                  const matrix::Dense<ValueType>* rho,
                                  const matrix::Dense<ValueType>* prev_rho,
                                  const matrix::Dense<ValueType>* alpha,
                                  const matrix::Dense<ValueType>* omega,
                                  const array<stopping_status>* stop_status)
{
    const auto num_rows = r->get_size()[0] * r->get_size()[1];
    return memory_bound_work_estimate{3 * num_rows * sizeof(ValueType),
                                      num_rows * sizeof(ValueType)};
}


template <typename ValueType>
memory_bound_work_estimate step_2(const matrix::Dense<ValueType>* r,
                                  matrix::Dense<ValueType>* s,
                                  const matrix::Dense<ValueType>* v,
                                  const matrix::Dense<ValueType>* rho,
                                  matrix::Dense<ValueType>* alpha,
                                  const matrix::Dense<ValueType>* beta,
                                  const array<stopping_status>* stop_status)
{
    const auto num_rows = r->get_size()[0] * r->get_size()[1];
    return memory_bound_work_estimate{2 * num_rows * sizeof(ValueType),
                                      num_rows * sizeof(ValueType)};
}


template <typename ValueType>
memory_bound_work_estimate step_3(
    matrix::Dense<ValueType>* x, matrix::Dense<ValueType>* r,
    const matrix::Dense<ValueType>* s, const matrix::Dense<ValueType>* t,
    const matrix::Dense<ValueType>* y, const matrix::Dense<ValueType>* z,
    const matrix::Dense<ValueType>* alpha, const matrix::Dense<ValueType>* beta,
    const matrix::Dense<ValueType>* gamma, matrix::Dense<ValueType>* omega,
    const array<stopping_status>* stop_status)
{
    const auto num_rows = x->get_size()[0] * x->get_size()[1];
    return memory_bound_work_estimate{5 * num_rows * sizeof(ValueType),
                                      2 * num_rows * sizeof(ValueType)};
}


template <typename ValueType>
memory_bound_work_estimate finalize(matrix::Dense<ValueType>* x,
                                    const matrix::Dense<ValueType>* y,
                                    const matrix::Dense<ValueType>* alpha,
                                    array<stopping_status>* stop_status)
{
    const auto num_rows = x->get_size()[0] * x->get_size()[1];
    return memory_bound_work_estimate{2 * num_rows * sizeof(ValueType),
                                      num_rows * sizeof(ValueType)};
}

}  // namespace work_estimate::bicgstab
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_BICGSTAB_KERNELS_HPP_
