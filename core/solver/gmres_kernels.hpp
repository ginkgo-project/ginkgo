// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_GMRES_KERNELS_HPP_
#define GKO_CORE_SOLVER_GMRES_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/base/kernel_declaration.hpp"
#include "ginkgo/core/base/work_estimate.hpp"


namespace gko {
namespace kernels {
namespace gmres {


#define GKO_DECLARE_GMRES_RESTART_KERNEL(_type)                             \
    void restart(std::shared_ptr<const DefaultExecutor> exec,               \
                 const matrix::Dense<_type>* residual,                      \
                 const matrix::Dense<remove_complex<_type>>* residual_norm, \
                 matrix::Dense<_type>* residual_norm_collection,            \
                 matrix::Dense<_type>* krylov_bases,                        \
                 size_type* final_iter_nums)


#define GKO_DECLARE_GMRES_MULTI_AXPY_KERNEL(_type)               \
    void multi_axpy(std::shared_ptr<const DefaultExecutor> exec, \
                    const matrix::Dense<_type>* krylov_bases,    \
                    const matrix::Dense<_type>* y,               \
                    matrix::Dense<_type>* before_preconditioner, \
                    const size_type* final_iter_nums,            \
                    stopping_status* stop_status)


#define GKO_DECLARE_GMRES_MULTI_DOT_KERNEL(_type)               \
    void multi_dot(std::shared_ptr<const DefaultExecutor> exec, \
                   const matrix::Dense<_type>* krylov_bases,    \
                   const matrix::Dense<_type>* next_krylov,     \
                   matrix::Dense<_type>* hessenberg_col)


#define GKO_DECLARE_ALL_AS_TEMPLATES                \
    template <typename ValueType>                   \
    GKO_DECLARE_GMRES_RESTART_KERNEL(ValueType);    \
    template <typename ValueType>                   \
    GKO_DECLARE_GMRES_MULTI_AXPY_KERNEL(ValueType); \
    template <typename ValueType>                   \
    GKO_DECLARE_GMRES_MULTI_DOT_KERNEL(ValueType)


}  // namespace gmres


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(gmres, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


namespace work_estimate::gmres {


template <typename ValueType>
memory_bound_work_estimate restart(
    const matrix::Dense<ValueType>* residual,
    const matrix::Dense<remove_complex<ValueType>>* residual_norm,
    matrix::Dense<ValueType>* residual_norm_collection,
    matrix::Dense<ValueType>* krylov_bases, size_type* final_iter_nums)
{
    const auto num_values = residual->get_size()[0] * residual->get_size()[1];
    const auto num_rhs = residual->get_size()[1];
    return memory_bound_work_estimate{
        (num_values + num_rhs) * sizeof(ValueType),
        (num_values + num_rhs) * sizeof(ValueType)};
}


template <typename ValueType>
memory_bound_work_estimate multi_axpy(
    const matrix::Dense<ValueType>* krylov_bases,
    const matrix::Dense<ValueType>* y,
    matrix::Dense<ValueType>* before_preconditioner,
    const size_type* final_iter_nums, stopping_status* stop_status)
{
    const auto num_values = before_preconditioner->get_size()[0] *
                            before_preconditioner->get_size()[1];
    const auto krylov_basis_values =
        krylov_bases->get_size()[0] * krylov_bases->get_size()[1];
    const auto y_values = y->get_size()[0] * y->get_size()[1];
    return memory_bound_work_estimate{
        (krylov_basis_values + y_values) * sizeof(ValueType),
        num_values * sizeof(ValueType)};
}


template <typename ValueType>
memory_bound_work_estimate multi_dot(
    const matrix::Dense<ValueType>* krylov_bases,
    const matrix::Dense<ValueType>* next_krylov,
    matrix::Dense<ValueType>* hessenberg_col)
{
    const auto krylov_basis_values =
        krylov_bases->get_size()[0] * krylov_bases->get_size()[1];
    const auto next_krylov_values =
        next_krylov->get_size()[0] * next_krylov->get_size()[1];
    const auto hessenberg_values =
        (hessenberg_col->get_size()[0] - 1) * hessenberg_col->get_size()[1];
    return memory_bound_work_estimate{
        (krylov_basis_values + next_krylov_values) * sizeof(ValueType),
        hessenberg_values * sizeof(ValueType)};
}


}  // namespace work_estimate::gmres
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_GMRES_KERNELS_HPP_
