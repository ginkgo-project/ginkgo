// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_COMMON_GMRES_KERNELS_HPP_
#define GKO_CORE_SOLVER_COMMON_GMRES_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/base/kernel_declaration.hpp"
#include "ginkgo/core/base/work_estimate.hpp"


namespace gko {
namespace kernels {
namespace common_gmres {


#define GKO_DECLARE_COMMON_GMRES_INITIALIZE_KERNEL(_type)                   \
    void initialize(                                                        \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const matrix::Dense<_type>* b, matrix::Dense<_type>* residual,      \
        matrix::Dense<_type>* givens_sin, matrix::Dense<_type>* givens_cos, \
        stopping_status* stop_status)


#define GKO_DECLARE_COMMON_GMRES_HESSENBERG_QR_KERNEL(_type)                \
    void hessenberg_qr(                                                     \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        matrix::Dense<_type>* givens_sin, matrix::Dense<_type>* givens_cos, \
        matrix::Dense<remove_complex<_type>>* residual_norm,                \
        matrix::Dense<_type>* residual_norm_collection,                     \
        matrix::Dense<_type>* hessenberg_iter, size_type iter,              \
        size_type* final_iter_nums, const stopping_status* stop_status)


#define GKO_DECLARE_COMMON_GMRES_SOLVE_KRYLOV_KERNEL(_type1)               \
    void solve_krylov(                                                     \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const matrix::Dense<_type1>* residual_norm_collection,             \
        const matrix::Dense<_type1>* hessenberg, matrix::Dense<_type1>* y, \
        const size_type* final_iter_nums, const stopping_status* stop_status)


#define GKO_DECLARE_ALL_AS_TEMPLATES                          \
    template <typename ValueType>                             \
    GKO_DECLARE_COMMON_GMRES_INITIALIZE_KERNEL(ValueType);    \
    template <typename ValueType>                             \
    GKO_DECLARE_COMMON_GMRES_HESSENBERG_QR_KERNEL(ValueType); \
    template <typename ValueType>                             \
    GKO_DECLARE_COMMON_GMRES_SOLVE_KRYLOV_KERNEL(ValueType)


}  // namespace common_gmres


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(common_gmres,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


namespace work_estimate::common_gmres {


template <typename ValueType>
memory_bound_work_estimate initialize(const matrix::Dense<ValueType>* b,
                                      matrix::Dense<ValueType>* residual,
                                      matrix::Dense<ValueType>* givens_sin,
                                      matrix::Dense<ValueType>* givens_cos,
                                      stopping_status* stop_status)
{
    const auto num_values = b->get_size()[0] * b->get_size()[1];
    const auto num_givens_values =
        givens_sin->get_size()[0] * givens_sin->get_size()[1];
    return memory_bound_work_estimate{
        num_values * sizeof(ValueType),
        (num_values + 2 * num_givens_values) * sizeof(ValueType)};
}


template <typename ValueType>
memory_bound_work_estimate hessenberg_qr(
    matrix::Dense<ValueType>* givens_sin, matrix::Dense<ValueType>* givens_cos,
    const matrix::Dense<remove_complex<ValueType>>* residual_norm,
    matrix::Dense<ValueType>* residual_norm_collection,
    matrix::Dense<ValueType>* hessenberg_iter, size_type iter,
    size_type* final_iter_nums, const stopping_status* stop_status)
{
    const auto num_rhs = hessenberg_iter->get_size()[1];
    const auto givens_values = (iter + 1) * num_rhs;
    const auto hessenberg_values = (iter + 2) * num_rhs;
    return memory_bound_work_estimate{
        (2 * givens_values + hessenberg_values + num_rhs) * sizeof(ValueType),
        (hessenberg_values + 5 * num_rhs) * sizeof(ValueType)};
}


template <typename ValueType>
memory_bound_work_estimate solve_krylov(
    const matrix::Dense<ValueType>* residual_norm_collection,
    const matrix::Dense<ValueType>* hessenberg, matrix::Dense<ValueType>* y,
    const size_type* final_iter_nums, const stopping_status* stop_status)
{
    const auto num_rhs = residual_norm_collection->get_size()[1];
    const auto krylov_dim = hessenberg->get_size()[0];
    const auto triangular_values = krylov_dim * (krylov_dim + 1) / 2 * num_rhs;
    const auto y_values = krylov_dim * num_rhs;
    return memory_bound_work_estimate{
        (triangular_values + 2 * y_values) * sizeof(ValueType),
        y_values * sizeof(ValueType)};
}

}  // namespace work_estimate::common_gmres
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_COMMON_GMRES_KERNELS_HPP_
