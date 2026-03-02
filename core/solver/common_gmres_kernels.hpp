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


namespace gko {
namespace kernels {
namespace common_gmres {


#define GKO_DECLARE_COMMON_GMRES_INITIALIZE_KERNEL(_type)        \
    void initialize(std::shared_ptr<const DefaultExecutor> exec, \
                    matrix::view::dense<const ValueType> b,      \
                    matrix::view::dense<ValueType> residual,     \
                    matrix::view::dense<ValueType> givens_sin,   \
                    matrix::view::dense<ValueType> givens_cos,   \
                    stopping_status* stop_status)


#define GKO_DECLARE_COMMON_GMRES_HESSENBERG_QR_KERNEL(_type)            \
    void hessenberg_qr(                                                 \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        matrix::view::dense<ValueType> givens_sin,                      \
        matrix::view::dense<ValueType> givens_cos,                      \
        matrix::Dense<remove_complex<_type>>* residual_norm,            \
        matrix::view::dense<ValueType> residual_norm_collection,        \
        matrix::view::dense<ValueType> hessenberg_iter, size_type iter, \
        size_type* final_iter_nums, const stopping_status* stop_status)


#define GKO_DECLARE_COMMON_GMRES_SOLVE_KRYLOV_KERNEL(_type1)                \
    void solve_krylov(                                                      \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        matrix::view::dense<const ValueType> residual_norm_collection,      \
        matrix::view::dense<const ValueType> hessenberg,                    \
        matrix::view::dense<ValueType> y, const size_type* final_iter_nums, \
        const stopping_status* stop_status)


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


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_COMMON_GMRES_KERNELS_HPP_
