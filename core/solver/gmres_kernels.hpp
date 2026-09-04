// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_GMRES_KERNELS_HPP_
#define GKO_CORE_SOLVER_GMRES_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/multivector.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace gmres {


#define GKO_DECLARE_GMRES_RESTART_KERNEL(ValueType)                         \
    void restart(                                                           \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        matrix::view::dense<const ValueType> residual,                      \
        matrix::view::dense<const remove_complex<ValueType>> residual_norm, \
        matrix::view::dense<ValueType> residual_norm_collection,            \
        matrix::view::dense<ValueType> krylov_bases,                        \
        size_type* final_iter_nums)


#define GKO_DECLARE_GMRES_MULTI_AXPY_KERNEL(ValueType)                    \
    void multi_axpy(std::shared_ptr<const DefaultExecutor> exec,          \
                    matrix::view::dense<const ValueType> krylov_bases,    \
                    matrix::view::dense<const ValueType> y,               \
                    matrix::view::dense<ValueType> before_preconditioner, \
                    const size_type* final_iter_nums,                     \
                    stopping_status* stop_status)


#define GKO_DECLARE_GMRES_MULTI_DOT_KERNEL(ValueType)                 \
    void multi_dot(std::shared_ptr<const DefaultExecutor> exec,       \
                   matrix::view::dense<const ValueType> krylov_bases, \
                   matrix::view::dense<const ValueType> next_krylov,  \
                   matrix::view::dense<ValueType> hessenberg_col)


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


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_GMRES_KERNELS_HPP_
