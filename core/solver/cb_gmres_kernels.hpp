// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_CB_GMRES_KERNELS_HPP_
#define GKO_CORE_SOLVER_CB_GMRES_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "accessor/reduced_row_major.hpp"
#include "accessor/scaled_reduced_row_major.hpp"
#include "core/base/extended_float.hpp"
#include "core/base/kernel_declaration.hpp"


// TODO Find way around using it!
#define GKO_UNPACK(...) __VA_ARGS__
/**
 * Instantiates a template for each value type with each lower precision type
 * supported by Ginkgo for CbGmres.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take two arguments:
 *                1. the first will be used as the regular ValueType
 *                     (precisions supported by Ginkgo), and
 *                2. the second the value type of the reduced precision.
 * @param _const  qualifier used for the storage type, indicating if it is a
 *                const accessor, or not.
 */
#define GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE_HELPER(_macro, _const)          \
    template _macro(                                                           \
        double,                                                                \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<3, double, _const double>>));    \
    template _macro(                                                           \
        double,                                                                \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<3, double, _const float>>));     \
    template _macro(                                                           \
        double,                                                                \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<3, double, _const half>>));      \
    template _macro(double,                                                    \
                    GKO_UNPACK(acc::range<acc::scaled_reduced_row_major<       \
                                   3, double, _const int64, 0b101>>));         \
    template _macro(double,                                                    \
                    GKO_UNPACK(acc::range<acc::scaled_reduced_row_major<       \
                                   3, double, _const int32, 0b101>>));         \
    template _macro(double,                                                    \
                    GKO_UNPACK(acc::range<acc::scaled_reduced_row_major<       \
                                   3, double, _const int16, 0b101>>));         \
    template _macro(                                                           \
        float,                                                                 \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<3, float, _const float>>));      \
    template _macro(                                                           \
        float,                                                                 \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<3, float, _const half>>));       \
    template _macro(float,                                                     \
                    GKO_UNPACK(acc::range<acc::scaled_reduced_row_major<       \
                                   3, float, _const int32, 0b101>>));          \
    template _macro(float,                                                     \
                    GKO_UNPACK(acc::range<acc::scaled_reduced_row_major<       \
                                   3, float, _const int16, 0b101>>));          \
    template _macro(                                                           \
        std::complex<double>,                                                  \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<3, std::complex<double>,         \
                                              _const std::complex<double>>>)); \
    template _macro(                                                           \
        std::complex<double>,                                                  \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<3, std::complex<double>,         \
                                              _const std::complex<float>>>));  \
    template _macro(                                                           \
        std::complex<float>,                                                   \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<3, std::complex<float>,          \
                                              _const std::complex<float>>>))

#define GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE(_macro) \
    GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE_HELPER(_macro, )

#define GKO_INSTANTIATE_FOR_EACH_CB_GMRES_CONST_TYPE(_macro) \
    GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE_HELPER(_macro, const)


namespace gko {
namespace kernels {


#define GKO_DECLARE_CB_GMRES_INITIALIZE_KERNEL(_type)                       \
    void initialize(                                                        \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const matrix::Dense<_type>* b, matrix::Dense<_type>* residual,      \
        matrix::Dense<_type>* givens_sin, matrix::Dense<_type>* givens_cos, \
        array<stopping_status>* stop_status, size_type krylov_dim)


#define GKO_DECLARE_CB_GMRES_RESTART_KERNEL(_type1, _range)            \
    void restart(std::shared_ptr<const DefaultExecutor> exec,          \
                 const matrix::Dense<_type1>* residual,                \
                 matrix::Dense<remove_complex<_type1>>* residual_norm, \
                 matrix::Dense<_type1>* residual_norm_collection,      \
                 matrix::Dense<remove_complex<_type1>>* arnoldi_norm,  \
                 _range krylov_bases,                                  \
                 matrix::Dense<_type1>* next_krylov_basis,             \
                 array<size_type>* final_iter_nums,                    \
                 array<char>& reduction_tmp, size_type krylov_dim)


#define GKO_DECLARE_CB_GMRES_ARNOLDI_KERNEL(_type1, _range)                   \
    void arnoldi(                                                             \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        matrix::Dense<_type1>* next_krylov_basis,                             \
        matrix::Dense<_type1>* givens_sin, matrix::Dense<_type1>* givens_cos, \
        matrix::Dense<remove_complex<_type1>>* residual_norm,                 \
        matrix::Dense<_type1>* residual_norm_collection, _range krylov_bases, \
        matrix::Dense<_type1>* hessenberg_iter,                               \
        matrix::Dense<_type1>* buffer_iter,                                   \
        matrix::Dense<remove_complex<_type1>>* arnoldi_norm, size_type iter,  \
        array<size_type>* final_iter_nums,                                    \
        const array<stopping_status>* stop_status,                            \
        array<stopping_status>* reorth_status, array<size_type>* num_reorth)

#define GKO_DECLARE_CB_GMRES_SOLVE_KRYLOV_KERNEL(_type1, _range)             \
    void solve_krylov(std::shared_ptr<const DefaultExecutor> exec,           \
                      const matrix::Dense<_type1>* residual_norm_collection, \
                      _range krylov_bases,                                   \
                      const matrix::Dense<_type1>* hessenberg,               \
                      matrix::Dense<_type1>* y,                              \
                      matrix::Dense<_type1>* before_preconditioner,          \
                      const array<size_type>* final_iter_nums)


#define GKO_DECLARE_ALL_AS_TEMPLATES                            \
    template <typename ValueType>                               \
    GKO_DECLARE_CB_GMRES_INITIALIZE_KERNEL(ValueType);          \
    template <typename ValueType, typename Accessor3d>          \
    GKO_DECLARE_CB_GMRES_RESTART_KERNEL(ValueType, Accessor3d); \
    template <typename ValueType, typename Accessor3d>          \
    GKO_DECLARE_CB_GMRES_ARNOLDI_KERNEL(ValueType, Accessor3d); \
    template <typename ValueType, typename Accessor3d>          \
    GKO_DECLARE_CB_GMRES_SOLVE_KRYLOV_KERNEL(ValueType, Accessor3d)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(cb_gmres, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_CB_GMRES_KERNELS_HPP_
