// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
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


/**
 * Per-cycle statistics of the orthogonalization, with w the vector the first
 * pass left and h = h_{1:j,j} the Hessenberg column it wrote:
 *
 *     psi_j   = ||V_j^T w||_2 / h_{j+1,j},   what one pass leaves,
 *     chi_j   = ||h||_2 / h_{j+1,j},         the cancellation ratio,
 *     theta_j >= ||V_j^T w||_2 / ||h||_2,    a lower bound on the loss of
 *                                            orthogonality of the basis,
 *
 * the last because V_j^T w = (I - V_j^T V_j) h up to rounding. The
 * localization ratio ||w||_inf / ||w||_2 is collected beside them, since it
 * decides how an absolute storage perturbation compares with a relative one.
 *
 * The second orthogonalization pass forms V_j^T w anyway, so this costs one
 * reduction of j+1 values per step. A null pointer turns it off.
 */
struct cb_gmres_psi_stats {
    double max = 0.0;
    double sum = 0.0;
    size_type count = 0;
    double lb_max = 0.0;
    double lb_sum = 0.0;
    double chi_max = 0.0;
    double chi_sum = 0.0;
    double loc_max = 0.0;
    double loc_sum = 0.0;
    size_type loc_count = 0;

    /** @param psi  ||V^T w|| / ||w||, @param chi  ||h|| / ||w||. */
    void add(double psi, double chi)
    {
        max = psi > max ? psi : max;
        sum += psi;
        ++count;
        chi_max = chi > chi_max ? chi : chi_max;
        chi_sum += chi;
        const double lb = chi > 0.0 ? psi / chi : 0.0;
        lb_max = lb > lb_max ? lb : lb_max;
        lb_sum += lb;
    }

    /** @param loc  ||w||_inf / ||w||_2 of one basis vector. */
    void add_loc(double loc)
    {
        loc_max = loc > loc_max ? loc : loc_max;
        loc_sum += loc;
        ++loc_count;
    }
};


#define GKO_DECLARE_CB_GMRES_INITIALIZE_KERNEL(ValueType)        \
    void initialize(std::shared_ptr<const DefaultExecutor> exec, \
                    matrix::view::dense<const ValueType> b,      \
                    matrix::view::dense<ValueType> residual,     \
                    matrix::view::dense<ValueType> givens_sin,   \
                    matrix::view::dense<ValueType> givens_cos,   \
                    array<stopping_status>& stop_status, size_type krylov_dim)


#define GKO_DECLARE_CB_GMRES_RESTART_KERNEL(ValueType, Accessor3d)             \
    void restart(std::shared_ptr<const DefaultExecutor> exec,                  \
                 matrix::view::dense<const ValueType> residual,                \
                 matrix::view::dense<remove_complex<ValueType>> residual_norm, \
                 matrix::view::dense<ValueType> residual_norm_collection,      \
                 matrix::view::dense<remove_complex<ValueType>> arnoldi_norm,  \
                 Accessor3d krylov_bases,                                      \
                 matrix::view::dense<ValueType> next_krylov_basis,             \
                 array<size_type>& final_iter_nums,                            \
                 array<char>& reduction_tmp, size_type krylov_dim)


#define GKO_DECLARE_CB_GMRES_ARNOLDI_KERNEL(ValueType, Accessor3d)             \
    void arnoldi(std::shared_ptr<const DefaultExecutor> exec,                  \
                 matrix::view::dense<ValueType> next_krylov_basis,             \
                 matrix::view::dense<ValueType> givens_sin,                    \
                 matrix::view::dense<ValueType> givens_cos,                    \
                 matrix::view::dense<remove_complex<ValueType>> residual_norm, \
                 matrix::view::dense<ValueType> residual_norm_collection,      \
                 Accessor3d krylov_bases,                                      \
                 matrix::view::dense<ValueType> hessenberg_iter,               \
                 matrix::view::dense<ValueType> buffer_iter,                   \
                 matrix::view::dense<remove_complex<ValueType>> arnoldi_norm,  \
                 size_type iter, array<size_type>& final_iter_nums,            \
                 const array<stopping_status>& stop_status,                    \
                 array<stopping_status>& reorth_status,                        \
                 array<size_type>& num_reorth, size_type max_reorth_steps,     \
                 size_type* reorth_passes, cb_gmres_psi_stats* psi_stats)

/**
 * One column of the Gram matrix of the stored basis:
 *     gram_col(k) = <krylov_bases(k, :), krylov_bases(iter, :)>,  k <= iter.
 *
 * Over a cycle these give I - Vbar^T Vbar, and with it the loss of
 * orthogonality itself rather than a bound on it. Each call traverses the
 * stored basis, so this runs only when the cycle log asks for it.
 */
#define GKO_DECLARE_CB_GMRES_GRAM_COLUMN_KERNEL(ValueType, Accessor3d) \
    void gram_column(std::shared_ptr<const DefaultExecutor> exec,      \
                     Accessor3d krylov_bases,                          \
                     matrix::view::dense<ValueType> gram_col, size_type iter)


#define GKO_DECLARE_CB_GMRES_SOLVE_KRYLOV_KERNEL(ValueType, Accessor3d) \
    void solve_krylov(                                                  \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        matrix::view::dense<const ValueType> residual_norm_collection,  \
        Accessor3d krylov_bases,                                        \
        matrix::view::dense<const ValueType> hessenberg,                \
        matrix::view::dense<ValueType> y,                               \
        matrix::view::dense<ValueType> before_preconditioner,           \
        const array<size_type>& final_iter_nums)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                \
    template <typename ValueType>                                   \
    GKO_DECLARE_CB_GMRES_INITIALIZE_KERNEL(ValueType);              \
    template <typename ValueType, typename Accessor3d>              \
    GKO_DECLARE_CB_GMRES_RESTART_KERNEL(ValueType, Accessor3d);     \
    template <typename ValueType, typename Accessor3d>              \
    GKO_DECLARE_CB_GMRES_ARNOLDI_KERNEL(ValueType, Accessor3d);     \
    template <typename ValueType, typename Accessor3d>              \
    GKO_DECLARE_CB_GMRES_GRAM_COLUMN_KERNEL(ValueType, Accessor3d); \
    template <typename ValueType, typename Accessor3d>              \
    GKO_DECLARE_CB_GMRES_SOLVE_KRYLOV_KERNEL(ValueType, Accessor3d)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(cb_gmres, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_CB_GMRES_KERNELS_HPP_
