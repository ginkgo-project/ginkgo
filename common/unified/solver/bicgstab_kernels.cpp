// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/bicgstab_kernels.hpp"

#include <ginkgo/core/base/math.hpp>

#include "common/unified/base/kernel_launch_solver.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The BICGSTAB solver namespace.
 *
 * @ingroup bicgstab
 */
namespace bicgstab {


template <typename ValueType>
void initialize(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::dense<const ValueType> b, matrix::view::dense<ValueType> r,
    matrix::view::dense<ValueType> rr, matrix::view::dense<ValueType> y,
    matrix::view::dense<ValueType> s, matrix::view::dense<ValueType> t,
    matrix::view::dense<ValueType> z, matrix::view::dense<ValueType> v,
    matrix::view::dense<ValueType> p, matrix::view::dense<ValueType> prev_rho,
    matrix::view::dense<ValueType> rho, matrix::view::dense<ValueType> alpha,
    matrix::view::dense<ValueType> beta, matrix::view::dense<ValueType> gamma,
    matrix::view::dense<ValueType> omega, array<stopping_status>* stop_status)
{
    if (b.size) {
        run_kernel_solver(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto b, auto r, auto rr, auto y,
                          auto s, auto t, auto z, auto v, auto p, auto prev_rho,
                          auto rho, auto alpha, auto beta, auto gamma,
                          auto omega, auto stop) {
                if (row == 0) {
                    rho[col] = prev_rho[col] = alpha[col] = beta[col] =
                        gamma[col] = omega[col] = one(rho[col]);
                    stop[col].reset();
                }
                r(row, col) = b(row, col);
                rr(row, col) = z(row, col) = v(row, col) = s(row, col) = t(
                    row, col) = y(row, col) = p(row, col) = zero(rr(row, col));
            },
            b.size, b.stride, default_stride(b), default_stride(r),
            default_stride(rr), default_stride(y), default_stride(s),
            default_stride(t), default_stride(z), default_stride(v),
            default_stride(p), row_vector(prev_rho), row_vector(rho),
            row_vector(alpha), row_vector(beta), row_vector(gamma),
            row_vector(omega), *stop_status);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto col, auto prev_rho, auto rho, auto alpha,
                          auto beta, auto gamma, auto omega, auto stop) {
                rho[col] = prev_rho[col] = alpha[col] = beta[col] = gamma[col] =
                    omega[col] = one(rho[col]);
                stop[col].reset();
            },
            b.size[1], row_vector(prev_rho), row_vector(rho), row_vector(alpha),
            row_vector(beta), row_vector(gamma), row_vector(omega),
            *stop_status);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            matrix::view::dense<const ValueType> r,
            matrix::view::dense<ValueType> p,
            matrix::view::dense<const ValueType> v,
            matrix::view::dense<const ValueType> rho,
            matrix::view::dense<const ValueType> prev_rho,
            matrix::view::dense<const ValueType> alpha,
            matrix::view::dense<const ValueType> omega,
            const array<stopping_status>* stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto r, auto p, auto v, auto rho,
                      auto prev_rho, auto alpha, auto omega, auto stop) {
            if (!stop[col].has_stopped()) {
                auto tmp = safe_divide(rho[col], prev_rho[col]) *
                           safe_divide(alpha[col], omega[col]);
                p(row, col) = r(row, col) +
                              tmp * (p(row, col) - omega[col] * v(row, col));
            }
        },
        r.size, r.stride, default_stride(r), default_stride(p),
        default_stride(v), row_vector(rho), row_vector(prev_rho),
        row_vector(alpha), row_vector(omega), *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DefaultExecutor> exec,
            matrix::view::dense<const ValueType> r,
            matrix::view::dense<ValueType> s,
            matrix::view::dense<const ValueType> v,
            matrix::view::dense<const ValueType> rho,
            matrix::view::dense<ValueType> alpha,
            matrix::view::dense<const ValueType> beta,
            const array<stopping_status>* stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto r, auto s, auto v, auto rho,
                      auto alpha, auto beta, auto stop) {
            if (!stop[col].has_stopped()) {
                auto tmp = safe_divide(rho[col], beta[col]);
                if (row == 0) {
                    alpha[col] = tmp;
                }
                s(row, col) = r(row, col) - tmp * v(row, col);
            }
        },
        r.size, r.stride, default_stride(r), default_stride(s),
        default_stride(v), row_vector(rho), row_vector(alpha), row_vector(beta),
        *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_2_KERNEL);


template <typename ValueType>
void step_3(std::shared_ptr<const DefaultExecutor> exec,
            matrix::view::dense<ValueType> x, matrix::view::dense<ValueType> r,
            matrix::view::dense<const ValueType> s,
            matrix::view::dense<const ValueType> t,
            matrix::view::dense<const ValueType> y,
            matrix::view::dense<const ValueType> z,
            matrix::view::dense<const ValueType> alpha,
            matrix::view::dense<const ValueType> beta,
            matrix::view::dense<const ValueType> gamma,
            matrix::view::dense<ValueType> omega,
            const array<stopping_status>* stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto x, auto r, auto s, auto t,
                      auto y, auto z, auto alpha, auto beta, auto gamma,
                      auto omega, auto stop) {
            if (!stop[col].has_stopped()) {
                auto tmp = safe_divide(gamma[col], beta[col]);
                if (row == 0) {
                    omega[col] = tmp;
                }
                x(row, col) += alpha[col] * y(row, col) + tmp * z(row, col);
                r(row, col) = s(row, col) - tmp * t(row, col);
            }
        },
        x.size, r.stride, x, default_stride(r), default_stride(s),
        default_stride(t), default_stride(y), default_stride(z),
        row_vector(alpha), row_vector(beta), row_vector(gamma),
        row_vector(omega), *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_3_KERNEL);


template <typename ValueType>
void finalize(std::shared_ptr<const DefaultExecutor> exec,
              matrix::view::dense<ValueType> x,
              matrix::view::dense<const ValueType> y,
              matrix::view::dense<const ValueType> alpha,
              array<stopping_status>* stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto x, auto y, auto alpha,
                      auto stop) {
            if (stop[col].has_stopped() && !stop[col].is_finalized()) {
                x(row, col) += alpha[col] * y(row, col);
            }
        },
        x.size, y.stride, x, default_stride(y), row_vector(alpha),
        *stop_status);
    run_kernel(
        exec,
        [] GKO_KERNEL(auto col, auto stop) {
            if (stop[col].has_stopped() && !stop[col].is_finalized()) {
                stop[col].finalize();
            }
        },
        x.size[1], *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_FINALIZE_KERNEL);


}  // namespace bicgstab
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
