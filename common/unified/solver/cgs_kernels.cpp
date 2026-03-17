// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/cgs_kernels.hpp"

#include <ginkgo/core/base/math.hpp>

#include "common/unified/base/kernel_launch_solver.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The CGS solver namespace.
 *
 * @ingroup cgs
 */
namespace cgs {


template <typename ValueType>
void initialize(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::dense<const ValueType> b, matrix::view::dense<ValueType> r,
    matrix::view::dense<ValueType> r_tld, matrix::view::dense<ValueType> p,
    matrix::view::dense<ValueType> q, matrix::view::dense<ValueType> u,
    matrix::view::dense<ValueType> u_hat, matrix::view::dense<ValueType> v_hat,
    matrix::view::dense<ValueType> t, matrix::view::dense<ValueType> alpha,
    matrix::view::dense<ValueType> beta, matrix::view::dense<ValueType> gamma,
    matrix::view::dense<ValueType> prev_rho, matrix::view::dense<ValueType> rho,
    array<stopping_status>& stop_status)
{
    if (b.size) {
        run_kernel_solver(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto b, auto r, auto r_tld,
                          auto p, auto q, auto u, auto u_hat, auto v_hat,
                          auto t, auto alpha, auto beta, auto gamma,
                          auto prev_rho, auto rho, auto stop) {
                if (row == 0) {
                    rho[col] = zero(rho[col]);
                    prev_rho[col] = alpha[col] = beta[col] = gamma[col] =
                        one(prev_rho[col]);
                    stop[col].reset();
                }
                r(row, col) = r_tld(row, col) = b(row, col);
                u(row, col) = u_hat(row, col) = p(row, col) = q(row, col) =
                    v_hat(row, col) = t(row, col) = zero(u(row, col));
            },
            b.size, b.stride, default_stride(b), default_stride(r),
            default_stride(r_tld), default_stride(p), default_stride(q),
            default_stride(u), default_stride(u_hat), default_stride(v_hat),
            default_stride(t), row_vector(alpha), row_vector(beta),
            row_vector(gamma), row_vector(prev_rho), row_vector(rho),
            stop_status);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto col, auto alpha, auto beta, auto gamma,
                          auto prev_rho, auto rho, auto stop) {
                rho[col] = zero(rho[col]);
                prev_rho[col] = alpha[col] = beta[col] = gamma[col] =
                    one(prev_rho[col]);
                stop[col].reset();
            },
            b.size[1], row_vector(alpha), row_vector(beta), row_vector(gamma),
            row_vector(prev_rho), row_vector(rho), stop_status);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            matrix::view::dense<const ValueType> r,
            matrix::view::dense<ValueType> u, matrix::view::dense<ValueType> p,
            matrix::view::dense<const ValueType> q,
            matrix::view::dense<ValueType> beta,
            matrix::view::dense<const ValueType> rho,
            matrix::view::dense<const ValueType> prev_rho,
            const array<stopping_status>& stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto r, auto u, auto p, auto q,
                      auto beta, auto rho, auto prev_rho, auto stop) {
            if (!stop[col].has_stopped()) {
                auto prev_rho_zero = is_zero(prev_rho[col]);
                auto tmp = prev_rho_zero ? beta[col] : rho[col] / prev_rho[col];
                if (row == 0 && !prev_rho_zero) {
                    beta[col] = tmp;
                }
                u(row, col) = r(row, col) + tmp * q(row, col);
                p(row, col) =
                    u(row, col) + tmp * (q(row, col) + tmp * p(row, col));
            }
        },
        r.size, r.stride, default_stride(r), default_stride(u),
        default_stride(p), default_stride(q), row_vector(beta), row_vector(rho),
        row_vector(prev_rho), stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DefaultExecutor> exec,
            matrix::view::dense<const ValueType> u,
            matrix::view::dense<const ValueType> v_hat,
            matrix::view::dense<ValueType> q, matrix::view::dense<ValueType> t,
            matrix::view::dense<ValueType> alpha,
            matrix::view::dense<const ValueType> rho,
            matrix::view::dense<const ValueType> gamma,
            const array<stopping_status>& stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto u, auto v_hat, auto q, auto t,
                      auto alpha, auto rho, auto gamma, auto stop) {
            if (!stop[col].has_stopped()) {
                auto gamma_is_zero = is_zero(gamma[col]);
                auto tmp = gamma_is_zero ? alpha[col] : rho[col] / gamma[col];
                if (row == 0 && !gamma_is_zero) {
                    alpha[col] = tmp;
                }
                q(row, col) = u(row, col) - tmp * v_hat(row, col);
                t(row, col) = u(row, col) + q(row, col);
            }
        },
        u.size, u.stride, default_stride(u), default_stride(v_hat),
        default_stride(q), default_stride(t), row_vector(alpha),
        row_vector(rho), row_vector(gamma), stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_2_KERNEL);

template <typename ValueType>
void step_3(std::shared_ptr<const DefaultExecutor> exec,
            matrix::view::dense<const ValueType> t,
            matrix::view::dense<const ValueType> u_hat,
            matrix::view::dense<ValueType> r, matrix::view::dense<ValueType> x,
            matrix::view::dense<const ValueType> alpha,
            const array<stopping_status>& stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto t, auto u_hat, auto r, auto x,
                      auto alpha, auto stop) {
            if (!stop[col].has_stopped()) {
                x(row, col) += alpha[col] * u_hat(row, col);
                r(row, col) -= alpha[col] * t(row, col);
            }
        },
        t.size, t.stride, default_stride(t), default_stride(u_hat),
        default_stride(r), x, row_vector(alpha), stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_3_KERNEL);


}  // namespace cgs
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
