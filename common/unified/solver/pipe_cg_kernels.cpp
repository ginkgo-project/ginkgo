// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/pipe_cg_kernels.hpp"

#include <ginkgo/core/base/math.hpp>

#include "common/unified/base/kernel_launch_solver.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The PIPE_CG solver namespace.
 *
 * @ingroup pipe_cg
 */
namespace pipe_cg {


template <typename ValueType>
void initialize_1(std::shared_ptr<const DefaultExecutor> exec,
                  matrix::view::dense<const ValueType> b,
                  matrix::view::dense<ValueType> r,
                  matrix::view::dense<ValueType> prev_rho,
                  array<stopping_status>& stop_status)
{
    if (b.size) {
        run_kernel_solver(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto b, auto r, auto prev_rho,
                          auto stop) {
                if (row == 0) {
                    prev_rho[col] = one(prev_rho[col]);
                    stop[col].reset();
                }
                r(row, col) = b(row, col);
            },
            b.size, b.stride, b, r, row_vector(prev_rho), stop_status);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto col, auto prev_rho, auto stop) {
                prev_rho[col] = one(prev_rho[col]);
                stop[col].reset();
            },
            b.size[1], row_vector(prev_rho), stop_status);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_INITIALIZE_1_KERNEL);

template <typename ValueType>
void initialize_2(std::shared_ptr<const DefaultExecutor> exec,
                  matrix::view::dense<ValueType> p,
                  matrix::view::dense<ValueType> q,
                  matrix::view::dense<ValueType> f,
                  matrix::view::dense<ValueType> g,
                  matrix::view::dense<ValueType> beta,
                  matrix::view::dense<const ValueType> z,
                  matrix::view::dense<const ValueType> w,
                  matrix::view::dense<const ValueType> m,
                  matrix::view::dense<const ValueType> n,
                  matrix::view::dense<const ValueType> delta)
{
    // beta = delta
    // p = z
    // q = w
    // f = m
    // g = n
    if (p.size) {
        run_kernel_solver(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto p, auto q, auto f, auto g,
                          auto beta, auto z, auto w, auto m, auto n,
                          auto delta) {
                if (row == 0) {
                    beta[col] = delta[col];
                }
                p(row, col) = z(row, col);
                q(row, col) = w(row, col);
                f(row, col) = m(row, col);
                g(row, col) = n(row, col);
            },
            p.size, p.stride, default_stride(p), default_stride(q),
            default_stride(f), default_stride(g), row_vector(beta), z, w,
            default_stride(m), default_stride(n), row_vector(delta));
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto col, auto beta, auto delta) {
                beta[col] = delta[col];
            },
            p.size[1], row_vector(beta), row_vector(delta));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_INITIALIZE_2_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            matrix::view::dense<ValueType> x, matrix::view::dense<ValueType> r,
            matrix::view::dense<ValueType> z1,
            matrix::view::dense<ValueType> z2, matrix::view::dense<ValueType> w,
            matrix::view::dense<const ValueType> p,
            matrix::view::dense<const ValueType> q,
            matrix::view::dense<const ValueType> f,
            matrix::view::dense<const ValueType> g,
            matrix::view::dense<const ValueType> rho,
            matrix::view::dense<const ValueType> beta,
            const array<stopping_status>& stop_status)
{
    // tmp = rho / beta
    // x = x + tmp * p
    // r = r - tmp * q
    // z = z - tmp * f
    // w = w - tmp * g
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto x, auto r, auto z1, auto z2,
                      auto w, auto p, auto q, auto f, auto g, auto rho,
                      auto beta, auto stop) {
            if (!stop[col].has_stopped()) {
                auto tmp = safe_divide(rho[col], beta[col]);
                x(row, col) += tmp * p(row, col);
                r(row, col) -= tmp * q(row, col);
                z1(row, col) -= tmp * f(row, col);
                z2(row, col) = z1(row, col);
                w(row, col) -= tmp * g(row, col);
            }
        },
        x.size, x.stride, default_stride(x), r, z1, z2, w, default_stride(p),
        default_stride(q), default_stride(f), default_stride(g),
        row_vector(rho), row_vector(beta), stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DefaultExecutor> exec,
            matrix::view::dense<ValueType> beta,
            matrix::view::dense<ValueType> p, matrix::view::dense<ValueType> q,
            matrix::view::dense<ValueType> f, matrix::view::dense<ValueType> g,
            matrix::view::dense<const ValueType> z,
            matrix::view::dense<const ValueType> w,
            matrix::view::dense<const ValueType> m,
            matrix::view::dense<const ValueType> n,
            matrix::view::dense<const ValueType> prev_rho,
            matrix::view::dense<const ValueType> rho,
            matrix::view::dense<const ValueType> delta,
            const array<stopping_status>& stop_status)
{
    // tmp = rho / prev_rho
    // beta = delta - |tmp|^2 * beta
    // p = z + tmp * p
    // q = w + tmp * q
    // f = m + tmp * f
    // g = n + tmp * g
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto beta, auto p, auto q, auto f,
                      auto g, auto z, auto w, auto m, auto n, auto prev_rho,
                      auto rho, auto delta, auto stop) {
            if (!stop[col].has_stopped()) {
                auto tmp = safe_divide(rho[col], prev_rho[col]);
                if (row == 0) {
                    auto abs_tmp = abs(tmp);
                    beta[col] = delta[col] - abs_tmp * abs_tmp * beta[col];
                    if (is_zero(beta[col])) {
                        beta[col] = delta[col];
                    }
                }
                p(row, col) = z(row, col) + tmp * p(row, col);
                q(row, col) = w(row, col) + tmp * q(row, col);
                f(row, col) = m(row, col) + tmp * f(row, col);
                g(row, col) = n(row, col) + tmp * g(row, col);
            }
        },
        p.size, p.stride, row_vector(beta), default_stride(p),
        default_stride(q), default_stride(f), default_stride(g), z, w,
        default_stride(m), default_stride(n), row_vector(prev_rho),
        row_vector(rho), row_vector(delta), stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_STEP_2_KERNEL);


}  // namespace pipe_cg
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
