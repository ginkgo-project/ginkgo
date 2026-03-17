// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/fcg_kernels.hpp"

#include <ginkgo/core/base/math.hpp>

#include "common/unified/base/kernel_launch_solver.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The FCG solver namespace.
 *
 * @ingroup fcg
 */
namespace fcg {


template <typename ValueType>
void initialize(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::dense<const ValueType> b, matrix::view::dense<ValueType> r,
    matrix::view::dense<ValueType> z, matrix::view::dense<ValueType> p,
    matrix::view::dense<ValueType> q, matrix::view::dense<ValueType> t,
    matrix::view::dense<ValueType> prev_rho, matrix::view::dense<ValueType> rho,
    matrix::view::dense<ValueType> rho_t, array<stopping_status>& stop_status)
{
    if (b.size) {
        run_kernel_solver(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto b, auto r, auto z, auto p,
                          auto q, auto t, auto prev_rho, auto rho, auto rho_t,
                          auto stop) {
                if (row == 0) {
                    rho[col] = zero(rho[col]);
                    prev_rho[col] = rho_t[col] = one(prev_rho[col]);
                    stop[col].reset();
                }
                t(row, col) = r(row, col) = b(row, col);
                z(row, col) = p(row, col) = q(row, col) = zero(z(row, col));
            },
            b.size, b.stride, default_stride(b), default_stride(r),
            default_stride(z), default_stride(p), default_stride(q),
            default_stride(t), row_vector(prev_rho), row_vector(rho),
            row_vector(rho_t), stop_status);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto col, auto prev_rho, auto rho, auto rho_t,
                          auto stop) {
                rho[col] = zero(rho[col]);
                prev_rho[col] = rho_t[col] = one(prev_rho[col]);
                stop[col].reset();
            },
            b.size[1], row_vector(prev_rho), row_vector(rho), row_vector(rho_t),
            stop_status);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FCG_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            matrix::view::dense<ValueType> p,
            matrix::view::dense<const ValueType> z,
            matrix::view::dense<const ValueType> rho_t,
            matrix::view::dense<const ValueType> prev_rho,
            const array<stopping_status>& stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto p, auto z, auto rho_t,
                      auto prev_rho, auto stop) {
            if (!stop[col].has_stopped()) {
                auto tmp = safe_divide(rho_t[col], prev_rho[col]);
                p(row, col) = z(row, col) + tmp * p(row, col);
            }
        },
        p.size, p.stride, default_stride(p), default_stride(z),
        row_vector(rho_t), row_vector(prev_rho), stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FCG_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DefaultExecutor> exec,
            matrix::view::dense<ValueType> x, matrix::view::dense<ValueType> r,
            matrix::view::dense<ValueType> t,
            matrix::view::dense<const ValueType> p,
            matrix::view::dense<const ValueType> q,
            matrix::view::dense<const ValueType> beta,
            matrix::view::dense<const ValueType> rho,
            const array<stopping_status>& stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto x, auto r, auto t, auto p,
                      auto q, auto beta, auto rho, auto stop) {
            if (!stop[col].has_stopped() && is_nonzero(beta[col])) {
                auto tmp = rho[col] / beta[col];
                auto prev_r = r(row, col);
                x(row, col) += tmp * p(row, col);
                r(row, col) -= tmp * q(row, col);
                t(row, col) = r(row, col) - prev_r;
            }
        },
        x.size, r.stride, x, default_stride(r), default_stride(t),
        default_stride(p), default_stride(q), row_vector(beta), row_vector(rho),
        stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FCG_STEP_2_KERNEL);


}  // namespace fcg
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
