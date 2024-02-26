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
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* r,
                matrix::Dense<ValueType>* rr, matrix::Dense<ValueType>* y,
                matrix::Dense<ValueType>* s, matrix::Dense<ValueType>* t,
                matrix::Dense<ValueType>* z, matrix::Dense<ValueType>* v,
                matrix::Dense<ValueType>* p, matrix::Dense<ValueType>* prev_rho,
                matrix::Dense<ValueType>* rho, matrix::Dense<ValueType>* alpha,
                matrix::Dense<ValueType>* beta, matrix::Dense<ValueType>* gamma,
                matrix::Dense<ValueType>* omega,
                array<stopping_status>* stop_status)
{
    if (b->get_size()) {
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
            b->get_size(), b->get_stride(), default_stride(b),
            default_stride(r), default_stride(rr), default_stride(y),
            default_stride(s), default_stride(t), default_stride(z),
            default_stride(v), default_stride(p), row_vector(prev_rho),
            row_vector(rho), row_vector(alpha), row_vector(beta),
            row_vector(gamma), row_vector(omega), *stop_status);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto col, auto prev_rho, auto rho, auto alpha,
                          auto beta, auto gamma, auto omega, auto stop) {
                rho[col] = prev_rho[col] = alpha[col] = beta[col] = gamma[col] =
                    omega[col] = one(rho[col]);
                stop[col].reset();
            },
            b->get_size()[1], row_vector(prev_rho), row_vector(rho),
            row_vector(alpha), row_vector(beta), row_vector(gamma),
            row_vector(omega), *stop_status);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType>* r, matrix::Dense<ValueType>* p,
            const matrix::Dense<ValueType>* v,
            const matrix::Dense<ValueType>* rho,
            const matrix::Dense<ValueType>* prev_rho,
            const matrix::Dense<ValueType>* alpha,
            const matrix::Dense<ValueType>* omega,
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
        r->get_size(), r->get_stride(), default_stride(r), default_stride(p),
        default_stride(v), row_vector(rho), row_vector(prev_rho),
        row_vector(alpha), row_vector(omega), *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType>* r, matrix::Dense<ValueType>* s,
            const matrix::Dense<ValueType>* v,
            const matrix::Dense<ValueType>* rho,
            matrix::Dense<ValueType>* alpha,
            const matrix::Dense<ValueType>* beta,
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
        r->get_size(), r->get_stride(), default_stride(r), default_stride(s),
        default_stride(v), row_vector(rho), row_vector(alpha), row_vector(beta),
        *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_2_KERNEL);


template <typename ValueType>
void step_3(
    std::shared_ptr<const DefaultExecutor> exec, matrix::Dense<ValueType>* x,
    matrix::Dense<ValueType>* r, const matrix::Dense<ValueType>* s,
    const matrix::Dense<ValueType>* t, const matrix::Dense<ValueType>* y,
    const matrix::Dense<ValueType>* z, const matrix::Dense<ValueType>* alpha,
    const matrix::Dense<ValueType>* beta, const matrix::Dense<ValueType>* gamma,
    matrix::Dense<ValueType>* omega, const array<stopping_status>* stop_status)
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
        x->get_size(), r->get_stride(), x, default_stride(r), default_stride(s),
        default_stride(t), default_stride(y), default_stride(z),
        row_vector(alpha), row_vector(beta), row_vector(gamma),
        row_vector(omega), *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_3_KERNEL);


template <typename ValueType>
void finalize(std::shared_ptr<const DefaultExecutor> exec,
              matrix::Dense<ValueType>* x, const matrix::Dense<ValueType>* y,
              const matrix::Dense<ValueType>* alpha,
              array<stopping_status>* stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto x, auto y, auto alpha,
                      auto stop) {
            if (stop[col].has_stopped() && !stop[col].is_finalized()) {
                x(row, col) += alpha[col] * y(row, col);
                stop[col].finalize();
            }
        },
        x->get_size(), y->get_stride(), x, default_stride(y), row_vector(alpha),
        *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_FINALIZE_KERNEL);


}  // namespace bicgstab
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
