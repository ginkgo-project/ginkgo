// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/minres_kernels.hpp"

#include <ginkgo/core/base/executor.hpp>

#include "common/unified/base/kernel_launch_solver.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Minres solver namespace.
 *
 * @ingroup minres
 */
namespace minres {
namespace detail {


template <typename T>
GKO_INLINE GKO_ATTRIBUTES void swap(T& a, T& b)
{
    T tmp{b};
    b = a;
    a = tmp;
}


}  // namespace detail


template <typename ValueType>
void initialize(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Dense<ValueType>* r, matrix::Dense<ValueType>* z,
    matrix::Dense<ValueType>* p, matrix::Dense<ValueType>* p_prev,
    matrix::Dense<ValueType>* q, matrix::Dense<ValueType>* q_prev,
    matrix::Dense<ValueType>* v, matrix::Dense<ValueType>* beta,
    matrix::Dense<ValueType>* gamma, matrix::Dense<ValueType>* delta,
    matrix::Dense<ValueType>* cos_prev, matrix::Dense<ValueType>* cos,
    matrix::Dense<ValueType>* sin_prev, matrix::Dense<ValueType>* sin,
    matrix::Dense<ValueType>* eta_next, matrix::Dense<ValueType>* eta,
    array<stopping_status>* stop_status)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto col, auto beta, auto gamma, auto delta,
                      auto cos_prev, auto cos, auto sin_prev, auto sin,
                      auto eta_next, auto eta, auto stop) {
            delta[col] = gamma[col] = cos_prev[col] = sin_prev[col] = sin[col] =
                zero(*delta);
            cos[col] = one(*delta);
            eta_next[col] = eta[col] = beta[col] = sqrt(beta[col]);
            stop[col].reset();
        },
        beta->get_num_stored_elements(), row_vector(beta), row_vector(gamma),
        row_vector(delta), row_vector(cos_prev), row_vector(cos),
        row_vector(sin_prev), row_vector(sin), row_vector(eta_next),
        row_vector(eta), *stop_status);

    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto r, auto z, auto p, auto p_prev,
                      auto q, auto q_prev, auto v, auto beta, auto stop) {
            q(row, col) = safe_divide(r(row, col), beta[col]);
            z(row, col) = safe_divide(z(row, col), beta[col]);
            p(row, col) = p_prev(row, col) = q_prev(row, col) = v(row, col) =
                zero(p(row, col));
        },
        r->get_size(), r->get_stride(), default_stride(r), default_stride(z),
        default_stride(p), default_stride(p_prev), default_stride(q),
        default_stride(q_prev), default_stride(v), row_vector(beta),
        *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MINRES_INITIALIZE_KERNEL);


template <typename ValueType>
GKO_KERNEL void update_givens_rotation(ValueType& alpha, const ValueType& beta,
                                       ValueType& cos, ValueType& sin)
{
    if (alpha == zero(alpha)) {
        cos = zero(cos);
        sin = one(sin);
    } else {
        const auto scale = abs(alpha) + abs(beta);
        const auto hypotenuse =
            scale * sqrt(abs(alpha / scale) * abs(alpha / scale) +
                         abs(beta / scale) * abs(beta / scale));
        cos = conj(alpha) / hypotenuse;
        sin = conj(beta) / hypotenuse;
    }
    alpha = cos * alpha + sin * beta;
}


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            matrix::Dense<ValueType>* alpha, matrix::Dense<ValueType>* beta,
            matrix::Dense<ValueType>* gamma, matrix::Dense<ValueType>* delta,
            matrix::Dense<ValueType>* cos_prev, matrix::Dense<ValueType>* cos,
            matrix::Dense<ValueType>* sin_prev, matrix::Dense<ValueType>* sin,
            matrix::Dense<ValueType>* eta, matrix::Dense<ValueType>* eta_next,
            matrix::Dense<ValueType>* tau,
            const array<stopping_status>* stop_status)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto col, auto alpha, auto beta, auto gamma, auto delta,
                      auto cos_prev, auto cos, auto sin_prev, auto sin,
                      auto eta_next, auto eta, auto tau, auto stop) {
            if (!stop[col].has_stopped()) {
                beta[col] = sqrt(beta[col]);
                delta[col] = sin_prev[col] * gamma[col];
                const auto tmp_d = gamma[col];
                const auto tmp_a = alpha[col];
                gamma[col] =
                    cos_prev[col] * cos[col] * tmp_d + sin[col] * tmp_a;
                alpha[col] =
                    -conj(sin[col]) * cos_prev[col] * tmp_d + cos[col] * tmp_a;

                detail::swap(cos[col], cos_prev[col]);
                detail::swap(sin[col], sin_prev[col]);
                update_givens_rotation(alpha[col], beta[col], cos[col],
                                       sin[col]);

                tau[col] = sin[col] * sin[col] * tau[col];
                eta[col] = eta_next[col];
                eta_next[col] = -conj(sin[col]) * eta[col];
            }
        },
        alpha->get_num_stored_elements(), row_vector(alpha), row_vector(beta),
        row_vector(gamma), row_vector(delta), row_vector(cos_prev),
        row_vector(cos), row_vector(sin_prev), row_vector(sin),
        row_vector(eta_next), row_vector(eta), row_vector(tau), *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MINRES_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DefaultExecutor> exec,
            matrix::Dense<ValueType>* x, matrix::Dense<ValueType>* p,
            const matrix::Dense<ValueType>* p_prev, matrix::Dense<ValueType>* z,
            const matrix::Dense<ValueType>* z_tilde,
            matrix::Dense<ValueType>* q, matrix::Dense<ValueType>* q_prev,
            matrix::Dense<ValueType>* v, const matrix::Dense<ValueType>* alpha,
            const matrix::Dense<ValueType>* beta,
            const matrix::Dense<ValueType>* gamma,
            const matrix::Dense<ValueType>* delta,
            const matrix::Dense<ValueType>* cos,
            const matrix::Dense<ValueType>* eta,
            const array<stopping_status>* stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto x, auto p, auto p_prev, auto q,
                      auto q_prev, auto v, auto z, auto z_tilde, auto alpha,
                      auto beta, auto gamma, auto delta, auto cos, auto eta,
                      auto stop) {
            if (!stop[col].has_stopped()) {
                p(row, col) =
                    safe_divide(z(row, col) - gamma[col] * p_prev(row, col) -
                                    delta[col] * p(row, col),
                                alpha[col]);
                x(row, col) = x(row, col) + cos[col] * eta[col] * p(row, col);

                q_prev(row, col) = v(row, col);
                const auto tmp = q(row, col);
                z(row, col) = safe_divide(z_tilde(row, col), beta[col]);
                q(row, col) = safe_divide(v(row, col), beta[col]);
                v(row, col) = tmp * beta[col];
            }
        },
        x->get_size(), p->get_stride(), x, default_stride(p),
        default_stride(p_prev), default_stride(q), default_stride(q_prev),
        default_stride(v), default_stride(z), default_stride(z_tilde),
        row_vector(alpha), row_vector(beta), row_vector(gamma),
        row_vector(delta), row_vector(cos), row_vector(eta), *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MINRES_STEP_2_KERNEL);


}  // namespace minres
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
