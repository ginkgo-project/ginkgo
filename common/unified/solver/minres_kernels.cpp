// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
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
 */
namespace minres {


template <typename ValueType>
void initialize(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::dense<const ValueType> r, matrix::view::dense<ValueType> z,
    matrix::view::dense<ValueType> p, matrix::view::dense<ValueType> p_prev,
    matrix::view::dense<ValueType> q, matrix::view::dense<ValueType> q_prev,
    matrix::view::dense<ValueType> v, matrix::view::dense<ValueType> beta,
    matrix::view::dense<ValueType> gamma, matrix::view::dense<ValueType> delta,
    matrix::view::dense<ValueType> cos_prev, matrix::view::dense<ValueType> cos,
    matrix::view::dense<ValueType> sin_prev, matrix::view::dense<ValueType> sin,
    matrix::view::dense<ValueType> eta_next, matrix::view::dense<ValueType> eta,
    array<stopping_status>& stop_status)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto col, auto beta, auto gamma, auto delta,
                      auto cos_prev, auto cos, auto sin_prev, auto sin,
                      auto eta_next, auto eta, auto stop) {
            using arithmetic_type = arithmetic_type_t<decltype(beta)>;
            delta[col] = gamma[col] = cos_prev[col] = sin_prev[col] = sin[col] =
                zero(delta[col]);
            cos[col] = one(delta[col]);
            eta_next[col] = eta[col] = beta[col] =
                sqrt(static_cast<arithmetic_type>(beta[col]));
            stop[col].reset();
        },
        beta.size[1], arithmetic<ValueType, 1>(beta),
        arithmetic<ValueType, 1>(gamma), arithmetic<ValueType, 1>(delta),
        arithmetic<ValueType, 1>(cos_prev), arithmetic<ValueType, 1>(cos),
        arithmetic<ValueType, 1>(sin_prev), arithmetic<ValueType, 1>(sin),
        arithmetic<ValueType, 1>(eta_next), arithmetic<ValueType, 1>(eta),
        stop_status);

    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto r, auto z, auto p, auto p_prev,
                      auto q, auto q_prev, auto v, auto beta, auto stop) {
            q(row, col) = safe_divide(r(row, col), beta[col]);
            z(row, col) = safe_divide(z(row, col), beta[col]);
            p(row, col) = p_prev(row, col) = q_prev(row, col) = v(row, col) =
                zero(p(row, col));
        },
        r.size, r.stride, arithmetic<ValueType>(r), arithmetic<ValueType>(z),
        arithmetic<ValueType>(p), arithmetic<ValueType>(p_prev),
        arithmetic<ValueType>(q), arithmetic<ValueType>(q_prev),
        arithmetic<ValueType>(v), arithmetic<ValueType, 1>(beta), stop_status);
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
void step_1(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::dense<ValueType> alpha, matrix::view::dense<ValueType> beta,
    matrix::view::dense<ValueType> gamma, matrix::view::dense<ValueType> delta,
    matrix::view::dense<ValueType> cos_prev, matrix::view::dense<ValueType> cos,
    matrix::view::dense<ValueType> sin_prev, matrix::view::dense<ValueType> sin,
    matrix::view::dense<ValueType> eta, matrix::view::dense<ValueType> eta_next,
    matrix::view::dense<ValueType> tau,
    const array<stopping_status>& stop_status)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto col, auto alpha, auto beta, auto gamma, auto delta,
                      auto cos_prev, auto cos, auto sin_prev, auto sin,
                      auto eta_next, auto eta, auto tau, auto stop) {
            using arithmetic_type = arithmetic_type_t<decltype(alpha)>;
            if (!stop[col].has_stopped()) {
                // computes with copies, since the elements can be references
                auto alpha_value = static_cast<arithmetic_type>(alpha[col]);
                const auto beta_value =
                    sqrt(static_cast<arithmetic_type>(beta[col]));
                const auto gamma_value =
                    static_cast<arithmetic_type>(gamma[col]);
                const auto cos_value = static_cast<arithmetic_type>(cos[col]);
                const auto cos_prev_value =
                    static_cast<arithmetic_type>(cos_prev[col]);
                const auto sin_value = static_cast<arithmetic_type>(sin[col]);
                const auto sin_prev_value =
                    static_cast<arithmetic_type>(sin_prev[col]);
                beta[col] = beta_value;
                delta[col] = sin_prev_value * gamma_value;
                gamma[col] = cos_prev_value * cos_value * gamma_value +
                             sin_value * alpha_value;
                alpha_value = -conj(sin_value) * cos_prev_value * gamma_value +
                              cos_value * alpha_value;

                // the current rotation becomes the previous one
                cos_prev[col] = cos_value;
                sin_prev[col] = sin_value;
                auto new_cos = cos_value;
                auto new_sin = sin_value;
                update_givens_rotation(alpha_value, beta_value, new_cos,
                                       new_sin);
                alpha[col] = alpha_value;
                cos[col] = new_cos;
                sin[col] = new_sin;

                tau[col] = new_sin * new_sin * tau[col];
                const auto eta_value =
                    static_cast<arithmetic_type>(eta_next[col]);
                eta[col] = eta_value;
                eta_next[col] = -conj(new_sin) * eta_value;
            }
        },
        alpha.size[1], arithmetic<ValueType, 1>(alpha),
        arithmetic<ValueType, 1>(beta), arithmetic<ValueType, 1>(gamma),
        arithmetic<ValueType, 1>(delta), arithmetic<ValueType, 1>(cos_prev),
        arithmetic<ValueType, 1>(cos), arithmetic<ValueType, 1>(sin_prev),
        arithmetic<ValueType, 1>(sin), arithmetic<ValueType, 1>(eta_next),
        arithmetic<ValueType, 1>(eta), arithmetic<ValueType, 1>(tau),
        stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MINRES_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DefaultExecutor> exec,
            matrix::view::dense<ValueType> x, matrix::view::dense<ValueType> p,
            matrix::view::dense<const ValueType> p_prev,
            matrix::view::dense<ValueType> z,
            matrix::view::dense<const ValueType> z_tilde,
            matrix::view::dense<ValueType> q,
            matrix::view::dense<ValueType> q_prev,
            matrix::view::dense<ValueType> v,
            matrix::view::dense<const ValueType> alpha,
            matrix::view::dense<const ValueType> beta,
            matrix::view::dense<const ValueType> gamma,
            matrix::view::dense<const ValueType> delta,
            matrix::view::dense<const ValueType> cos,
            matrix::view::dense<const ValueType> eta,
            const array<stopping_status>& stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto x, auto p, auto p_prev, auto q,
                      auto q_prev, auto v, auto z, auto z_tilde, auto alpha,
                      auto beta, auto gamma, auto delta, auto cos, auto eta,
                      auto stop) {
            using arithmetic_type = arithmetic_type_t<decltype(q)>;
            if (!stop[col].has_stopped()) {
                p(row, col) =
                    safe_divide(z(row, col) - gamma[col] * p_prev(row, col) -
                                    delta[col] * p(row, col),
                                alpha[col]);
                x(row, col) = x(row, col) + cos[col] * eta[col] * p(row, col);

                q_prev(row, col) = v(row, col);
                // a copy, not the reference to the element that changes below
                const auto tmp = static_cast<arithmetic_type>(q(row, col));
                z(row, col) = safe_divide(z_tilde(row, col), beta[col]);
                q(row, col) = safe_divide(v(row, col), beta[col]);
                v(row, col) = tmp * beta[col];
            }
        },
        x.size, p.stride, arithmetic<ValueType>(x), arithmetic<ValueType>(p),
        arithmetic<ValueType>(p_prev), arithmetic<ValueType>(q),
        arithmetic<ValueType>(q_prev), arithmetic<ValueType>(v),
        arithmetic<ValueType>(z), arithmetic<ValueType>(z_tilde),
        arithmetic<ValueType, 1>(alpha), arithmetic<ValueType, 1>(beta),
        arithmetic<ValueType, 1>(gamma), arithmetic<ValueType, 1>(delta),
        arithmetic<ValueType, 1>(cos), arithmetic<ValueType, 1>(eta),
        stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MINRES_STEP_2_KERNEL);


}  // namespace minres
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
