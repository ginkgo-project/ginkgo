// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/minres_kernels.hpp"

#include <utility>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Minres solver namespace.
 *
 * @ingroup minres
 */
namespace minres {


template <typename ValueType>
void initialize(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::dense<const ValueType> r, matrix::view::dense<ValueType> z,
    matrix::view::dense<ValueType> p, matrix::view::dense<ValueType> p_prev,
    matrix::view::dense<ValueType> q, matrix::view::dense<ValueType> q_prev,
    matrix::view::dense<ValueType> q_tilde, matrix::view::dense<ValueType> beta,
    matrix::view::dense<ValueType> gamma, matrix::view::dense<ValueType> delta,
    matrix::view::dense<ValueType> cos_prev, matrix::view::dense<ValueType> cos,
    matrix::view::dense<ValueType> sin_prev, matrix::view::dense<ValueType> sin,
    matrix::view::dense<ValueType> eta_next, matrix::view::dense<ValueType> eta,
    array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < r.size[1]; ++j) {
        delta->at(j) = gamma->at(j) = cos_prev->at(j) = sin_prev->at(j) =
            sin->at(j) = zero<ValueType>();
        cos->at(j) = one<ValueType>();
        eta_next->at(j) = eta->at(j) = beta->at(j) = sqrt(beta->at(j));
        stop_status->get_data()[j].reset();
    }
    for (size_type i = 0; i < r.size[0]; ++i) {
        for (size_type j = 0; j < r.size[1]; ++j) {
            q(i, j) = safe_divide(r(i, j), beta->at(j));
            z(i, j) = safe_divide(z(i, j), beta->at(j));
            p(i, j) = p_prev(i, j) = q_prev(i, j) = q_tilde(i, j) =
                zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MINRES_INITIALIZE_KERNEL);


template <typename ValueType>
void update_givens_rotation(ValueType& alpha, const ValueType& beta,
                            ValueType& cos, ValueType& sin)
{
    if (alpha == zero<ValueType>()) {
        cos = zero<ValueType>();
        sin = one<ValueType>();
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
    const array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < alpha.size[1]; ++j) {
        if (stop_status->get_const_data()[j].has_stopped()) {
            continue;
        }
        beta->at(j) = sqrt(beta->at(j));
        delta->at(j) = sin_prev->at(j) * gamma->at(j);
        auto tmp_d = gamma->at(j);
        auto tmp_a = alpha->at(j);
        gamma->at(j) =
            cos_prev->at(j) * cos->at(j) * tmp_d + sin->at(j) * tmp_a;
        alpha->at(j) =
            -conj(sin->at(j)) * cos_prev->at(j) * tmp_d + cos->at(j) * tmp_a;

        std::swap(cos->at(j), cos_prev->at(j));
        std::swap(sin->at(j), sin_prev->at(j));
        update_givens_rotation(alpha->at(j), beta->at(j), cos->at(j),
                               sin->at(j));

        tau->at(j) = sin->at(j) * sin->at(j) * tau->at(j);
        eta->at(j) = eta_next->at(j);
        eta_next->at(j) = -conj(sin->at(j)) * eta->at(j);
    }
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
            const array<stopping_status>* stop_status)
{
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            p(i, j) = safe_divide(
                z(i, j) - gamma->at(j) * p_prev(i, j) - delta->at(j) * p(i, j),
                alpha->at(j));
            x(i, j) = x(i, j) + cos->at(j) * eta->at(j) * p(i, j);

            q_prev(i, j) = v(i, j);
            const auto tmp = q(i, j);
            q(i, j) = safe_divide(v(i, j), beta->at(j));
            v(i, j) = tmp * beta->at(j);
            z(i, j) = safe_divide(z_tilde(i, j), beta->at(j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MINRES_STEP_2_KERNEL);


}  // namespace minres
}  // namespace reference
}  // namespace kernels
}  // namespace gko
