// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/minres_kernels.hpp"

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
    const matrix::Dense<ValueType>* r, matrix::Dense<ValueType>* z,
    matrix::Dense<ValueType>* p, matrix::Dense<ValueType>* p_prev,
    matrix::Dense<ValueType>* q, matrix::Dense<ValueType>* q_prev,
    matrix::Dense<ValueType>* q_tilde, matrix::Dense<ValueType>* beta,
    matrix::Dense<ValueType>* gamma, matrix::Dense<ValueType>* delta,
    matrix::Dense<ValueType>* cos_prev, matrix::Dense<ValueType>* cos,
    matrix::Dense<ValueType>* sin_prev, matrix::Dense<ValueType>* sin,
    matrix::Dense<ValueType>* eta_next, matrix::Dense<ValueType>* eta,
    array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < r->get_size()[1]; ++j) {
        delta->at(j) = gamma->at(j) = cos_prev->at(j) = sin_prev->at(j) =
            sin->at(j) = zero<ValueType>();
        cos->at(j) = one<ValueType>();
        eta_next->at(j) = eta->at(j) = beta->at(j) = sqrt(beta->at(j));
        stop_status->get_data()[j].reset();
    }
    for (size_type i = 0; i < r->get_size()[0]; ++i) {
        for (size_type j = 0; j < r->get_size()[1]; ++j) {
            q->at(i, j) = safe_divide(r->at(i, j), beta->at(j));
            z->at(i, j) = safe_divide(z->at(i, j), beta->at(j));
            p->at(i, j) = p_prev->at(i, j) = q_prev->at(i, j) =
                q_tilde->at(i, j) = zero<ValueType>();
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
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            matrix::Dense<ValueType>* alpha, matrix::Dense<ValueType>* beta,
            matrix::Dense<ValueType>* gamma, matrix::Dense<ValueType>* delta,
            matrix::Dense<ValueType>* cos_prev, matrix::Dense<ValueType>* cos,
            matrix::Dense<ValueType>* sin_prev, matrix::Dense<ValueType>* sin,
            matrix::Dense<ValueType>* eta, matrix::Dense<ValueType>* eta_next,
            matrix::Dense<ValueType>* tau,
            const array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < alpha->get_size()[1]; ++j) {
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
    for (size_type i = 0; i < x->get_size()[0]; ++i) {
        for (size_type j = 0; j < x->get_size()[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            p->at(i, j) =
                safe_divide(z->at(i, j) - gamma->at(j) * p_prev->at(i, j) -
                                delta->at(j) * p->at(i, j),
                            alpha->at(j));
            x->at(i, j) = x->at(i, j) + cos->at(j) * eta->at(j) * p->at(i, j);

            q_prev->at(i, j) = v->at(i, j);
            const auto tmp = q->at(i, j);
            q->at(i, j) = safe_divide(v->at(i, j), beta->at(j));
            v->at(i, j) = tmp * beta->at(j);
            z->at(i, j) = safe_divide(z_tilde->at(i, j), beta->at(j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MINRES_STEP_2_KERNEL);


}  // namespace minres
}  // namespace reference
}  // namespace kernels
}  // namespace gko
