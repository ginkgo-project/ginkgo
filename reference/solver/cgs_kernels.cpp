// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/cgs_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The CGS solver namespace.
 *
 * @ingroup cgs
 */
namespace cgs {


template <typename ValueType>
void initialize(
    std::shared_ptr<const ReferenceExecutor> exec,
    matrix::view::dense<const ValueType> b, matrix::view::dense<ValueType> r,
    matrix::view::dense<ValueType> r_tld, matrix::view::dense<ValueType> p,
    matrix::view::dense<ValueType> q, matrix::view::dense<ValueType> u,
    matrix::view::dense<ValueType> u_hat, matrix::view::dense<ValueType> v_hat,
    matrix::view::dense<ValueType> t, matrix::view::dense<ValueType> alpha,
    matrix::view::dense<ValueType> beta, matrix::view::dense<ValueType> gamma,
    matrix::view::dense<ValueType> rho_prev, matrix::view::dense<ValueType> rho,
    array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < b.size[1]; ++j) {
        rho(0, j) = zero<ValueType>();
        rho_prev(0, j) = one<ValueType>();
        alpha(0, j) = one<ValueType>();
        beta(0, j) = one<ValueType>();
        gamma(0, j) = one<ValueType>();
        stop_status->get_data()[j].reset();
    }
    for (size_type i = 0; i < b.size[0]; ++i) {
        for (size_type j = 0; j < b.size[1]; ++j) {
            r(i, j) = b(i, j);
            r_tld(i, j) = b(i, j);
            u(i, j) = u_hat(i, j) = p(i, j) = q(i, j) = v_hat(i, j) = t(i, j) =
                zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::view::dense<const ValueType> r,
            matrix::view::dense<ValueType> u, matrix::view::dense<ValueType> p,
            matrix::view::dense<const ValueType> q,
            matrix::view::dense<ValueType> beta,
            matrix::view::dense<const ValueType> rho,
            matrix::view::dense<const ValueType> rho_prev,
            const array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < p.size[1]; ++j) {
        if (stop_status->get_const_data()[j].has_stopped()) {
            continue;
        }
        if (is_nonzero(rho_prev(0, j))) {
            beta(0, j) = rho(0, j) / rho_prev(0, j);
        }
    }
    for (size_type i = 0; i < p.size[0]; ++i) {
        for (size_type j = 0; j < p.size[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            u(i, j) = r(i, j) + beta(0, j) * q(i, j);
            p(i, j) = u(i, j) + beta(0, j) * (q(i, j) + beta(0, j) * p(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::view::dense<const ValueType> u,
            matrix::view::dense<const ValueType> v_hat,
            matrix::view::dense<ValueType> q, matrix::view::dense<ValueType> t,
            matrix::view::dense<ValueType> alpha,
            matrix::view::dense<const ValueType> rho,
            matrix::view::dense<const ValueType> gamma,
            const array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < u.size[1]; ++j) {
        if (stop_status->get_const_data()[j].has_stopped()) {
            continue;
        }
        if (is_nonzero(gamma(0, j))) {
            alpha(0, j) = rho(0, j) / gamma(0, j);
        }
    }
    for (size_type i = 0; i < u.size[0]; ++i) {
        for (size_type j = 0; j < u.size[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            q(i, j) = u(i, j) - alpha(0, j) * v_hat(i, j);
            t(i, j) = u(i, j) + q(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_2_KERNEL);


template <typename ValueType>
void step_3(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::view::dense<const ValueType> t,
            matrix::view::dense<const ValueType> u_hat,
            matrix::view::dense<ValueType> r, matrix::view::dense<ValueType> x,
            matrix::view::dense<const ValueType> alpha,
            const array<stopping_status>* stop_status)
{
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            x(i, j) += alpha(0, j) * u_hat(i, j);
            r(i, j) -= alpha(0, j) * t(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_3_KERNEL);


}  // namespace cgs
}  // namespace reference
}  // namespace kernels
}  // namespace gko
