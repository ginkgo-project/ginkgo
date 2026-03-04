// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/pipe_cg_kernels.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The PIPE_CG solver namespace.
 *
 * @ingroup pipe_cg
 */
namespace pipe_cg {


template <typename ValueType>
void initialize_1(std::shared_ptr<const ReferenceExecutor> exec,
                  matrix::view::dense<const ValueType> b,
                  matrix::view::dense<ValueType> r,
                  matrix::view::dense<ValueType> prev_rho,
                  array<stopping_status>& stop_status)
{
    for (size_type j = 0; j < b.size[1]; ++j) {
        prev_rho(0, j) = one<ValueType>();
        stop_status.get_data()[j].reset();
    }
    for (size_type i = 0; i < b.size[0]; ++i) {
        for (size_type j = 0; j < b.size[1]; ++j) {
            r(i, j) = b(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_INITIALIZE_1_KERNEL);


template <typename ValueType>
void initialize_2(std::shared_ptr<const ReferenceExecutor> exec,
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
    for (size_type j = 0; j < p.size[1]; ++j) {
        // beta = delta
        beta(0, j) = delta(0, j);
    }
    for (size_type i = 0; i < p.size[0]; ++i) {
        // p = z
        // q = w
        // f = m
        // g = n
        for (size_type j = 0; j < p.size[1]; ++j) {
            p(i, j) = z(i, j);
            q(i, j) = w(i, j);
            f(i, j) = m(i, j);
            g(i, j) = n(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_INITIALIZE_2_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const ReferenceExecutor> exec,
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
    for (size_type i = 0; i < p.size[0]; ++i) {
        for (size_type j = 0; j < p.size[1]; ++j) {
            if (stop_status.get_const_data()[j].has_stopped()) {
                continue;
            }
            if (is_nonzero(beta(0, j))) {
                auto tmp = rho(0, j) / beta(0, j);
                x(i, j) += tmp * p(i, j);
                r(i, j) -= tmp * q(i, j);
                z1(i, j) -= tmp * f(i, j);
                z2(i, j) = z1(i, j);
                w(i, j) -= tmp * g(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const ReferenceExecutor> exec,
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
    for (size_type j = 0; j < p.size[1]; ++j) {
        if (stop_status.get_const_data()[j].has_stopped()) {
            continue;
        }
        if (is_nonzero(prev_rho(0, j))) {
            auto tmp = rho(0, j) / prev_rho(0, j);
            auto abs_tmp = abs(tmp);
            beta(0, j) = delta(0, j) - abs_tmp * abs_tmp * beta(0, j);
            if (is_zero(beta(0, j))) {
                beta(0, j) = delta(0, j);
            }

            for (size_type i = 0; i < p.size[0]; ++i) {
                p(i, j) = z(i, j) + tmp * p(i, j);
                q(i, j) = w(i, j) + tmp * q(i, j);
                f(i, j) = m(i, j) + tmp * f(i, j);
                g(i, j) = n(i, j) + tmp * g(i, j);
            }
        } else {
            beta(0, j) = delta(0, j);
            for (size_type i = 0; i < p.size[0]; ++i) {
                p(i, j) = z(i, j);
                q(i, j) = w(i, j);
                f(i, j) = m(i, j);
                g(i, j) = n(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_STEP_2_KERNEL);


}  // namespace pipe_cg
}  // namespace reference
}  // namespace kernels
}  // namespace gko
