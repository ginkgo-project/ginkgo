// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/bicgstab_kernels.hpp"

#include <algorithm>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The BiCGSTAB solver namespace.
 *
 * @ingroup bicgstab
 */
namespace bicgstab {


template <typename ValueType>
void initialize(
    std::shared_ptr<const ReferenceExecutor> exec,
    matrix::view::dense<const ValueType> b, matrix::view::dense<ValueType> r,
    matrix::view::dense<ValueType> rr, matrix::view::dense<ValueType> y,
    matrix::view::dense<ValueType> s, matrix::view::dense<ValueType> t,
    matrix::view::dense<ValueType> z, matrix::view::dense<ValueType> v,
    matrix::view::dense<ValueType> p, matrix::view::dense<ValueType> prev_rho,
    matrix::view::dense<ValueType> rho, matrix::view::dense<ValueType> alpha,
    matrix::view::dense<ValueType> beta, matrix::view::dense<ValueType> gamma,
    matrix::view::dense<ValueType> omega, array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < b.size[1]; ++j) {
        rho(0, j) = one<ValueType>();
        prev_rho(0, j) = one<ValueType>();
        alpha(0, j) = one<ValueType>();
        beta(0, j) = one<ValueType>();
        gamma(0, j) = one<ValueType>();
        omega(0, j) = one<ValueType>();
        stop_status->get_data()[j].reset();
    }
    for (size_type i = 0; i < b.size[0]; ++i) {
        for (size_type j = 0; j < b.size[1]; ++j) {
            r(i, j) = b(i, j);
            rr(i, j) = zero<ValueType>();
            z(i, j) = zero<ValueType>();
            v(i, j) = zero<ValueType>();
            s(i, j) = zero<ValueType>();
            t(i, j) = zero<ValueType>();
            y(i, j) = zero<ValueType>();
            p(i, j) = zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::view::dense<const ValueType> r,
            matrix::view::dense<ValueType> p,
            matrix::view::dense<const ValueType> v,
            matrix::view::dense<const ValueType> rho,
            matrix::view::dense<const ValueType> prev_rho,
            matrix::view::dense<const ValueType> alpha,
            matrix::view::dense<const ValueType> omega,
            const array<stopping_status>* stop_status)
{
    for (size_type i = 0; i < p.size[0]; ++i) {
        for (size_type j = 0; j < p.size[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            if (is_nonzero(prev_rho(0, j) * omega(0, j))) {
                const auto tmp =
                    rho(0, j) / prev_rho(0, j) * alpha(0, j) / omega(0, j);
                p(i, j) = r(i, j) + tmp * (p(i, j) - omega(0, j) * v(i, j));
            } else {
                p(i, j) = r(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::view::dense<const ValueType> r,
            matrix::view::dense<ValueType> s,
            matrix::view::dense<const ValueType> v,
            matrix::view::dense<const ValueType> rho,
            matrix::view::dense<ValueType> alpha,
            matrix::view::dense<const ValueType> beta,
            const array<stopping_status>* stop_status)
{
    for (size_type i = 0; i < s.size[0]; ++i) {
        for (size_type j = 0; j < s.size[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            if (is_nonzero(beta(0, j))) {
                alpha(0, j) = rho(0, j) / beta(0, j);
                s(i, j) = r(i, j) - alpha(0, j) * v(i, j);
            } else {
                alpha(0, j) = zero<ValueType>();
                s(i, j) = r(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_2_KERNEL);


template <typename ValueType>
void step_3(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::view::dense<ValueType> x, matrix::view::dense<ValueType> r,
            matrix::view::dense<const ValueType> s,
            matrix::view::dense<const ValueType> t,
            matrix::view::dense<const ValueType> y,
            matrix::view::dense<const ValueType> z,
            matrix::view::dense<const ValueType> alpha,
            matrix::view::dense<const ValueType> beta,
            matrix::view::dense<const ValueType> gamma,
            matrix::view::dense<ValueType> omega,
            const array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < x.size[1]; ++j) {
        if (stop_status->get_const_data()[j].has_stopped()) {
            continue;
        }
        if (is_nonzero(beta(0, j))) {
            omega(0, j) = gamma(0, j) / beta(0, j);
        } else {
            omega(0, j) = zero<ValueType>();
        }
    }
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            x(i, j) += alpha(0, j) * y(i, j) + omega(0, j) * z(i, j);
            r(i, j) = s(i, j) - omega(0, j) * t(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_3_KERNEL);


template <typename ValueType>
void finalize(std::shared_ptr<const ReferenceExecutor> exec,
              matrix::view::dense<ValueType> x,
              matrix::view::dense<const ValueType> y,
              matrix::view::dense<const ValueType> alpha,
              array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < x.size[1]; ++j) {
        if (stop_status->get_const_data()[j].has_stopped() &&
            !stop_status->get_const_data()[j].is_finalized()) {
            for (size_type i = 0; i < x.size[0]; ++i) {
                x(i, j) += alpha(0, j) * y(i, j);
                stop_status->get_data()[j].finalize();
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_FINALIZE_KERNEL);


}  // namespace bicgstab
}  // namespace reference
}  // namespace kernels
}  // namespace gko
