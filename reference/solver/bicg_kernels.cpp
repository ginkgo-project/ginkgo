// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/bicg_kernels.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The BICG solver namespace.
 *
 * @ingroup bicg
 */
namespace bicg {


template <typename ValueType>
void initialize(
    std::shared_ptr<const ReferenceExecutor> exec,
    matrix::view::dense<const ValueType> b, matrix::view::dense<ValueType> r,
    matrix::view::dense<ValueType> z, matrix::view::dense<ValueType> p,
    matrix::view::dense<ValueType> q, matrix::view::dense<ValueType> prev_rho,
    matrix::view::dense<ValueType> rho, matrix::view::dense<ValueType> r2,
    matrix::view::dense<ValueType> z2, matrix::view::dense<ValueType> p2,
    matrix::view::dense<ValueType> q2, array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < b.size[1]; ++j) {
        rho->at(j) = zero<ValueType>();
        prev_rho->at(j) = one<ValueType>();
        stop_status->get_data()[j].reset();
    }
    for (size_type i = 0; i < b.size[0]; ++i) {
        for (size_type j = 0; j < b.size[1]; ++j) {
            r(i, j) = b(i, j);
            r2(i, j) = b(i, j);
            z(i, j) = p(i, j) = q(i, j) = zero<ValueType>();
            z2(i, j) = p2(i, j) = q2(i, j) = zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICG_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::view::dense<ValueType> p,
            matrix::view::dense<const ValueType> z,
            matrix::view::dense<ValueType> p2,
            matrix::view::dense<const ValueType> z2,
            matrix::view::dense<const ValueType> rho,
            matrix::view::dense<const ValueType> prev_rho,
            const array<stopping_status>* stop_status)
{
    for (size_type i = 0; i < p.size[0]; ++i) {
        for (size_type j = 0; j < p.size[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            if (is_zero(prev_rho->at(j))) {
                p(i, j) = z(i, j);
                p2(i, j) = z2(i, j);
            } else {
                auto tmp = rho->at(j) / prev_rho->at(j);
                p(i, j) = z(i, j) + tmp * p(i, j);
                p2(i, j) = z2(i, j) + tmp * p2(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICG_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::view::dense<ValueType> x, matrix::view::dense<ValueType> r,
            matrix::view::dense<ValueType> r2,
            matrix::view::dense<const ValueType> p,
            matrix::view::dense<const ValueType> q,
            matrix::view::dense<const ValueType> q2,
            matrix::view::dense<const ValueType> beta,
            matrix::view::dense<const ValueType> rho,
            const array<stopping_status>* stop_status)
{
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            if (is_nonzero(beta->at(j))) {
                auto tmp = rho->at(j) / beta->at(j);
                x(i, j) += tmp * p(i, j);
                r(i, j) -= tmp * q(i, j);
                r2(i, j) -= tmp * q2(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICG_STEP_2_KERNEL);


}  // namespace bicg
}  // namespace reference
}  // namespace kernels
}  // namespace gko
