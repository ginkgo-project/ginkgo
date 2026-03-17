// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/cg_kernels.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The CG solver namespace.
 *
 * @ingroup cg
 */
namespace cg {


template <typename ValueType>
void initialize(
    std::shared_ptr<const ReferenceExecutor> exec,
    matrix::view::dense<const ValueType> b, matrix::view::dense<ValueType> r,
    matrix::view::dense<ValueType> z, matrix::view::dense<ValueType> p,
    matrix::view::dense<ValueType> q, matrix::view::dense<ValueType> prev_rho,
    matrix::view::dense<ValueType> rho, array<stopping_status>& stop_status)
{
    for (size_type j = 0; j < b.size[1]; ++j) {
        rho(0, j) = zero<ValueType>();
        prev_rho(0, j) = one<ValueType>();
        stop_status.get_data()[j].reset();
    }
    for (size_type i = 0; i < b.size[0]; ++i) {
        for (size_type j = 0; j < b.size[1]; ++j) {
            r(i, j) = b(i, j);
            z(i, j) = p(i, j) = q(i, j) = zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::view::dense<ValueType> p,
            matrix::view::dense<const ValueType> z,
            matrix::view::dense<const ValueType> rho,
            matrix::view::dense<const ValueType> prev_rho,
            const array<stopping_status>& stop_status)
{
    for (size_type i = 0; i < p.size[0]; ++i) {
        for (size_type j = 0; j < p.size[1]; ++j) {
            if (stop_status.get_const_data()[j].has_stopped()) {
                continue;
            }
            if (is_zero(prev_rho(0, j))) {
                p(i, j) = z(i, j);
            } else {
                auto tmp = rho(0, j) / prev_rho(0, j);
                p(i, j) = z(i, j) + tmp * p(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::view::dense<ValueType> x, matrix::view::dense<ValueType> r,
            matrix::view::dense<const ValueType> p,
            matrix::view::dense<const ValueType> q,
            matrix::view::dense<const ValueType> beta,
            matrix::view::dense<const ValueType> rho,
            const array<stopping_status>& stop_status)
{
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            if (stop_status.get_const_data()[j].has_stopped()) {
                continue;
            }
            if (is_nonzero(beta(0, j))) {
                auto tmp = rho(0, j) / beta(0, j);
                x(i, j) += tmp * p(i, j);
                r(i, j) -= tmp * q(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_STEP_2_KERNEL);


}  // namespace cg
}  // namespace reference
}  // namespace kernels
}  // namespace gko
