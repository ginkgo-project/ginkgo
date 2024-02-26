// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/fcg_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch_solver.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The FCG solver namespace.
 *
 * @ingroup fcg
 */
namespace fcg {


template <typename ValueType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* r,
                matrix::Dense<ValueType>* z, matrix::Dense<ValueType>* p,
                matrix::Dense<ValueType>* q, matrix::Dense<ValueType>* t,
                matrix::Dense<ValueType>* prev_rho,
                matrix::Dense<ValueType>* rho, matrix::Dense<ValueType>* rho_t,
                array<stopping_status>* stop_status)
{
    if (b->get_size()) {
        run_kernel_solver(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto b, auto r, auto z, auto p,
                          auto q, auto t, auto prev_rho, auto rho, auto rho_t,
                          auto stop) {
                if (row == 0) {
                    rho[col] = zero(rho[col]);
                    prev_rho[col] = rho_t[col] = one(prev_rho[col]);
                    stop[col].reset();
                }
                t(row, col) = r(row, col) = b(row, col);
                z(row, col) = p(row, col) = q(row, col) = zero(z(row, col));
            },
            b->get_size(), b->get_stride(), default_stride(b),
            default_stride(r), default_stride(z), default_stride(p),
            default_stride(q), default_stride(t), row_vector(prev_rho),
            row_vector(rho), row_vector(rho_t), *stop_status);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto col, auto prev_rho, auto rho, auto rho_t,
                          auto stop) {
                rho[col] = zero(rho[col]);
                prev_rho[col] = rho_t[col] = one(prev_rho[col]);
                stop[col].reset();
            },
            b->get_size()[1], row_vector(prev_rho), row_vector(rho),
            row_vector(rho_t), *stop_status);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FCG_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            matrix::Dense<ValueType>* p, const matrix::Dense<ValueType>* z,
            const matrix::Dense<ValueType>* rho_t,
            const matrix::Dense<ValueType>* prev_rho,
            const array<stopping_status>* stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto p, auto z, auto rho_t,
                      auto prev_rho, auto stop) {
            if (!stop[col].has_stopped()) {
                auto tmp = safe_divide(rho_t[col], prev_rho[col]);
                p(row, col) = z(row, col) + tmp * p(row, col);
            }
        },
        p->get_size(), p->get_stride(), default_stride(p), default_stride(z),
        row_vector(rho_t), row_vector(prev_rho), *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FCG_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DefaultExecutor> exec,
            matrix::Dense<ValueType>* x, matrix::Dense<ValueType>* r,
            matrix::Dense<ValueType>* t, const matrix::Dense<ValueType>* p,
            const matrix::Dense<ValueType>* q,
            const matrix::Dense<ValueType>* beta,
            const matrix::Dense<ValueType>* rho,
            const array<stopping_status>* stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto x, auto r, auto t, auto p,
                      auto q, auto beta, auto rho, auto stop) {
            if (!stop[col].has_stopped() && is_nonzero(beta[col])) {
                auto tmp = rho[col] / beta[col];
                auto prev_r = r(row, col);
                x(row, col) += tmp * p(row, col);
                r(row, col) -= tmp * q(row, col);
                t(row, col) = r(row, col) - prev_r;
            }
        },
        x->get_size(), r->get_stride(), x, default_stride(r), default_stride(t),
        default_stride(p), default_stride(q), row_vector(beta), row_vector(rho),
        *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FCG_STEP_2_KERNEL);


}  // namespace fcg
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
