// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/gcr_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch_solver.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The GCR solver namespace.
 *
 * @ingroup grc
 */
namespace gcr {


template <typename ValueType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Dense<ValueType>* b,
                matrix::Dense<ValueType>* residual,
                stopping_status* stop_status)
{
    if (b->get_size()) {
        run_kernel_solver(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto b, auto residual,
                          auto stop) {
                if (row == 0) {
                    stop[col].reset();
                }
                residual(row, col) = b(row, col);
            },
            b->get_size(), b->get_stride(), default_stride(b),
            default_stride(residual), stop_status);
    } else {
        run_kernel(
            exec, [] GKO_KERNEL(auto col, auto stop) { stop[col].reset(); },
            b->get_size()[1], stop_status);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GCR_INITIALIZE_KERNEL);


template <typename ValueType>
void restart(std::shared_ptr<const DefaultExecutor> exec,
             const matrix::Dense<ValueType>* residual,
             const matrix::Dense<ValueType>* A_residual,
             matrix::Dense<ValueType>* p_bases,
             matrix::Dense<ValueType>* Ap_bases, size_type* final_iter_nums)
{
    if (residual->get_size()) {
        run_kernel_solver(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto residual, auto A_residual,
                          auto p_bases, auto Ap_bases, auto final_iter_nums) {
                if (row == 0) {
                    final_iter_nums[col] = 0;
                }
                p_bases(row, col) = residual(row, col);
                Ap_bases(row, col) = A_residual(row, col);
            },
            residual->get_size(), residual->get_stride(),
            default_stride(residual), default_stride(A_residual), p_bases,
            Ap_bases, final_iter_nums);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto col, auto final_iter_nums) {
                final_iter_nums[col] = 0;
            },
            residual->get_size()[1], final_iter_nums);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GCR_RESTART_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            matrix::Dense<ValueType>* x, matrix::Dense<ValueType>* residual,
            const matrix::Dense<ValueType>* p,
            const matrix::Dense<ValueType>* Ap,
            const matrix::Dense<remove_complex<ValueType>>* Ap_norm,
            const matrix::Dense<ValueType>* rAp,
            const stopping_status* stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto x, auto residual, auto p,
                      auto Ap, auto Ap_norm, auto rAp, auto stop) {
            if (!stop[col].has_stopped()) {
                auto tmp = rAp[col] / Ap_norm[col];
                x(row, col) += tmp * p(row, col);
                residual(row, col) -= tmp * Ap(row, col);
            }
        },
        x->get_size(), p->get_stride(), x, residual, p, Ap, Ap_norm, rAp,
        stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GCR_STEP_1_KERNEL);

}  // namespace gcr
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
