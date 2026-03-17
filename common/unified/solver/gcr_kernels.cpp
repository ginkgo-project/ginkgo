// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
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
                matrix::view::dense<const ValueType> b,
                matrix::view::dense<ValueType> residual,
                stopping_status* stop_status)
{
    if (b.size) {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto b, auto residual,
                          auto stop) {
                if (row == 0) {
                    stop[col].reset();
                }
                residual(row, col) = b(row, col);
            },
            b.size, b, residual, stop_status);
    } else {
        run_kernel(
            exec, [] GKO_KERNEL(auto col, auto stop) { stop[col].reset(); },
            b.size[1], stop_status);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GCR_INITIALIZE_KERNEL);


template <typename ValueType>
void restart(std::shared_ptr<const DefaultExecutor> exec,
             matrix::view::dense<const ValueType> residual,
             matrix::view::dense<const ValueType> A_residual,
             matrix::view::dense<ValueType> p_bases,
             matrix::view::dense<ValueType> Ap_bases,
             size_type* final_iter_nums)
{
    if (residual.size) {
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
            residual.size, residual.stride, default_stride(residual),
            default_stride(A_residual), p_bases, Ap_bases, final_iter_nums);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto col, auto final_iter_nums) {
                final_iter_nums[col] = 0;
            },
            residual.size[1], final_iter_nums);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GCR_RESTART_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            matrix::view::dense<ValueType> x,
            matrix::view::dense<ValueType> residual,
            matrix::view::dense<const ValueType> p,
            matrix::view::dense<const ValueType> Ap,
            matrix::view::dense<const remove_complex<ValueType>> Ap_norm,
            matrix::view::dense<const ValueType> rAp,
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
        x.size, p.stride, x, residual, p, Ap, Ap_norm, rAp, stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GCR_STEP_1_KERNEL);

}  // namespace gcr
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
