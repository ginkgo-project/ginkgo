// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/gcr_kernels.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/solver/gcr.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The GCR solver namespace.
 *
 * @ingroup gcr
 */
namespace gcr {


template <typename ValueType>
void initialize(std::shared_ptr<const ReferenceExecutor> exec,
                matrix::view::dense<const ValueType> b,
                matrix::view::dense<ValueType> residual,
                stopping_status* stop_status)
{
    for (size_type j = 0; j < b.size[1]; ++j) {
        for (size_type i = 0; i < b.size[0]; ++i) {
            residual(i, j) = b(i, j);
        }
        stop_status[j].reset();
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GCR_INITIALIZE_KERNEL);


template <typename ValueType>
void restart(std::shared_ptr<const ReferenceExecutor> exec,
             matrix::view::dense<const ValueType> residual,
             matrix::view::dense<const ValueType> A_residual,
             matrix::view::dense<ValueType> p_bases,
             matrix::view::dense<ValueType> Ap_bases,
             size_type* final_iter_nums)
{
    for (size_type j = 0; j < residual.size[1]; ++j) {
        for (size_type i = 0; i < residual.size[0]; ++i) {
            p_bases(i, j) = residual(i, j);
            Ap_bases(i, j) = A_residual(i, j);
        }
        final_iter_nums[j] = 0;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GCR_RESTART_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::view::dense<ValueType> x,
            matrix::view::dense<ValueType> residual,
            matrix::view::dense<const ValueType> p,
            matrix::view::dense<const ValueType> Ap,
            matrix::view::dense<const remove_complex<ValueType>> Ap_norm,
            matrix::view::dense<const ValueType> rAp,
            const stopping_status* stop_status)
{
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            if (stop_status[j].has_stopped()) {
                continue;
            }
            if (Ap_norm(0, j) != zero<ValueType>()) {
                auto tmp = rAp(0, j) / Ap_norm(0, j);
                x(i, j) += tmp * p(i, j);
                residual(i, j) -= tmp * Ap(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GCR_STEP_1_KERNEL);


}  // namespace gcr
}  // namespace reference
}  // namespace kernels
}  // namespace gko
