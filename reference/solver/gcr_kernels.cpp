// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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
                const matrix::Dense<ValueType>* b,
                matrix::Dense<ValueType>* residual,
                stopping_status* stop_status)
{
    for (size_type j = 0; j < b->get_size()[1]; ++j) {
        for (size_type i = 0; i < b->get_size()[0]; ++i) {
            residual->at(i, j) = b->at(i, j);
        }
        stop_status[j].reset();
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GCR_INITIALIZE_KERNEL);


template <typename ValueType>
void restart(std::shared_ptr<const ReferenceExecutor> exec,
             const matrix::Dense<ValueType>* residual,
             const matrix::Dense<ValueType>* A_residual,
             matrix::Dense<ValueType>* p_bases,
             matrix::Dense<ValueType>* Ap_bases, size_type* final_iter_nums)
{
    for (size_type j = 0; j < residual->get_size()[1]; ++j) {
        for (size_type i = 0; i < residual->get_size()[0]; ++i) {
            p_bases->at(i, j) = residual->at(i, j);
            Ap_bases->at(i, j) = A_residual->at(i, j);
        }
        final_iter_nums[j] = 0;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GCR_RESTART_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::Dense<ValueType>* x, matrix::Dense<ValueType>* residual,
            const matrix::Dense<ValueType>* p,
            const matrix::Dense<ValueType>* Ap,
            const matrix::Dense<remove_complex<ValueType>>* Ap_norm,
            const matrix::Dense<ValueType>* rAp,
            const stopping_status* stop_status)
{
    for (size_type i = 0; i < x->get_size()[0]; ++i) {
        for (size_type j = 0; j < x->get_size()[1]; ++j) {
            if (stop_status[j].has_stopped()) {
                continue;
            }
            if (Ap_norm->at(j) != zero<ValueType>()) {
                auto tmp = rAp->at(j) / Ap_norm->at(j);
                x->at(i, j) += tmp * p->at(i, j);
                residual->at(i, j) -= tmp * Ap->at(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GCR_STEP_1_KERNEL);


}  // namespace gcr
}  // namespace reference
}  // namespace kernels
}  // namespace gko
