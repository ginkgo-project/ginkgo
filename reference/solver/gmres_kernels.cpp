// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/gmres_kernels.hpp"


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The GMRES solver namespace.
 *
 * @ingroup gmres
 */
namespace gmres {


template <typename ValueType>
void restart(std::shared_ptr<const ReferenceExecutor> exec,
             const matrix::Dense<ValueType>* residual,
             const matrix::Dense<remove_complex<ValueType>>* residual_norm,
             matrix::Dense<ValueType>* residual_norm_collection,
             matrix::Dense<ValueType>* krylov_bases, size_type* final_iter_nums)
{
    for (size_type j = 0; j < residual->get_size()[1]; ++j) {
        residual_norm_collection->at(0, j) = residual_norm->at(0, j);
        for (size_type i = 0; i < residual->get_size()[0]; ++i) {
            krylov_bases->at(i, j) =
                residual->at(i, j) / residual_norm->at(0, j);
        }
        final_iter_nums[j] = 0;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_RESTART_KERNEL);


template <typename ValueType>
void multi_axpy(std::shared_ptr<const ReferenceExecutor> exec,
                const matrix::Dense<ValueType>* krylov_bases,
                const matrix::Dense<ValueType>* y,
                matrix::Dense<ValueType>* before_preconditioner,
                const size_type* final_iter_nums, stopping_status* stop_status)
{
    const auto krylov_bases_rowoffset = before_preconditioner->get_size()[0];
    for (size_type k = 0; k < before_preconditioner->get_size()[1]; ++k) {
        if (stop_status[k].is_finalized()) {
            continue;
        }
        for (size_type i = 0; i < before_preconditioner->get_size()[0]; ++i) {
            before_preconditioner->at(i, k) = zero<ValueType>();
            for (size_type j = 0; j < final_iter_nums[k]; ++j) {
                before_preconditioner->at(i, k) +=
                    krylov_bases->at(i + j * krylov_bases_rowoffset, k) *
                    y->at(j, k);
            }
        }
        if (stop_status[k].has_stopped()) {
            stop_status[k].finalize();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_MULTI_AXPY_KERNEL);


}  // namespace gmres
}  // namespace reference
}  // namespace kernels
}  // namespace gko
