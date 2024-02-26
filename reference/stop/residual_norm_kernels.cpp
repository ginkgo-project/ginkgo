// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/stop/residual_norm_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Residual norm stopping criterion.
 * @ref resnorm
 * @ingroup resnorm
 */
namespace residual_norm {


template <typename ValueType>
void residual_norm(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Dense<ValueType>* tau,
                   const matrix::Dense<ValueType>* orig_tau,
                   ValueType rel_residual_goal, uint8 stoppingId,
                   bool setFinalized, array<stopping_status>* stop_status,
                   array<bool>* device_storage, bool* all_converged,
                   bool* one_changed)
{
    static_assert(is_complex_s<ValueType>::value == false,
                  "ValueType must not be complex in this function!");
    *all_converged = true;
    *one_changed = false;
    for (size_type i = 0; i < tau->get_size()[1]; ++i) {
        if (tau->at(i) <= rel_residual_goal * orig_tau->at(i)) {
            stop_status->get_data()[i].converge(stoppingId, setFinalized);
            *one_changed = true;
        }
    }
    for (size_type i = 0; i < stop_status->get_size(); ++i) {
        if (!stop_status->get_const_data()[i].has_stopped()) {
            *all_converged = false;
            break;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(
    GKO_DECLARE_RESIDUAL_NORM_KERNEL);


}  // namespace residual_norm


/**
 * @brief The Implicit Residual norm stopping criterion.
 * @ref implicit_resnorm
 * @ingroup resnorm
 */
namespace implicit_residual_norm {


template <typename ValueType>
void implicit_residual_norm(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Dense<ValueType>* tau,
    const matrix::Dense<remove_complex<ValueType>>* orig_tau,
    remove_complex<ValueType> rel_residual_goal, uint8 stoppingId,
    bool setFinalized, array<stopping_status>* stop_status,
    array<bool>* device_storage, bool* all_converged, bool* one_changed)
{
    *all_converged = true;
    *one_changed = false;
    for (size_type i = 0; i < tau->get_size()[1]; ++i) {
        if (sqrt(abs(tau->at(i))) <= rel_residual_goal * orig_tau->at(i)) {
            stop_status->get_data()[i].converge(stoppingId, setFinalized);
            *one_changed = true;
        }
    }
    for (size_type i = 0; i < stop_status->get_size(); ++i) {
        if (!stop_status->get_const_data()[i].has_stopped()) {
            *all_converged = false;
            break;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IMPLICIT_RESIDUAL_NORM_KERNEL);


}  // namespace implicit_residual_norm
}  // namespace reference
}  // namespace kernels
}  // namespace gko
