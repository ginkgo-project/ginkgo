// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/stop/residual_norm_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/base/array_access.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Residual norm stopping criterion namespace.
 * @ref resnorm
 * @ingroup resnorm
 */
namespace residual_norm {


template <typename ValueType>
void residual_norm(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::Dense<ValueType>* tau,
                   const matrix::Dense<ValueType>* orig_tau,
                   ValueType rel_residual_goal, uint8 stoppingId,
                   bool setFinalized, array<stopping_status>* stop_status,
                   array<bool>* device_storage, bool* all_converged,
                   bool* one_changed)
{
    static_assert(is_complex_s<ValueType>::value == false,
                  "ValueType must not be complex in this function!");
    auto device_storage_val = device_storage->get_data();
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{1}, [=](sycl::id<1>) {
            device_storage_val[0] = true;
            device_storage_val[1] = false;
        });
    });

    auto orig_tau_val = orig_tau->get_const_values();
    auto tau_val = tau->get_const_values();
    auto stop_status_val = stop_status->get_data();
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<1>{tau->get_size()[1]}, [=](sycl::id<1> idx_id) {
                const auto tidx = idx_id[0];
                if (tau_val[tidx] <= rel_residual_goal * orig_tau_val[tidx]) {
                    stop_status_val[tidx].converge(stoppingId, setFinalized);
                    device_storage_val[1] = true;
                }
                // because only false is written to all_converged, write
                // conflicts should not cause any problem
                else if (!stop_status_val[tidx].has_stopped()) {
                    device_storage_val[0] = false;
                }
            });
    });

    /* Represents all_converged, one_changed */
    *all_converged = get_element(*device_storage, 0);
    *one_changed = get_element(*device_storage, 1);
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
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::Dense<ValueType>* tau,
    const matrix::Dense<remove_complex<ValueType>>* orig_tau,
    remove_complex<ValueType> rel_residual_goal, uint8 stoppingId,
    bool setFinalized, array<stopping_status>* stop_status,
    array<bool>* device_storage, bool* all_converged, bool* one_changed)
{
    auto device_storage_val = device_storage->get_data();
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{1}, [=](sycl::id<1>) {
            device_storage_val[0] = true;
            device_storage_val[1] = false;
        });
    });

    auto orig_tau_val = orig_tau->get_const_values();
    auto tau_val = tau->get_const_values();
    auto stop_status_val = stop_status->get_data();
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<1>{tau->get_size()[1]}, [=](sycl::id<1> idx_id) {
                const auto tidx = idx_id[0];
                if (std::sqrt(std::abs(tau_val[tidx])) <=
                    rel_residual_goal * orig_tau_val[tidx]) {
                    stop_status_val[tidx].converge(stoppingId, setFinalized);
                    device_storage_val[1] = true;
                }
                // because only false is written to all_converged, write
                // conflicts should not cause any problem
                else if (!stop_status_val[tidx].has_stopped()) {
                    device_storage_val[0] = false;
                }
            });
    });

    /* Represents all_converged, one_changed */
    *all_converged = get_element(*device_storage, 0);
    *one_changed = get_element(*device_storage, 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IMPLICIT_RESIDUAL_NORM_KERNEL);


}  // namespace implicit_residual_norm
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
