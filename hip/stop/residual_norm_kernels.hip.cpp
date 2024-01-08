// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/stop/residual_norm_kernels.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/base/array_access.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The Residual norm stopping criterion namespace.
 * @ref resnorm
 * @ingroup resnorm
 */
namespace residual_norm {


constexpr int default_block_size = 512;


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void residual_norm_kernel(
    size_type num_cols, ValueType rel_residual_goal,
    const ValueType* __restrict__ tau, const ValueType* __restrict__ orig_tau,
    uint8 stoppingId, bool setFinalized,
    stopping_status* __restrict__ stop_status,
    bool* __restrict__ device_storage)
{
    const auto tidx = thread::get_thread_id_flat();
    if (tidx < num_cols) {
        if (tau[tidx] <= rel_residual_goal * orig_tau[tidx]) {
            stop_status[tidx].converge(stoppingId, setFinalized);
            device_storage[1] = true;
        }
        // because only false is written to all_converged, write conflicts
        // should not cause any problem
        else if (!stop_status[tidx].has_stopped()) {
            device_storage[0] = false;
        }
    }
}


__global__ __launch_bounds__(1) void init_kernel(
    bool* __restrict__ device_storage)
{
    device_storage[0] = true;
    device_storage[1] = false;
}


template <typename ValueType>
void residual_norm(std::shared_ptr<const HipExecutor> exec,
                   const matrix::Dense<ValueType>* tau,
                   const matrix::Dense<ValueType>* orig_tau,
                   ValueType rel_residual_goal, uint8 stoppingId,
                   bool setFinalized, array<stopping_status>* stop_status,
                   array<bool>* device_storage, bool* all_converged,
                   bool* one_changed)
{
    static_assert(is_complex_s<ValueType>::value == false,
                  "ValueType must not be complex in this function!");
    init_kernel<<<1, 1, 0, exec->get_stream()>>>(
        as_device_type(device_storage->get_data()));

    const auto block_size = default_block_size;
    const auto grid_size = ceildiv(tau->get_size()[1], block_size);

    if (grid_size > 0) {
        residual_norm_kernel<<<grid_size, block_size, 0, exec->get_stream()>>>(
            tau->get_size()[1], as_device_type(rel_residual_goal),
            as_device_type(tau->get_const_values()),
            as_device_type(orig_tau->get_const_values()), stoppingId,
            setFinalized, as_device_type(stop_status->get_data()),
            as_device_type(device_storage->get_data()));
    }

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


constexpr int default_block_size = 512;


template <typename ValueType>
__global__
__launch_bounds__(default_block_size) void implicit_residual_norm_kernel(
    size_type num_cols, remove_complex<ValueType> rel_residual_goal,
    const ValueType* __restrict__ tau,
    const remove_complex<ValueType>* __restrict__ orig_tau, uint8 stoppingId,
    bool setFinalized, stopping_status* __restrict__ stop_status,
    bool* __restrict__ device_storage)
{
    const auto tidx = thread::get_thread_id_flat();
    if (tidx < num_cols) {
        if (sqrt(abs(tau[tidx])) <= rel_residual_goal * orig_tau[tidx]) {
            stop_status[tidx].converge(stoppingId, setFinalized);
            device_storage[1] = true;
        }
        // because only false is written to all_converged, write conflicts
        // should not cause any problem
        else if (!stop_status[tidx].has_stopped()) {
            device_storage[0] = false;
        }
    }
}


__global__ __launch_bounds__(1) void init_kernel(
    bool* __restrict__ device_storage)
{
    device_storage[0] = true;
    device_storage[1] = false;
}


template <typename ValueType>
void implicit_residual_norm(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::Dense<ValueType>* tau,
    const matrix::Dense<remove_complex<ValueType>>* orig_tau,
    remove_complex<ValueType> rel_residual_goal, uint8 stoppingId,
    bool setFinalized, array<stopping_status>* stop_status,
    array<bool>* device_storage, bool* all_converged, bool* one_changed)
{
    init_kernel<<<1, 1, 0, exec->get_stream()>>>(
        as_device_type(device_storage->get_data()));

    const auto block_size = default_block_size;
    const auto grid_size = ceildiv(tau->get_size()[1], block_size);

    if (grid_size > 0) {
        implicit_residual_norm_kernel<<<grid_size, block_size, 0,
                                        exec->get_stream()>>>(
            tau->get_size()[1], as_device_type(rel_residual_goal),
            as_device_type(tau->get_const_values()),
            as_device_type(orig_tau->get_const_values()), stoppingId,
            setFinalized, as_device_type(stop_status->get_data()),
            as_device_type(device_storage->get_data()));
    }

    /* Represents all_converged, one_changed */
    *all_converged = get_element(*device_storage, 0);
    *one_changed = get_element(*device_storage, 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IMPLICIT_RESIDUAL_NORM_KERNEL);


}  // namespace implicit_residual_norm
}  // namespace hip
}  // namespace kernels
}  // namespace gko
