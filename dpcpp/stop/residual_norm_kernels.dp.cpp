/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/stop/residual_norm_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


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


constexpr int default_block_size = 256;


template <typename ValueType>
void residual_norm_kernel(size_type num_cols, ValueType rel_residual_goal,
                          const ValueType *__restrict__ tau,
                          const ValueType *__restrict__ orig_tau,
                          uint8 stoppingId, bool setFinalized,
                          stopping_status *__restrict__ stop_status,
                          bool *__restrict__ device_storage,
                          sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    if (tidx < num_cols) {
        if (tau[tidx] < rel_residual_goal * orig_tau[tidx]) {
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

template <typename ValueType>
void residual_norm_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                          sycl::queue *stream, size_type num_cols,
                          ValueType rel_residual_goal, const ValueType *tau,
                          const ValueType *orig_tau, uint8 stoppingId,
                          bool setFinalized, stopping_status *stop_status,
                          bool *device_storage)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                residual_norm_kernel(num_cols, rel_residual_goal, tau, orig_tau,
                                     stoppingId, setFinalized, stop_status,
                                     device_storage, item_ct1);
            });
    });
}


void init_kernel(bool *__restrict__ device_storage)
{
    device_storage[0] = true;
    device_storage[1] = false;
}

void init_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                 sycl::queue *stream, bool *device_storage)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1) { init_kernel(device_storage); });
    });
}


template <typename ValueType>
void residual_norm(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::Dense<ValueType> *tau,
                   const matrix::Dense<ValueType> *orig_tau,
                   ValueType rel_residual_goal, uint8 stoppingId,
                   bool setFinalized, Array<stopping_status> *stop_status,
                   Array<bool> *device_storage, bool *all_converged,
                   bool *one_changed)
{
    static_assert(is_complex_s<ValueType>::value == false,
                  "ValueType must not be complex in this function!");
    init_kernel(1, 1, 0, exec->get_queue(), device_storage->get_data());

    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(tau->get_size()[1], block_size.x), 1, 1);

    residual_norm_kernel(grid_size, block_size, 0, exec->get_queue(),
                         tau->get_size()[1], rel_residual_goal,
                         tau->get_const_values(), orig_tau->get_const_values(),
                         stoppingId, setFinalized, stop_status->get_data(),
                         device_storage->get_data());

    /* Represents all_converged, one_changed */
    *all_converged = exec->copy_val_to_host(device_storage->get_const_data());
    *one_changed = exec->copy_val_to_host(device_storage->get_const_data() + 1);
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


constexpr int default_block_size = 256;


template <typename ValueType>
void implicit_residual_norm_kernel(
    size_type num_cols, remove_complex<ValueType> rel_residual_goal,
    const ValueType *__restrict__ tau,
    const remove_complex<ValueType> *__restrict__ orig_tau, uint8 stoppingId,
    bool setFinalized, stopping_status *__restrict__ stop_status,
    bool *__restrict__ device_storage, sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    if (tidx < num_cols) {
        if (std::sqrt(std::abs(tau[tidx])) <
            rel_residual_goal * orig_tau[tidx]) {
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

template <typename ValueType>
void implicit_residual_norm_kernel(
    dim3 grid, dim3 block, size_t dynamic_shared_memory, sycl::queue *stream,
    size_type num_cols, remove_complex<ValueType> rel_residual_goal,
    const ValueType *tau, const remove_complex<ValueType> *orig_tau,
    uint8 stoppingId, bool setFinalized, stopping_status *stop_status,
    bool *device_storage)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                implicit_residual_norm_kernel(
                    num_cols, rel_residual_goal, tau, orig_tau, stoppingId,
                    setFinalized, stop_status, device_storage, item_ct1);
            });
    });
}


void init_kernel(bool *__restrict__ device_storage)
{
    device_storage[0] = true;
    device_storage[1] = false;
}

void init_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                 sycl::queue *stream, bool *device_storage)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1) { init_kernel(device_storage); });
    });
}


template <typename ValueType>
void implicit_residual_norm(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::Dense<ValueType> *tau,
    const matrix::Dense<remove_complex<ValueType>> *orig_tau,
    remove_complex<ValueType> rel_residual_goal, uint8 stoppingId,
    bool setFinalized, Array<stopping_status> *stop_status,
    Array<bool> *device_storage, bool *all_converged, bool *one_changed)
{
    init_kernel(1, 1, 0, exec->get_queue(), device_storage->get_data());

    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(tau->get_size()[1], block_size.x), 1, 1);

    implicit_residual_norm_kernel(
        grid_size, block_size, 0, exec->get_queue(), tau->get_size()[1],
        rel_residual_goal, tau->get_const_values(),
        orig_tau->get_const_values(), stoppingId, setFinalized,
        stop_status->get_data(), device_storage->get_data());

    /* Represents all_converged, one_changed */
    *all_converged = exec->copy_val_to_host(device_storage->get_const_data());
    *one_changed = exec->copy_val_to_host(device_storage->get_const_data() + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IMPLICIT_RESIDUAL_NORM_KERNEL);


}  // namespace implicit_residual_norm
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
