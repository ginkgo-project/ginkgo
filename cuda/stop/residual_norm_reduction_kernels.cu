/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/stop/residual_norm_reduction_kernels.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "core/stop/residual_norm_reduction.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"

namespace gko {
namespace kernels {
namespace cuda {
namespace residual_norm_reduction {


constexpr int default_block_size = 512;


template <typename ValueType>
__global__
    __launch_bounds__(default_block_size) void residual_norm_reduction_kernel(
        size_type num_cols, remove_complex<ValueType> rel_residual_goal,
        const ValueType *__restrict__ tau,
        const ValueType *__restrict__ orig_tau, uint8 stoppingId,
        bool setFinalized, stopping_status *__restrict__ stop_status,
        bool *__restrict__ device_storage)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (tidx < num_cols) {
        if (abs(tau[tidx]) < rel_residual_goal * abs(orig_tau[tidx])) {
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
void residual_norm_reduction(std::shared_ptr<const CudaExecutor> exec,
                             const matrix::Dense<ValueType> *tau,
                             const matrix::Dense<ValueType> *orig_tau,
                             remove_complex<ValueType> rel_residual_goal,
                             uint8 stoppingId, bool setFinalized,
                             Array<stopping_status> *stop_status,
                             Array<bool> *device_storage, bool *all_converged,
                             bool *one_changed)
{
    /* Represents all_converged, one_changed */
    bool tmp[2] = {true, false};
    exec->copy_from(exec->get_master().get(), 2, tmp,
                    device_storage->get_data());

    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(tau->get_size()[1], block_size.x), 1, 1);

    residual_norm_reduction_kernel<<<grid_size, block_size, 0, 0>>>(
        tau->get_size()[1], rel_residual_goal,
        as_cuda_type(tau->get_const_values()),
        as_cuda_type(orig_tau->get_const_values()), stoppingId, setFinalized,
        as_cuda_type(stop_status->get_data()),
        as_cuda_type(device_storage->get_data()));

    exec->get_master()->copy_from(exec.get(), 2,
                                  device_storage->get_const_data(), tmp);
    *all_converged = tmp[0];
    *one_changed = tmp[1];
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_RESIDUAL_NORM_REDUCTION_KERNEL);


}  // namespace residual_norm_reduction
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
