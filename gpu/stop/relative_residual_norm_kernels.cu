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

#include "core/solver/bicgstab_kernels.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "gpu/base/math.hpp"
#include "gpu/base/types.hpp"

namespace gko {
namespace kernels {
namespace gpu {
namespace relative_residual_norm {


constexpr int default_block_size = 512;


template <typename ValueType>
__global__
    __launch_bounds__(default_block_size) void relative_residual_norm_kernel(
        size_type num_cols, remove_complex<ValueType> rel_residual_goal,
        const ValueType *__restrict__ tau,
        const ValueType *__restrict__ orig_tau, bool *__restrict__ converged,
        bool *__restrict__ all_converged)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (tidx < num_cols) {
        if (abs(tau[tidx]) < rel_residual_goal * abs(orig_tau[tidx])) {
            converged[tidx] = true;
        }
        // because only false is written to all_converged, write conflicts
        // should not cause any problem
        else if (converged[tidx] == false) {
            *all_converged = false;
        }
    }
}

template <typename ValueType>
void relative_residual_norm(std::shared_ptr<const GpuExecutor> exec,
                            const matrix::Dense<ValueType> *tau,
                            const matrix::Dense<ValueType> *orig_tau,
                            remove_complex<ValueType> rel_residual_goal,
                            Array<bool> *converged, bool *all_converged)
{
    Array<bool> d_all_converged(exec, 1);
    Array<bool> all_converged_array(exec->get_master());

    // initialize all_converged with true
    *all_converged = true;
    all_converged_array.manage(1, all_converged);

    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(tau->get_size().num_cols, block_size.x), 1, 1);

    test_convergence_kernel<<<grid_size, block_size, 0, 0>>>(
        tau->get_size().num_cols, rel_residual_goal,
        as_cuda_type(tau->get_const_values()),
        as_cuda_type(orig_tau->get_const_values()),
        as_cuda_type(converged->get_data()),
        as_cuda_type(d_all_converged.get_data()));

    all_converged_array = d_all_converged;
    all_converged_array.release();
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_RELATIVE_RESIDUAL_NORM_KERNEL);


}  // namespace relative_residual_norm
}  // namespace gpu
}  // namespace kernels
}  // namespace gko
