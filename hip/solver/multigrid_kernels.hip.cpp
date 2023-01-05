/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include "core/solver/multigrid_kernels.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The MULTIGRID solver namespace.
 *
 * @ingroup multigrid
 */
namespace multigrid {


constexpr int default_block_size = 512;


namespace kernel {


#include "common/cuda_hip/solver/multigrid_kernels.hpp.inc"


}  // namespace kernel


template <typename ValueType>
void kcycle_step_1(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Dense<ValueType>* rho,
                   const matrix::Dense<ValueType>* v,
                   matrix::Dense<ValueType>* g, matrix::Dense<ValueType>* d,
                   matrix::Dense<ValueType>* e)
{
    const auto nrows = e->get_size()[0];
    const auto nrhs = e->get_size()[1];
    constexpr int max_size = (1U << 31) - 1;
    const size_type grid_nrows =
        max_size / nrhs < nrows ? max_size / nrhs : nrows;
    const auto grid = ceildiv(grid_nrows * nrhs, default_block_size);
    if (grid > 0) {
        hipLaunchKernelGGL(
            kernel::kcycle_step_1_kernel, grid, default_block_size, 0, 0, nrows,
            nrhs, e->get_stride(), grid_nrows,
            as_hip_type(alpha->get_const_values()),
            as_hip_type(rho->get_const_values()),
            as_hip_type(v->get_const_values()), as_hip_type(g->get_values()),
            as_hip_type(d->get_values()), as_hip_type(e->get_values()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_1_KERNEL);


template <typename ValueType>
void kcycle_step_2(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Dense<ValueType>* rho,
                   const matrix::Dense<ValueType>* gamma,
                   const matrix::Dense<ValueType>* beta,
                   const matrix::Dense<ValueType>* zeta,
                   const matrix::Dense<ValueType>* d,
                   matrix::Dense<ValueType>* e)
{
    const auto nrows = e->get_size()[0];
    const auto nrhs = e->get_size()[1];
    constexpr int max_size = (1U << 31) - 1;
    const size_type grid_nrows =
        max_size / nrhs < nrows ? max_size / nrhs : nrows;
    const auto grid = ceildiv(grid_nrows * nrhs, default_block_size);
    if (grid > 0) {
        hipLaunchKernelGGL(
            kernel::kcycle_step_2_kernel, grid, default_block_size, 0, 0, nrows,
            nrhs, e->get_stride(), grid_nrows,
            as_hip_type(alpha->get_const_values()),
            as_hip_type(rho->get_const_values()),
            as_hip_type(gamma->get_const_values()),
            as_hip_type(beta->get_const_values()),
            as_hip_type(zeta->get_const_values()),
            as_hip_type(d->get_const_values()), as_hip_type(e->get_values()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_2_KERNEL);


template <typename ValueType>
void kcycle_check_stop(std::shared_ptr<const DefaultExecutor> exec,
                       const matrix::Dense<ValueType>* old_norm,
                       const matrix::Dense<ValueType>* new_norm,
                       const ValueType rel_tol, bool& is_stop)
{
    gko::array<bool> dis_stop(exec, 1);
    components::fill_array(exec, dis_stop.get_data(), dis_stop.get_num_elems(),
                           true);
    const auto nrhs = new_norm->get_size()[1];
    const auto grid = ceildiv(nrhs, default_block_size);
    if (grid > 0) {
        hipLaunchKernelGGL(
            kernel::kcycle_check_stop_kernel, grid, default_block_size, 0, 0,
            nrhs, as_hip_type(old_norm->get_const_values()),
            as_hip_type(new_norm->get_const_values()), as_hip_type(rel_tol),
            as_hip_type(dis_stop.get_data()));
    }
    is_stop = exec->copy_val_to_host(dis_stop.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(
    GKO_DECLARE_MULTIGRID_KCYCLE_CHECK_STOP_KERNEL);


}  // namespace multigrid
}  // namespace hip
}  // namespace kernels
}  // namespace gko
