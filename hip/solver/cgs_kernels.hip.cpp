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

#include "core/solver/cgs_kernels.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The CGS solver namespace.
 *
 * @ingroup cgs
 */
namespace cgs {


constexpr int default_block_size = 512;


#include "common/solver/cgs_kernels.hpp.inc"


template <typename ValueType>
void initialize(std::shared_ptr<const HipExecutor> exec,
                const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *r,
                matrix::Dense<ValueType> *r_tld, matrix::Dense<ValueType> *p,
                matrix::Dense<ValueType> *q, matrix::Dense<ValueType> *u,
                matrix::Dense<ValueType> *u_hat,
                matrix::Dense<ValueType> *v_hat, matrix::Dense<ValueType> *t,
                matrix::Dense<ValueType> *alpha, matrix::Dense<ValueType> *beta,
                matrix::Dense<ValueType> *gamma,
                matrix::Dense<ValueType> *rho_prev,
                matrix::Dense<ValueType> *rho,
                Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(b->get_size()[0] * b->get_stride(), block_size.x), 1, 1);

    hipLaunchKernelGGL(
        initialize_kernel, dim3(grid_size), dim3(block_size), 0, 0,
        b->get_size()[0], b->get_size()[1], b->get_stride(),
        as_hip_type(b->get_const_values()), as_hip_type(r->get_values()),
        as_hip_type(r_tld->get_values()), as_hip_type(p->get_values()),
        as_hip_type(q->get_values()), as_hip_type(u->get_values()),
        as_hip_type(u_hat->get_values()), as_hip_type(v_hat->get_values()),
        as_hip_type(t->get_values()), as_hip_type(alpha->get_values()),
        as_hip_type(beta->get_values()), as_hip_type(gamma->get_values()),
        as_hip_type(rho_prev->get_values()), as_hip_type(rho->get_values()),
        as_hip_type(stop_status->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const HipExecutor> exec,
            const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *u,
            matrix::Dense<ValueType> *p, const matrix::Dense<ValueType> *q,
            matrix::Dense<ValueType> *beta, const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *rho_prev,
            const Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(p->get_size()[0] * p->get_stride(), block_size.x), 1, 1);

    hipLaunchKernelGGL(
        step_1_kernel, dim3(grid_size), dim3(block_size), 0, 0,
        p->get_size()[0], p->get_size()[1], p->get_stride(),
        as_hip_type(r->get_const_values()), as_hip_type(u->get_values()),
        as_hip_type(p->get_values()), as_hip_type(q->get_const_values()),
        as_hip_type(beta->get_values()), as_hip_type(rho->get_const_values()),
        as_hip_type(rho_prev->get_const_values()),
        as_hip_type(stop_status->get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const HipExecutor> exec,
            const matrix::Dense<ValueType> *u,
            const matrix::Dense<ValueType> *v_hat, matrix::Dense<ValueType> *q,
            matrix::Dense<ValueType> *t, matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *gamma,
            const Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(u->get_size()[0] * u->get_stride(), block_size.x), 1, 1);

    hipLaunchKernelGGL(
        step_2_kernel, dim3(grid_size), dim3(block_size), 0, 0,
        u->get_size()[0], u->get_size()[1], u->get_stride(),
        as_hip_type(u->get_const_values()),
        as_hip_type(v_hat->get_const_values()), as_hip_type(q->get_values()),
        as_hip_type(t->get_values()), as_hip_type(alpha->get_values()),
        as_hip_type(rho->get_const_values()),
        as_hip_type(gamma->get_const_values()),
        as_hip_type(stop_status->get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_2_KERNEL);


template <typename ValueType>
void step_3(std::shared_ptr<const HipExecutor> exec,
            const matrix::Dense<ValueType> *t,
            const matrix::Dense<ValueType> *u_hat, matrix::Dense<ValueType> *r,
            matrix::Dense<ValueType> *x, const matrix::Dense<ValueType> *alpha,
            const Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(t->get_size()[0] * t->get_stride(), block_size.x), 1, 1);

    hipLaunchKernelGGL(
        step_3_kernel, dim3(grid_size), dim3(block_size), 0, 0,
        t->get_size()[0], t->get_size()[1], t->get_stride(), x->get_stride(),
        as_hip_type(t->get_const_values()),
        as_hip_type(u_hat->get_const_values()), as_hip_type(r->get_values()),
        as_hip_type(x->get_values()), as_hip_type(alpha->get_const_values()),
        as_hip_type(stop_status->get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_3_KERNEL);


}  // namespace cgs
}  // namespace hip
}  // namespace kernels
}  // namespace gko
