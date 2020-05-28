/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/solver/bicgstab_kernels.hpp"


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
 * @brief The BICGSTAB solver namespace.
 *
 * @ingroup bicgstab
 */
namespace bicgstab {


constexpr int default_block_size = 512;


#include "common/solver/bicgstab_kernels.hpp.inc"


template <typename ValueType>
void initialize(const std::shared_ptr<const DefaultExecutor> &exec,
                const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *r,
                matrix::Dense<ValueType> *rr, matrix::Dense<ValueType> *y,
                matrix::Dense<ValueType> *s, matrix::Dense<ValueType> *t,
                matrix::Dense<ValueType> *z, matrix::Dense<ValueType> *v,
                matrix::Dense<ValueType> *p, matrix::Dense<ValueType> *prev_rho,
                matrix::Dense<ValueType> *rho, matrix::Dense<ValueType> *alpha,
                matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *gamma,
                matrix::Dense<ValueType> *omega,
                Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(b->get_size()[0] * b->get_stride(), block_size.x), 1, 1);

    hipLaunchKernelGGL(
        initialize_kernel, dim3(grid_size), dim3(block_size), 0, 0,
        b->get_size()[0], b->get_size()[1], b->get_stride(),
        as_hip_type(b->get_const_values()), as_hip_type(r->get_values()),
        as_hip_type(rr->get_values()), as_hip_type(y->get_values()),
        as_hip_type(s->get_values()), as_hip_type(t->get_values()),
        as_hip_type(z->get_values()), as_hip_type(v->get_values()),
        as_hip_type(p->get_values()), as_hip_type(prev_rho->get_values()),
        as_hip_type(rho->get_values()), as_hip_type(alpha->get_values()),
        as_hip_type(beta->get_values()), as_hip_type(gamma->get_values()),
        as_hip_type(omega->get_values()), as_hip_type(stop_status->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(const std::shared_ptr<const DefaultExecutor> &exec,
            const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *p,
            const matrix::Dense<ValueType> *v,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *prev_rho,
            const matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *omega,
            const Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(r->get_size()[0] * r->get_stride(), block_size.x), 1, 1);

    hipLaunchKernelGGL(step_1_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       r->get_size()[0], r->get_size()[1], r->get_stride(),
                       as_hip_type(r->get_const_values()),
                       as_hip_type(p->get_values()),
                       as_hip_type(v->get_const_values()),
                       as_hip_type(rho->get_const_values()),
                       as_hip_type(prev_rho->get_const_values()),
                       as_hip_type(alpha->get_const_values()),
                       as_hip_type(omega->get_const_values()),
                       as_hip_type(stop_status->get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_1_KERNEL);


template <typename ValueType>
void step_2(const std::shared_ptr<const DefaultExecutor> &exec,
            const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *s,
            const matrix::Dense<ValueType> *v,
            const matrix::Dense<ValueType> *rho,
            matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *beta,
            const Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(r->get_size()[0] * r->get_stride(), block_size.x), 1, 1);

    hipLaunchKernelGGL(
        step_2_kernel, dim3(grid_size), dim3(block_size), 0, 0,
        r->get_size()[0], r->get_size()[1], r->get_stride(),
        as_hip_type(r->get_const_values()), as_hip_type(s->get_values()),
        as_hip_type(v->get_const_values()),
        as_hip_type(rho->get_const_values()), as_hip_type(alpha->get_values()),
        as_hip_type(beta->get_const_values()),
        as_hip_type(stop_status->get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_2_KERNEL);


template <typename ValueType>
void step_3(
    const std::shared_ptr<const DefaultExecutor> &exec,
    matrix::Dense<ValueType> *x, matrix::Dense<ValueType> *r,
    const matrix::Dense<ValueType> *s, const matrix::Dense<ValueType> *t,
    const matrix::Dense<ValueType> *y, const matrix::Dense<ValueType> *z,
    const matrix::Dense<ValueType> *alpha, const matrix::Dense<ValueType> *beta,
    const matrix::Dense<ValueType> *gamma, matrix::Dense<ValueType> *omega,
    const Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(r->get_size()[0] * r->get_stride(), block_size.x), 1, 1);

    hipLaunchKernelGGL(
        step_3_kernel, dim3(grid_size), dim3(block_size), 0, 0,
        r->get_size()[0], r->get_size()[1], r->get_stride(), x->get_stride(),
        as_hip_type(x->get_values()), as_hip_type(r->get_values()),
        as_hip_type(s->get_const_values()), as_hip_type(t->get_const_values()),
        as_hip_type(y->get_const_values()), as_hip_type(z->get_const_values()),
        as_hip_type(alpha->get_const_values()),
        as_hip_type(beta->get_const_values()),
        as_hip_type(gamma->get_const_values()),
        as_hip_type(omega->get_values()),
        as_hip_type(stop_status->get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_3_KERNEL);


template <typename ValueType>
void finalize(const std::shared_ptr<const DefaultExecutor> &exec,
              matrix::Dense<ValueType> *x, const matrix::Dense<ValueType> *y,
              const matrix::Dense<ValueType> *alpha,
              Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(y->get_size()[0] * y->get_stride(), block_size.x), 1, 1);

    hipLaunchKernelGGL(finalize_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       y->get_size()[0], y->get_size()[1], y->get_stride(),
                       x->get_stride(), as_hip_type(x->get_values()),
                       as_hip_type(y->get_const_values()),
                       as_hip_type(alpha->get_const_values()),
                       as_hip_type(stop_status->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_FINALIZE_KERNEL);


}  // namespace bicgstab
}  // namespace hip
}  // namespace kernels
}  // namespace gko
