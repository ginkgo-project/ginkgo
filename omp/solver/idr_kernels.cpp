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

#include "core/solver/idr_kernels.hpp"


#include <algorithm>


#include <omp.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The IDR solver namespace.
 *
 * @ingroup idr
 */
namespace idr {


template <typename ValueType>
void step_1(std::shared_ptr<const OmpExecutor> exec,
            const matrix::Dense<ValueType> *m, matrix::Dense<ValueType> *f,
            const matrix::Dense<ValueType> *c,
            const matrix::Dense<ValueType> *v,
            const matrix::Dense<ValueType> *residual,
            const matrix::Dense<ValueType> *g,
            const Array<stopping_status> *stop_status) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:idr): change the code imported from solver/bicgstab if needed
//    const dim3 block_size(default_block_size, 1, 1);
//    const dim3 grid_size(
//        ceildiv(r->get_size()[0] * r->get_stride(), block_size.x), 1, 1);
//
//    step_1_kernel<<<grid_size, block_size, 0, 0>>>(
//        r->get_size()[0], r->get_size()[1], r->get_stride(),
//        as_cuda_type(r->get_const_values()), as_cuda_type(p->get_values()),
//        as_cuda_type(v->get_const_values()),
//        as_cuda_type(rho->get_const_values()),
//        as_cuda_type(prev_rho->get_const_values()),
//        as_cuda_type(alpha->get_const_values()),
//        as_cuda_type(omega->get_const_values()),
//        as_cuda_type(stop_status->get_const_data()));
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const OmpExecutor> exec,
            const matrix::Dense<ValueType> *u, matrix::Dense<ValueType> *c,
            const matrix::Dense<ValueType> *preconditioned_vector,
            const Array<stopping_status> *stop_status) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:idr): change the code imported from solver/bicgstab if needed
//    const dim3 block_size(default_block_size, 1, 1);
//    const dim3 grid_size(
//        ceildiv(r->get_size()[0] * r->get_stride(), block_size.x), 1, 1);
//
//    step_2_kernel<<<grid_size, block_size, 0, 0>>>(
//        r->get_size()[0], r->get_size()[1], r->get_stride(),
//        as_cuda_type(r->get_const_values()), as_cuda_type(s->get_values()),
//        as_cuda_type(v->get_const_values()),
//        as_cuda_type(rho->get_const_values()),
//        as_cuda_type(alpha->get_values()),
//        as_cuda_type(beta->get_const_values()),
//        as_cuda_type(stop_status->get_const_data()));
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_2_KERNEL);


template <typename ValueType>
void step_3(std::shared_ptr<const OmpExecutor> exec,
            matrix::Dense<ValueType> *p, matrix::Dense<ValueType> *g,
            const matrix::Dense<ValueType> *u,
            const matrix::Dense<ValueType> *m,
            const matrix::Dense<ValueType> *f,
            const matrix::Dense<ValueType> *c,
            const matrix::Dense<ValueType> *residual,
            const matrix::Dense<ValueType> *x,
            const Array<stopping_status> *stop_status) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:idr): change the code imported from solver/bicgstab if needed
//    const dim3 block_size(default_block_size, 1, 1);
//    const dim3 grid_size(
//        ceildiv(r->get_size()[0] * r->get_stride(), block_size.x), 1, 1);
//
//    step_3_kernel<<<grid_size, block_size, 0, 0>>>(
//        r->get_size()[0], r->get_size()[1], r->get_stride(), x->get_stride(),
//        as_cuda_type(x->get_values()), as_cuda_type(r->get_values()),
//        as_cuda_type(s->get_const_values()),
//        as_cuda_type(t->get_const_values()),
//        as_cuda_type(y->get_const_values()),
//        as_cuda_type(z->get_const_values()),
//        as_cuda_type(alpha->get_const_values()),
//        as_cuda_type(beta->get_const_values()),
//        as_cuda_type(gamma->get_const_values()),
//        as_cuda_type(omega->get_values()),
//        as_cuda_type(stop_status->get_const_data()));
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_3_KERNEL);


template <typename ValueType>
void step_4(std::shared_ptr<const OmpExecutor> exec, const ValueType kappa,
            matrix::Dense<ValueType> *omega, const matrix::Dense<ValueType> *t,
            const matrix::Dense<ValueType> *residual,
            matrix::Dense<ValueType> *residual_norm,
            const matrix::Dense<ValueType> *v, matrix::Dense<ValueType> *x,
            Array<stopping_status> *stop_status) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:idr): change the code imported from solver/bicgstab if needed
//    const dim3 block_size(default_block_size, 1, 1);
//    const dim3 grid_size(
//        ceildiv(y->get_size()[0] * y->get_stride(), block_size.x), 1, 1);
//
//    finalize_kernel<<<grid_size, block_size, 0, 0>>>(
//        y->get_size()[0], y->get_size()[1], y->get_stride(), x->get_stride(),
//        as_cuda_type(x->get_values()), as_cuda_type(y->get_const_values()),
//        as_cuda_type(alpha->get_const_values()),
//        as_cuda_type(stop_status->get_data()));
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_4_KERNEL);


}  // namespace idr
}  // namespace omp
}  // namespace kernels
}  // namespace gko
