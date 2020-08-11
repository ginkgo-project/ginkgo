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


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The IDR solver namespace.
 *
 * @ingroup idr
 */
namespace idr {


constexpr int default_block_size = 512;


template <typename ValueType>
void initialize(std::shared_ptr<const CudaExecutor> exec,
                matrix::Dense<ValueType> *m, matrix::Dense<ValueType> *g,
                Array<stopping_status> *stop_status) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const CudaExecutor> exec, const size_type k,
            const matrix::Dense<ValueType> *m,
            const matrix::Dense<ValueType> *f,
            const matrix::Dense<ValueType> *residual,
            const matrix::Dense<ValueType> *g, matrix::Dense<ValueType> *c,
            matrix::Dense<ValueType> *v,
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
void step_2(std::shared_ptr<const CudaExecutor> exec, const size_type k,
            const matrix::Dense<ValueType> *omega,
            const matrix::Dense<ValueType> *preconditioned_vector,
            const matrix::Dense<ValueType> *c, matrix::Dense<ValueType> *u,
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
void step_3(std::shared_ptr<const CudaExecutor> exec, const size_type k,
            const matrix::Dense<ValueType> *p, matrix::Dense<ValueType> *g,
            matrix::Dense<ValueType> *u, matrix::Dense<ValueType> *m,
            matrix::Dense<ValueType> *f, matrix::Dense<ValueType> *residual,
            matrix::Dense<ValueType> *x,
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
void compute_omega(
    std::shared_ptr<const CudaExecutor> exec,
    const remove_complex<ValueType> kappa, const matrix::Dense<ValueType> *tht,
    const matrix::Dense<remove_complex<ValueType>> *t_norm,
    const matrix::Dense<remove_complex<ValueType>> *residual_norm,
    matrix::Dense<ValueType> *rho, matrix::Dense<ValueType> *omega,
    const Array<stopping_status> *stop_status) GKO_NOT_IMPLEMENTED;
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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_COMPUTE_OMEGA_KERNEL);


}  // namespace idr
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
