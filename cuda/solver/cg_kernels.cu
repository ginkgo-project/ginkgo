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

#include "core/solver/cg_kernels.hpp"


#include <ginkgo/core/base/math.hpp>
#include "core/base/exception_helpers.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace cg {


constexpr int default_block_size = 512;


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void initialize_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ b, ValueType *__restrict__ r,
    ValueType *__restrict__ z, ValueType *__restrict__ p,
    ValueType *__restrict__ q, ValueType *__restrict__ prev_rho,
    ValueType *__restrict__ rho, stopping_status *__restrict__ stop_status)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tidx < num_cols) {
        rho[tidx] = zero<ValueType>();
        prev_rho[tidx] = one<ValueType>();
        stop_status[tidx].reset();
    }

    if (tidx < num_rows * stride) {
        r[tidx] = b[tidx];
        z[tidx] = zero<ValueType>();
        p[tidx] = zero<ValueType>();
        q[tidx] = zero<ValueType>();
    }
}


template <typename ValueType>
void initialize(std::shared_ptr<const CudaExecutor> exec,
                const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *r,
                matrix::Dense<ValueType> *z, matrix::Dense<ValueType> *p,
                matrix::Dense<ValueType> *q, matrix::Dense<ValueType> *prev_rho,
                matrix::Dense<ValueType> *rho,
                Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(b->get_size()[0] * b->get_stride(), block_size.x), 1, 1);

    initialize_kernel<<<grid_size, block_size, 0, 0>>>(
        b->get_size()[0], b->get_size()[1], b->get_stride(),
        as_cuda_type(b->get_const_values()), as_cuda_type(r->get_values()),
        as_cuda_type(z->get_values()), as_cuda_type(p->get_values()),
        as_cuda_type(q->get_values()), as_cuda_type(prev_rho->get_values()),
        as_cuda_type(rho->get_values()), as_cuda_type(stop_status->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_INITIALIZE_KERNEL);


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_1_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    ValueType *__restrict__ p, const ValueType *__restrict__ z,
    const ValueType *__restrict__ rho, const ValueType *__restrict__ prev_rho,
    const stopping_status *__restrict__ stop_status)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    const auto col = tidx % stride;
    if (col >= num_cols || tidx >= num_rows * stride ||
        stop_status[col].has_stopped()) {
        return;
    }
    const auto tmp = rho[col] / prev_rho[col];
    p[tidx] =
        prev_rho[col] == zero<ValueType>() ? z[tidx] : z[tidx] + tmp * p[tidx];
}


template <typename ValueType>
void step_1(std::shared_ptr<const CudaExecutor> exec,
            matrix::Dense<ValueType> *p, const matrix::Dense<ValueType> *z,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *prev_rho,
            const Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(p->get_size()[0] * p->get_stride(), block_size.x), 1, 1);

    step_1_kernel<<<grid_size, block_size, 0, 0>>>(
        p->get_size()[0], p->get_size()[1], p->get_stride(),
        as_cuda_type(p->get_values()), as_cuda_type(z->get_const_values()),
        as_cuda_type(rho->get_const_values()),
        as_cuda_type(prev_rho->get_const_values()),
        as_cuda_type(stop_status->get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_STEP_1_KERNEL);


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_2_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    size_type x_stride, ValueType *__restrict__ x, ValueType *__restrict__ r,
    const ValueType *__restrict__ p, const ValueType *__restrict__ q,
    const ValueType *__restrict__ beta, const ValueType *__restrict__ rho,
    const stopping_status *__restrict__ stop_status)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    const auto row = tidx / stride;
    const auto col = tidx % stride;

    if (col >= num_cols || tidx >= num_rows * num_cols ||
        stop_status[col].has_stopped()) {
        return;
    }
    if (beta[col] != zero<ValueType>()) {
        const auto tmp = rho[col] / beta[col];
        x[row * x_stride + col] += tmp * p[tidx];
        r[tidx] -= tmp * q[tidx];
    }
}


template <typename ValueType>
void step_2(std::shared_ptr<const CudaExecutor> exec,
            matrix::Dense<ValueType> *x, matrix::Dense<ValueType> *r,
            const matrix::Dense<ValueType> *p,
            const matrix::Dense<ValueType> *q,
            const matrix::Dense<ValueType> *beta,
            const matrix::Dense<ValueType> *rho,
            const Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(p->get_size()[0] * p->get_stride(), block_size.x), 1, 1);

    step_2_kernel<<<grid_size, block_size, 0, 0>>>(
        p->get_size()[0], p->get_size()[1], p->get_stride(), x->get_stride(),
        as_cuda_type(x->get_values()), as_cuda_type(r->get_values()),
        as_cuda_type(p->get_const_values()),
        as_cuda_type(q->get_const_values()),
        as_cuda_type(beta->get_const_values()),
        as_cuda_type(rho->get_const_values()),
        as_cuda_type(stop_status->get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_STEP_2_KERNEL);


}  // namespace cg
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
