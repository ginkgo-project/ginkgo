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
#include "gpu/base/types.hpp"


namespace gko {
namespace kernels {
namespace gpu {
namespace bicgstab {


constexpr int default_block_size = 512;


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void initialize_kernel(
    size_type num_rows, size_type num_cols, size_type padding,
    const ValueType *b, ValueType *r, ValueType *z, ValueType *p, ValueType *v,
    ValueType *t, ValueType *y, ValueType *rr, ValueType *s,
    ValueType *prev_rho, ValueType *rho, ValueType *beta, ValueType *alpha,
    ValueType *omega)
{
    const size_type tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num_cols) {
        rho[tidx] = one<ValueType>();
        alpha[tidx] = one<ValueType>();
        beta[tidx] = one<ValueType>();
        omega[tidx] = one<ValueType>();
        prev_rho[tidx] = one<ValueType>();
    }

    if (tidx < num_rows * padding) {
        r[tidx] = b[tidx];
        rr[tidx] = zero<ValueType>();
        z[tidx] = zero<ValueType>();
        p[tidx] = zero<ValueType>();
        v[tidx] = zero<ValueType>();
        t[tidx] = zero<ValueType>();
        s[tidx] = zero<ValueType>();
        y[tidx] = zero<ValueType>();
    }
}


template <typename ValueType>
void initialize(const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *r,
                matrix::Dense<ValueType> *rr, matrix::Dense<ValueType> *y,
                matrix::Dense<ValueType> *s, matrix::Dense<ValueType> *t,
                matrix::Dense<ValueType> *z, matrix::Dense<ValueType> *v,
                matrix::Dense<ValueType> *p, matrix::Dense<ValueType> *prev_rho,
                matrix::Dense<ValueType> *rho, matrix::Dense<ValueType> *alpha,
                matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *omega)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(b->get_num_rows() * b->get_padding(), block_size.x), 1, 1);

    initialize_kernel<<<grid_size, block_size, 0, 0>>>(
        b->get_num_rows(), b->get_num_cols(), b->get_padding(),
        as_cuda_type(b->get_const_values()), as_cuda_type(r->get_values()),
        as_cuda_type(z->get_values()), as_cuda_type(p->get_values()),
        as_cuda_type(v->get_values()), as_cuda_type(t->get_values()),
        as_cuda_type(y->get_values()), as_cuda_type(rr->get_values()),
        as_cuda_type(s->get_values()), as_cuda_type(prev_rho->get_values()),
        as_cuda_type(rho->get_values()), as_cuda_type(beta->get_values()),
        as_cuda_type(alpha->get_values()), as_cuda_type(omega->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_INITIALIZE_KERNEL);


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_1_kernel(
    size_type num_rows, size_type num_cols, size_type padding, ValueType *p,
    const ValueType *r, const ValueType *v, const ValueType *rho,
    const ValueType *prev_rho, const ValueType *alpha, const ValueType *omega)
{
    const size_type tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const size_type col = tidx % padding;
    ValueType tmp = zero<ValueType>();

    if (col >= num_cols || tidx >= num_rows * padding) {
        return;
    }
    tmp = (prev_rho[col] * omega[col] != zero<ValueType>())
              ? rho[col] / prev_rho[col] * alpha[col] / omega[col]
              : zero<ValueType>();
    p[tidx] = (tmp == zero<ValueType>())
                  ? r[tidx]
                  : r[tidx] + tmp * (p[tidx] - omega[col] * v[tidx]);
}


template <typename ValueType>
void step_1(const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *p,
            const matrix::Dense<ValueType> *v,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *prev_rho,
            const matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *omega)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(p->get_num_rows() * p->get_padding(), block_size.x), 1, 1);

    step_1_kernel<<<grid_size, block_size, 0, 0>>>(
        p->get_num_rows(), p->get_num_cols(), p->get_padding(),
        as_cuda_type(p->get_values()), as_cuda_type(r->get_const_values()),
        as_cuda_type(v->get_const_values()),
        as_cuda_type(rho->get_const_values()),
        as_cuda_type(prev_rho->get_const_values()),
        as_cuda_type(alpha->get_const_values()),
        as_cuda_type(omega->get_const_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_1_KERNEL);


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_2_kernel(
    size_type num_rows, size_type num_cols, size_type padding, ValueType *s,
    const ValueType *r, const ValueType *v, ValueType *alpha,
    const ValueType *beta, const ValueType *rho)
{
    const size_type tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const size_type col = tidx % padding;
    if (tidx < num_rows * padding) {
        alpha[col] = (beta[col] != zero<ValueType>()) ? rho[col] / beta[col]
                                                      : zero<ValueType>();
        s[tidx] = (alpha[col] == zero<ValueType>())
                      ? r[tidx]
                      : r[tidx] - alpha[col] * v[tidx];
    }
}


template <typename ValueType>
void step_2(const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *s,
            const matrix::Dense<ValueType> *v,
            const matrix::Dense<ValueType> *rho,
            matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *beta)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(s->get_num_rows() * s->get_padding(), block_size.x), 1, 1);

    step_2_kernel<<<grid_size, block_size, 0, 0>>>(
        s->get_num_rows(), s->get_num_cols(), s->get_padding(),
        as_cuda_type(s->get_values()), as_cuda_type(r->get_const_values()),
        as_cuda_type(v->get_const_values()), as_cuda_type(alpha->get_values()),
        as_cuda_type(beta->get_const_values()),
        as_cuda_type(rho->get_const_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_2_KERNEL);


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_3_kernel(
    size_type num_rows, size_type num_cols, size_type padding, ValueType *x,
    size_type padding_b, ValueType *r, const ValueType *y, const ValueType *z,
    const ValueType *s, const ValueType *t, ValueType *omega,
    const ValueType *alpha, const ValueType *beta)
{
    const size_type tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const size_type col = tidx % padding;
    if (tidx < num_rows * padding && tidx < num_rows * padding_b) {
        omega[col] = (beta[col] != zero<ValueType>()) ? omega[col] / beta[col]
                                                      : zero<ValueType>();
        // x[tidx] = (omega[col] == zero<ValueType>()) ? x[tidx] : x[tidx] +
        // alpha[col] * y[tidx] + omega[col]*z[tidx];
        x[tidx] = x[tidx] + alpha[col] * y[tidx] + omega[col] * z[tidx];
        // r[tidx] = (omega[col] == zero<ValueType>()) ? r[tidx] : r[tidx] - tmp
        // * q[tidx];
        r[tidx] = s[tidx] - omega[col] * t[tidx];
    }
}

template <typename ValueType>
void step_3(matrix::Dense<ValueType> *x, matrix::Dense<ValueType> *r,
            const matrix::Dense<ValueType> *s,
            const matrix::Dense<ValueType> *t,
            const matrix::Dense<ValueType> *y,
            const matrix::Dense<ValueType> *z,
            const matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *beta,
            matrix::Dense<ValueType> *omega)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(x->get_num_rows() * x->get_padding(), block_size.x), 1, 1);

    step_3_kernel<<<grid_size, block_size, 0, 0>>>(
        x->get_num_rows(), x->get_num_cols(), x->get_padding(),
        as_cuda_type(x->get_values()), t->get_padding(),
        as_cuda_type(r->get_values()), as_cuda_type(y->get_const_values()),
        as_cuda_type(z->get_const_values()),
        as_cuda_type(s->get_const_values()),
        as_cuda_type(t->get_const_values()), as_cuda_type(omega->get_values()),
        as_cuda_type(alpha->get_const_values()),
        as_cuda_type(beta->get_const_values()));
}
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_3_KERNEL);


}  // namespace bicgstab
}  // namespace gpu
}  // namespace kernels
}  // namespace gko
