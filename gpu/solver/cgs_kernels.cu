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

#include "core/solver/cgs_kernels.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "gpu/base/math.hpp"
#include "gpu/base/types.hpp"


namespace gko {
namespace kernels {
namespace gpu {
namespace cgs {


constexpr int default_block_size = 512;


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void initialize_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ b, ValueType *__restrict__ r,
    ValueType *__restrict__ r_tld, ValueType *__restrict__ p,
    ValueType *__restrict__ q, ValueType *__restrict__ u,
    ValueType *__restrict__ u_hat, ValueType *__restrict__ v_hat,
    ValueType *__restrict__ t, ValueType *__restrict__ alpha,
    ValueType *__restrict__ beta, ValueType *__restrict__ gamma,
    ValueType *__restrict__ rho_prev, ValueType *__restrict__ rho,
    bool *__restrict__ converged)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tidx < num_cols) {
        rho[tidx] = zero<ValueType>();
        alpha[tidx] = one<ValueType>();
        beta[tidx] = one<ValueType>();
        gamma[tidx] = one<ValueType>();
        rho_prev[tidx] = one<ValueType>();
        converged[tidx] = false;
    }

    if (tidx < num_rows * stride) {
        r[tidx] = b[tidx];
        r_tld[tidx] = b[tidx];
        u[tidx] = zero<ValueType>();
        p[tidx] = zero<ValueType>();
        q[tidx] = zero<ValueType>();
        u_hat[tidx] = zero<ValueType>();
        v_hat[tidx] = zero<ValueType>();
        t[tidx] = zero<ValueType>();
    }
}


template <typename ValueType>
void initialize(std::shared_ptr<const GpuExecutor> exec,
                const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *r,
                matrix::Dense<ValueType> *r_tld, matrix::Dense<ValueType> *p,
                matrix::Dense<ValueType> *q, matrix::Dense<ValueType> *u,
                matrix::Dense<ValueType> *u_hat,
                matrix::Dense<ValueType> *v_hat, matrix::Dense<ValueType> *t,
                matrix::Dense<ValueType> *alpha, matrix::Dense<ValueType> *beta,
                matrix::Dense<ValueType> *gamma,
                matrix::Dense<ValueType> *rho_prev,
                matrix::Dense<ValueType> *rho, Array<bool> *converged)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(b->get_size().num_rows * b->get_stride(), block_size.x), 1, 1);

    initialize_kernel<<<grid_size, block_size, 0, 0>>>(
        b->get_size().num_rows, b->get_size().num_cols, b->get_stride(),
        as_cuda_type(b->get_const_values()), as_cuda_type(r->get_values()),
        as_cuda_type(r_tld->get_values()), as_cuda_type(p->get_values()),
        as_cuda_type(q->get_values()), as_cuda_type(u->get_values()),
        as_cuda_type(u_hat->get_values()), as_cuda_type(v_hat->get_values()),
        as_cuda_type(t->get_values()), as_cuda_type(alpha->get_values()),
        as_cuda_type(beta->get_values()), as_cuda_type(gamma->get_values()),
        as_cuda_type(rho_prev->get_values()), as_cuda_type(rho->get_values()),
        as_cuda_type(converged->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_INITIALIZE_KERNEL);


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void test_convergence_kernel(
    size_type num_cols, remove_complex<ValueType> rel_residual_goal,
    const ValueType *__restrict__ tau, const ValueType *__restrict__ orig_tau,
    bool *__restrict__ converged, bool *__restrict__ all_converged)
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
void test_convergence(std::shared_ptr<const GpuExecutor> exec,
                      const matrix::Dense<ValueType> *tau,
                      const matrix::Dense<ValueType> *orig_tau,
                      remove_complex<ValueType> rel_residual_goal,
                      Array<bool> *converged, bool *all_converged)
{
    *all_converged = true;
    auto all_converged_array =
        Array<bool>::view(exec->get_master(), 1, all_converged);
    Array<bool> d_all_converged(exec, all_converged_array);

    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(tau->get_size().num_cols, block_size.x), 1, 1);

    test_convergence_kernel<<<grid_size, block_size, 0, 0>>>(
        tau->get_size().num_cols, rel_residual_goal,
        as_cuda_type(tau->get_const_values()),
        as_cuda_type(orig_tau->get_const_values()),
        as_cuda_type(converged->get_data()),
        as_cuda_type(d_all_converged.get_data()));

    all_converged_array = d_all_converged;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_TEST_CONVERGENCE_KERNEL);


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_1_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ r, ValueType *__restrict__ u,
    ValueType *__restrict__ p, const ValueType *__restrict__ q,
    ValueType *__restrict__ beta, const ValueType *__restrict__ rho,
    const ValueType *__restrict__ rho_prev, const bool *__restrict__ converged)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    const auto col = tidx % stride;

    if (col >= num_cols || tidx >= num_rows * stride || converged[col]) {
        return;
    }
    if (rho_prev[col] != zero<ValueType>()) {
        beta[col] = rho[col] / rho_prev[col];
        u[tidx] = r[tidx] + beta[col] * q[tidx];
        p[tidx] = u[tidx] + beta[col] * (q[tidx] + beta[col] * p[tidx]);
    }
}


template <typename ValueType>
void step_1(std::shared_ptr<const GpuExecutor> exec,
            const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *u,
            matrix::Dense<ValueType> *p, const matrix::Dense<ValueType> *q,
            matrix::Dense<ValueType> *beta, const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *rho_prev,
            const Array<bool> &converged)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(p->get_size().num_rows * p->get_stride(), block_size.x), 1, 1);

    step_1_kernel<<<grid_size, block_size, 0, 0>>>(
        p->get_size().num_rows, p->get_size().num_cols, p->get_stride(),
        as_cuda_type(r->get_const_values()), as_cuda_type(u->get_values()),
        as_cuda_type(p->get_values()), as_cuda_type(q->get_const_values()),
        as_cuda_type(beta->get_values()), as_cuda_type(rho->get_const_values()),
        as_cuda_type(rho_prev->get_const_values()),
        as_cuda_type(converged.get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_1_KERNEL);


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_2_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ u, const ValueType *__restrict__ v_hat,
    ValueType *__restrict__ q, ValueType *__restrict__ t,
    ValueType *__restrict__ alpha, const ValueType *__restrict__ rho,
    const ValueType *__restrict__ gamma, const bool *__restrict__ converged)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    const auto col = tidx % stride;

    if (col >= num_cols || tidx >= num_rows * stride || converged[col]) {
        return;
    }
    if (gamma[col] != zero<ValueType>()) {
        alpha[col] = rho[col] / gamma[col];
        q[tidx] = u[tidx] - alpha[col] * v_hat[tidx];
        t[tidx] = u[tidx] + q[tidx];
    }
}


template <typename ValueType>
void step_2(std::shared_ptr<const GpuExecutor> exec,
            const matrix::Dense<ValueType> *u,
            const matrix::Dense<ValueType> *v_hat, matrix::Dense<ValueType> *q,
            matrix::Dense<ValueType> *t, matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *gamma, const Array<bool> &converged)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(u->get_size().num_rows * u->get_stride(), block_size.x), 1, 1);

    step_2_kernel<<<grid_size, block_size, 0, 0>>>(
        u->get_size().num_rows, u->get_size().num_cols, u->get_stride(),
        as_cuda_type(u->get_const_values()),
        as_cuda_type(v_hat->get_const_values()), as_cuda_type(q->get_values()),
        as_cuda_type(t->get_values()), as_cuda_type(alpha->get_values()),
        as_cuda_type(rho->get_const_values()),
        as_cuda_type(gamma->get_const_values()),
        as_cuda_type(converged.get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_2_KERNEL);


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_3_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    size_type x_stride, const ValueType *__restrict__ t,
    const ValueType *__restrict__ v_hat, ValueType *__restrict__ r,
    ValueType *__restrict__ x, const ValueType *__restrict__ alpha,
    const bool *__restrict__ converged)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    const auto row = tidx / stride;
    const auto col = tidx % stride;
    if (col >= num_cols || tidx >= num_rows * stride || converged[col]) {
        return;
    }
    const auto x_pos = row * x_stride + col;
    auto t_x = x[x_pos] + alpha[col] * v_hat[tidx];
    auto t_r = r[tidx] - alpha[col] * t[tidx];
    x[x_pos] = t_x;
    r[tidx] = t_r;
}


template <typename ValueType>
void step_3(std::shared_ptr<const GpuExecutor> exec,
            const matrix::Dense<ValueType> *t,
            const matrix::Dense<ValueType> *u_hat, matrix::Dense<ValueType> *r,
            matrix::Dense<ValueType> *x, const matrix::Dense<ValueType> *alpha,
            const Array<bool> &converged)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(t->get_size().num_rows * t->get_stride(), block_size.x), 1, 1);

    step_3_kernel<<<grid_size, block_size, 0, 0>>>(
        t->get_size().num_rows, t->get_size().num_cols, t->get_stride(),
        x->get_stride(), as_cuda_type(t->get_const_values()),
        as_cuda_type(u_hat->get_const_values()), as_cuda_type(r->get_values()),
        as_cuda_type(x->get_values()), as_cuda_type(alpha->get_const_values()),
        as_cuda_type(converged.get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_3_KERNEL);


}  // namespace cgs
}  // namespace gpu
}  // namespace kernels
}  // namespace gko
