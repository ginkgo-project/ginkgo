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
namespace bicgstab {


constexpr int default_block_size = 512;


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void initialize_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ b, ValueType *__restrict__ r,
    ValueType *__restrict__ rr, ValueType *__restrict__ y,
    ValueType *__restrict__ s, ValueType *__restrict__ t,
    ValueType *__restrict__ z, ValueType *__restrict__ v,
    ValueType *__restrict__ p, ValueType *__restrict__ prev_rho,
    ValueType *__restrict__ rho, ValueType *__restrict__ alpha,
    ValueType *__restrict__ beta, ValueType *__restrict__ gamma,
    ValueType *__restrict__ omega, stopping_status *__restrict__ stop_status)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tidx < num_cols) {
        prev_rho[tidx] = one<ValueType>();
        rho[tidx] = one<ValueType>();
        alpha[tidx] = one<ValueType>();
        beta[tidx] = one<ValueType>();
        gamma[tidx] = one<ValueType>();
        omega[tidx] = one<ValueType>();
        stop_status[tidx].reset();
    }

    if (tidx < num_rows * stride) {
        r[tidx] = b[tidx];
        rr[tidx] = zero<ValueType>();
        y[tidx] = zero<ValueType>();
        s[tidx] = zero<ValueType>();
        t[tidx] = zero<ValueType>();
        z[tidx] = zero<ValueType>();
        v[tidx] = zero<ValueType>();
        p[tidx] = zero<ValueType>();
    }
}


template <typename ValueType>
void initialize(std::shared_ptr<const GpuExecutor> exec,
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
        ceildiv(b->get_size().num_rows * b->get_stride(), block_size.x), 1, 1);

    initialize_kernel<<<grid_size, block_size, 0, 0>>>(
        b->get_size().num_rows, b->get_size().num_cols, b->get_stride(),
        as_cuda_type(b->get_const_values()), as_cuda_type(r->get_values()),
        as_cuda_type(rr->get_values()), as_cuda_type(y->get_values()),
        as_cuda_type(s->get_values()), as_cuda_type(t->get_values()),
        as_cuda_type(z->get_values()), as_cuda_type(v->get_values()),
        as_cuda_type(p->get_values()), as_cuda_type(prev_rho->get_values()),
        as_cuda_type(rho->get_values()), as_cuda_type(alpha->get_values()),
        as_cuda_type(beta->get_values()), as_cuda_type(gamma->get_values()),
        as_cuda_type(omega->get_values()),
        as_cuda_type(stop_status->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_INITIALIZE_KERNEL);


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void test_convergence_kernel(
    size_type num_cols, remove_complex<ValueType> rel_residual_goal,
    const ValueType *__restrict__ tau, const ValueType *__restrict__ orig_tau,
    uint8 stoppingId, bool setFinalized,
    stopping_status *__restrict__ stop_status, bool *__restrict__ all_converged,
    bool *__restrict__ one_changed)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (tidx < num_cols) {
        if (abs(tau[tidx]) < rel_residual_goal * abs(orig_tau[tidx])) {
            stop_status[tidx].converge(stoppingId, setFinalized);
            *one_changed = true;
        }
        // because only false is written to all_converged, write conflicts
        // should not cause any problem
        else if (!stop_status[tidx].has_stopped()) {
            *all_converged = false;
        }
    }
}

template <typename ValueType>
void test_convergence(std::shared_ptr<const GpuExecutor> exec,
                      const matrix::Dense<ValueType> *tau,
                      const matrix::Dense<ValueType> *orig_tau,
                      remove_complex<ValueType> rel_residual_goal,
                      uint8 stoppingId, bool setFinalized,
                      Array<stopping_status> *stop_status, bool *all_converged,
                      bool *one_changed)
{
    Array<bool> d_all_converged(exec, 1);
    Array<bool> all_converged_array(exec->get_master());
    // initialize all_converged with true
    *all_converged = true;
    all_converged_array.manage(1, all_converged);
    d_all_converged = all_converged_array;

    Array<bool> d_one_changed(exec, 1);
    Array<bool> one_changed_array(exec->get_master());
    *one_changed = false;
    one_changed_array.manage(1, one_changed);
    d_one_changed = one_changed_array;

    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(tau->get_size().num_cols, block_size.x), 1, 1);

    test_convergence_kernel<<<grid_size, block_size, 0, 0>>>(
        tau->get_size().num_cols, rel_residual_goal,
        as_cuda_type(tau->get_const_values()),
        as_cuda_type(orig_tau->get_const_values()), stoppingId, setFinalized,
        as_cuda_type(stop_status->get_data()),
        as_cuda_type(d_all_converged.get_data()),
        as_cuda_type(d_one_changed.get_data()));

    all_converged_array = d_all_converged;
    all_converged_array.release();

    one_changed_array = d_one_changed;
    one_changed_array.release();
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BICGSTAB_TEST_CONVERGENCE_KERNEL);


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_1_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ r, ValueType *__restrict__ p,
    const ValueType *__restrict__ v, const ValueType *__restrict__ rho,
    const ValueType *__restrict__ prev_rho, const ValueType *__restrict__ alpha,
    const ValueType *__restrict__ omega,
    const stopping_status *__restrict__ stop_status)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    const auto col = tidx % stride;
    if (col >= num_cols || tidx >= num_rows * stride ||
        stop_status[col].has_stopped()) {
        return;
    }
    auto res = r[tidx];
    if (prev_rho[col] * omega[col] != zero<ValueType>()) {
        const auto tmp = (rho[col] / prev_rho[col]) * (alpha[col] / omega[col]);
        res += tmp * (p[tidx] - omega[col] * v[tidx]);
    }
    p[tidx] = res;
}


template <typename ValueType>
void step_1(std::shared_ptr<const GpuExecutor> exec,
            const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *p,
            const matrix::Dense<ValueType> *v,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *prev_rho,
            const matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *omega,
            const Array<stopping_status> &stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(r->get_size().num_rows * r->get_stride(), block_size.x), 1, 1);

    step_1_kernel<<<grid_size, block_size, 0, 0>>>(
        r->get_size().num_rows, r->get_size().num_cols, r->get_stride(),
        as_cuda_type(r->get_const_values()), as_cuda_type(p->get_values()),
        as_cuda_type(v->get_const_values()),
        as_cuda_type(rho->get_const_values()),
        as_cuda_type(prev_rho->get_const_values()),
        as_cuda_type(alpha->get_const_values()),
        as_cuda_type(omega->get_const_values()),
        as_cuda_type(stop_status.get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_1_KERNEL);


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_2_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ r, ValueType *__restrict__ s,
    const ValueType *__restrict__ v, const ValueType *__restrict__ rho,
    ValueType *__restrict__ alpha, const ValueType *__restrict__ beta,
    const stopping_status *__restrict__ stop_status)
{
    const size_type tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    const size_type col = tidx % stride;
    if (col >= num_cols || tidx >= num_rows * stride ||
        stop_status[col].has_stopped()) {
        return;
    }
    auto t_alpha = zero<ValueType>();
    auto t_s = r[tidx];
    if (beta[col] != zero<ValueType>()) {
        t_alpha = rho[col] / beta[col];
        t_s -= t_alpha * v[tidx];
    }
    alpha[col] = t_alpha;
    s[tidx] = t_s;
}


template <typename ValueType>
void step_2(std::shared_ptr<const GpuExecutor> exec,
            const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *s,
            const matrix::Dense<ValueType> *v,
            const matrix::Dense<ValueType> *rho,
            matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *beta,
            const Array<stopping_status> &stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(r->get_size().num_rows * r->get_stride(), block_size.x), 1, 1);

    step_2_kernel<<<grid_size, block_size, 0, 0>>>(
        r->get_size().num_rows, r->get_size().num_cols, r->get_stride(),
        as_cuda_type(r->get_const_values()), as_cuda_type(s->get_values()),
        as_cuda_type(v->get_const_values()),
        as_cuda_type(rho->get_const_values()),
        as_cuda_type(alpha->get_values()),
        as_cuda_type(beta->get_const_values()),
        as_cuda_type(stop_status.get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_2_KERNEL);


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_3_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    size_type x_stride, ValueType *__restrict__ x, ValueType *__restrict__ r,
    const ValueType *__restrict__ s, const ValueType *__restrict__ t,
    const ValueType *__restrict__ y, const ValueType *__restrict__ z,
    const ValueType *__restrict__ alpha, const ValueType *__restrict__ beta,
    const ValueType *__restrict__ gamma, ValueType *__restrict__ omega,
    const stopping_status *__restrict__ stop_status)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    const auto row = tidx / stride;
    const auto col = tidx % stride;
    if (col >= num_cols || tidx >= num_rows * stride ||
        stop_status[col].has_stopped()) {
        return;
    }
    const auto x_pos = row * x_stride + col;
    auto t_omega = zero<ValueType>();
    auto t_x = x[x_pos] + alpha[col] * y[tidx];
    auto t_r = s[tidx];
    if (beta[col] != zero<ValueType>()) {
        t_omega = gamma[col] / beta[col];
        t_x += t_omega * z[tidx];
        t_r -= t_omega * t[tidx];
    }
    omega[col] = t_omega;
    x[x_pos] = t_x;
    r[tidx] = t_r;
}


template <typename ValueType>
void step_3(
    std::shared_ptr<const GpuExecutor> exec, matrix::Dense<ValueType> *x,
    matrix::Dense<ValueType> *r, const matrix::Dense<ValueType> *s,
    const matrix::Dense<ValueType> *t, const matrix::Dense<ValueType> *y,
    const matrix::Dense<ValueType> *z, const matrix::Dense<ValueType> *alpha,
    const matrix::Dense<ValueType> *beta, const matrix::Dense<ValueType> *gamma,
    matrix::Dense<ValueType> *omega, const Array<stopping_status> &stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(r->get_size().num_rows * r->get_stride(), block_size.x), 1, 1);

    step_3_kernel<<<grid_size, block_size, 0, 0>>>(
        r->get_size().num_rows, r->get_size().num_cols, r->get_stride(),
        x->get_stride(), as_cuda_type(x->get_values()),
        as_cuda_type(r->get_values()), as_cuda_type(s->get_const_values()),
        as_cuda_type(t->get_const_values()),
        as_cuda_type(y->get_const_values()),
        as_cuda_type(z->get_const_values()),
        as_cuda_type(alpha->get_const_values()),
        as_cuda_type(beta->get_const_values()),
        as_cuda_type(gamma->get_const_values()),
        as_cuda_type(omega->get_values()),
        as_cuda_type(stop_status.get_const_data()));
}
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_3_KERNEL);


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void finalize_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    size_type x_stride, ValueType *__restrict__ x,
    const ValueType *__restrict__ y, const ValueType *__restrict__ alpha,
    stopping_status *__restrict__ stop_status)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    const auto row = tidx / stride;
    const auto col = tidx % stride;
    if (col >= num_cols || tidx >= num_rows * stride ||
        stop_status[col].is_finalized() || !stop_status[col].has_stopped()) {
        return;
    }
    const auto x_pos = row * x_stride + col;
    x[x_pos] = x[x_pos] + alpha[col] * y[tidx];
    stop_status[col].finalize();
}


template <typename ValueType>
void finalize(std::shared_ptr<const GpuExecutor> exec,
              matrix::Dense<ValueType> *x, const matrix::Dense<ValueType> *y,
              const matrix::Dense<ValueType> *alpha,
              Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(y->get_size().num_rows * y->get_stride(), block_size.x), 1, 1);

    finalize_kernel<<<grid_size, block_size, 0, 0>>>(
        y->get_size().num_rows, y->get_size().num_cols, y->get_stride(),
        x->get_stride(), as_cuda_type(x->get_values()),
        as_cuda_type(y->get_const_values()),
        as_cuda_type(alpha->get_const_values()),
        as_cuda_type(stop_status->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_FINALIZE_KERNEL);


}  // namespace bicgstab
}  // namespace gpu
}  // namespace kernels
}  // namespace gko
