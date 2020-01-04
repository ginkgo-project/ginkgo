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
