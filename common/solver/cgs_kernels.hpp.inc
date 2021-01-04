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
    ValueType *__restrict__ r_tld, ValueType *__restrict__ p,
    ValueType *__restrict__ q, ValueType *__restrict__ u,
    ValueType *__restrict__ u_hat, ValueType *__restrict__ v_hat,
    ValueType *__restrict__ t, ValueType *__restrict__ alpha,
    ValueType *__restrict__ beta, ValueType *__restrict__ gamma,
    ValueType *__restrict__ rho_prev, ValueType *__restrict__ rho,
    stopping_status *__restrict__ stop_status)
{
    const auto tidx = thread::get_thread_id_flat();

    if (tidx < num_cols) {
        rho[tidx] = zero<ValueType>();
        alpha[tidx] = one<ValueType>();
        beta[tidx] = one<ValueType>();
        gamma[tidx] = one<ValueType>();
        rho_prev[tidx] = one<ValueType>();
        stop_status[tidx].reset();
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
__global__ __launch_bounds__(default_block_size) void step_1_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ r, ValueType *__restrict__ u,
    ValueType *__restrict__ p, const ValueType *__restrict__ q,
    ValueType *__restrict__ beta, const ValueType *__restrict__ rho,
    const ValueType *__restrict__ rho_prev,
    const stopping_status *__restrict__ stop_status)
{
    const auto tidx = thread::get_thread_id_flat();
    const auto col = tidx % stride;

    if (col >= num_cols || tidx >= num_rows * stride ||
        stop_status[col].has_stopped()) {
        return;
    }
    if (rho_prev[col] != zero<ValueType>()) {
        beta[col] = rho[col] / rho_prev[col];
        u[tidx] = r[tidx] + beta[col] * q[tidx];
        p[tidx] = u[tidx] + beta[col] * (q[tidx] + beta[col] * p[tidx]);
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_2_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ u, const ValueType *__restrict__ v_hat,
    ValueType *__restrict__ q, ValueType *__restrict__ t,
    ValueType *__restrict__ alpha, const ValueType *__restrict__ rho,
    const ValueType *__restrict__ gamma,
    const stopping_status *__restrict__ stop_status)
{
    const auto tidx = thread::get_thread_id_flat();
    const auto col = tidx % stride;

    if (col >= num_cols || tidx >= num_rows * stride ||
        stop_status[col].has_stopped()) {
        return;
    }
    if (gamma[col] != zero<ValueType>()) {
        alpha[col] = rho[col] / gamma[col];
        q[tidx] = u[tidx] - alpha[col] * v_hat[tidx];
        t[tidx] = u[tidx] + q[tidx];
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_3_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    size_type x_stride, const ValueType *__restrict__ t,
    const ValueType *__restrict__ v_hat, ValueType *__restrict__ r,
    ValueType *__restrict__ x, const ValueType *__restrict__ alpha,
    const stopping_status *__restrict__ stop_status)
{
    const auto tidx = thread::get_thread_id_flat();
    const auto row = tidx / stride;
    const auto col = tidx % stride;
    if (col >= num_cols || tidx >= num_rows * stride ||
        stop_status[col].has_stopped()) {
        return;
    }
    const auto x_pos = row * x_stride + col;
    auto t_x = x[x_pos] + alpha[col] * v_hat[tidx];
    auto t_r = r[tidx] - alpha[col] * t[tidx];
    x[x_pos] = t_x;
    r[tidx] = t_r;
}