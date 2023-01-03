/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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


// grid_nrows is the number of rows handled in the whole grid at the same time.
// Thus, the threads whose index is larger than grid_nrows * nrhs are not used.
// Let the thread handle the same col (has same scalar) in whole loop.
template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void kcycle_step_1_kernel(
    const size_type num_rows, const size_type nrhs, const size_type stride,
    const size_type grid_nrows, const ValueType* __restrict__ alpha,
    const ValueType* __restrict__ rho, const ValueType* __restrict__ v,
    ValueType* __restrict__ g, ValueType* __restrict__ d,
    ValueType* __restrict__ e)
{
    const auto tidx = thread::get_thread_id_flat();
    const auto col = tidx % nrhs;
    const auto num_elems = grid_nrows * nrhs;
    if (tidx >= num_elems) {
        return;
    }
    const auto total_elems = num_rows * stride;
    const auto grid_stride = grid_nrows * stride;
    const auto temp = alpha[col] / rho[col];
    const bool update = is_finite(temp);
    for (auto idx = tidx / nrhs * stride + col; idx < total_elems;
         idx += grid_stride) {
        auto store_e = e[idx];
        if (update) {
            g[idx] -= temp * v[idx];
            store_e *= temp;
            e[idx] = store_e;
        }
        d[idx] = store_e;
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void kcycle_step_2_kernel(
    const size_type num_rows, const size_type nrhs, const size_type stride,
    const size_type grid_nrows, const ValueType* __restrict__ alpha,
    const ValueType* __restrict__ rho, const ValueType* __restrict__ gamma,
    const ValueType* __restrict__ beta, const ValueType* __restrict__ zeta,
    const ValueType* __restrict__ d, ValueType* __restrict__ e)
{
    const auto tidx = thread::get_thread_id_flat();
    const auto col = tidx % nrhs;
    const auto num_elems = grid_nrows * nrhs;
    if (tidx >= num_elems) {
        return;
    }
    const auto total_elems = num_rows * stride;
    const auto grid_stride = grid_nrows * stride;
    const auto scalar_d =
        zeta[col] / (beta[col] - gamma[col] * gamma[col] / rho[col]);
    const auto scalar_e = one<ValueType>() - gamma[col] / alpha[col] * scalar_d;
    if (is_finite(scalar_d) && is_finite(scalar_e)) {
        for (auto idx = tidx / nrhs * stride + col; idx < total_elems;
             idx += grid_stride) {
            e[idx] = scalar_e * e[idx] + scalar_d * d[idx];
        }
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void kcycle_check_stop_kernel(
    const size_type nrhs, const ValueType* __restrict__ old_norm,
    const ValueType* __restrict__ new_norm, const ValueType rel_tol,
    bool* __restrict__ is_stop)
{
    auto tidx = thread::get_thread_id_flat();
    if (tidx >= nrhs) {
        return;
    }
    if (new_norm[tidx] > rel_tol * old_norm[tidx]) {
        *is_stop = false;
    }
}
