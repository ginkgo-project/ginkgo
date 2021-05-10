/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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
__device__ __forceinline__ void add_scaled(
    const gko::batch_dense::BatchEntry<const ValueType> &alpha,
    const gko::batch_dense::BatchEntry<const ValueType> &x,
    const gko::batch_dense::BatchEntry<ValueType> &y)
{
    if (alpha.num_rhs == 1) {
        for (int i = threadIdx.x; i < x.num_rows; i += blockDim.x) {
            for (int j = 0; j < x.num_rhs; ++j) {
                y.values[i * y.stride + j] +=
                    alpha.values[0] * x.values[i * x.stride + j];
            }
        }
    } else {
        for (int i = threadIdx.x; i < x.num_rows; i += blockDim.x) {
            for (int j = 0; j < x.num_rhs; ++j) {
                y.values[i * y.stride + j] +=
                    alpha.values[j] * x.values[i * x.stride + j];
            }
        }
    }
}

template <typename ValueType>
__global__ __launch_bounds__(default_block_size, sm_multiplier) void add_scaled(
    const gko::batch_dense::UniformBatch<const ValueType> alpha,
    const gko::batch_dense::UniformBatch<const ValueType> x,
    const gko::batch_dense::UniformBatch<ValueType> y)
{
    for (size_type ibatch = blockIdx.x; ibatch < x.num_batch;
         ibatch += gridDim.x) {
        const auto alpha_b = gko::batch::batch_entry(alpha, ibatch);
        const auto x_b = gko::batch::batch_entry(x, ibatch);
        const auto y_b = gko::batch::batch_entry(y, ibatch);
        add_scaled(alpha_b, x_b, y_b);
    }
}


/**
 * Computes the 2-norms of some column vectors in global or shared memory.
 *
 * @param result  Holds norm value for each vector in x.
 */
template <typename ValueType>
__device__ __forceinline__ void compute_norm2(
    const gko::batch_dense::BatchEntry<const ValueType> &x,
    const gko::batch_dense::BatchEntry<remove_complex<ValueType>> &result)
{
    for (int j = threadIdx.x; j < x.num_rhs; j += blockDim.x) {
        result.values[j] = zero<remove_complex<ValueType>>();
    }
    __syncthreads();
    // TODO: replace this with a parallel reduce below
    if (threadIdx.x == 0) {
        for (int i = 0; i < x.num_rows; i++) {
            for (int j = 0; j < x.num_rhs; ++j) {
                result.values[j] += squared_norm(x.values[i * x.stride + j]);
            }
        }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < x.num_rhs; j += blockDim.x) {
        result.values[j] = sqrt(result.values[j]);
    }
}

template <typename ValueType>
__global__
    __launch_bounds__(default_block_size, sm_multiplier) void compute_norm2(
        const gko::batch_dense::UniformBatch<const ValueType> x,
        const gko::batch_dense::UniformBatch<remove_complex<ValueType>> result)
{
    for (size_type ibatch = blockIdx.x; ibatch < x.num_batch;
         ibatch += gridDim.x) {
        const auto x_b = gko::batch::batch_entry(x, ibatch);
        const auto r_b = gko::batch::batch_entry(result, ibatch);
        compute_norm2(x_b, r_b);
    }
}


/**
 * 'Scales' a dense matrix or set of vectors with a dense vector.
 * It is equivalent to multiplication by a diagonal matrix whose diagonal
 * entries are given by the scaling vector.
 *
 * One thread is assigned to each entry.
 */
template <typename ValueType>
__device__ __forceinline__ void batch_scale(
    const gko::batch_dense::BatchEntry<const ValueType> &scale_vec,
    const gko::batch_dense::BatchEntry<ValueType> &a)
{
    for (int iz = threadIdx.x; iz < a.num_rows * a.num_rhs; iz += blockDim.x) {
        const int row = iz / a.num_rhs;
        const int col = iz % a.num_rhs;
        a.values[row * a.stride + col] *= scale_vec.values[row];
    }
}
