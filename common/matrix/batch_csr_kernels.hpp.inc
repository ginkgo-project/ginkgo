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


/**
 * Device kernel for SpMV of one CSR matrix in a batch.
 *
 * Assumes the input and output multi-vectors are stored row-major.
 */
template <typename ValueType>
__device__ __forceinline__ void spmv_kernel(
    const gko::batch_csr::BatchEntry<const ValueType> &a,
    const gko::batch_dense::BatchEntry<const ValueType> &b,
    const gko::batch_dense::BatchEntry<ValueType> &c)
{
    for (int row = threadIdx.x; row < a.num_rows; row += blockDim.x) {
        for (int j = 0; j < c.num_rhs; ++j) {
            c.values[row * c.stride + j] = zero<ValueType>();
        }
        for (auto k = a.row_ptrs[row]; k < a.row_ptrs[row + 1]; ++k) {
            auto val = a.values[k];
            auto col = a.col_idxs[k];
            for (int j = 0; j < b.num_rhs; ++j) {
                c.values[row * c.stride + j] +=
                    val * b.values[col * b.stride + j];
            }
        }
    }
}

template <typename ValueType>
__global__ __launch_bounds__(default_block_size, sm_multiplier) void spmv(
    const gko::batch_csr::UniformBatch<const ValueType> a,
    const batch_dense::UniformBatch<const ValueType> b,
    const batch_dense::UniformBatch<ValueType> c)
{
    for (size_type ibatch = blockIdx.x; ibatch < a.num_batch;
         ibatch += gridDim.x) {
        const auto a_b = gko::batch::batch_entry(a, ibatch);
        const auto b_b = gko::batch::batch_entry(b, ibatch);
        const auto c_b = gko::batch::batch_entry(c, ibatch);
        spmv_kernel(a_b, b_b, c_b);
    }
}


/**
 * Device kernel for 'advanced' SpMV of one CSR matrix in a batch.
 *
 * Assumes the input and output multi-vectors are stored row-major.
 */
template <typename ValueType>
__device__ __forceinline__ void advanced_spmv_kernel(
    const ValueType alpha, const gko::batch_csr::BatchEntry<const ValueType> &a,
    const gko::batch_dense::BatchEntry<const ValueType> &b,
    const ValueType beta, const gko::batch_dense::BatchEntry<ValueType> &c)
{
    for (int row = threadIdx.x; row < a.num_rows; row += blockDim.x) {
        for (int j = 0; j < c.num_rhs; ++j) {
            c.values[row * c.stride + j] *= beta;
        }
        for (int k = a.row_ptrs[row]; k < a.row_ptrs[row + 1]; ++k) {
            const auto val = a.values[k];
            const auto col = a.col_idxs[k];
            for (int j = 0; j < c.num_rhs; ++j) {
                c.values[row * c.stride + j] +=
                    alpha * val * b.values[col * b.stride + j];
            }
        }
    }
}

template <typename ValueType>
__global__
    __launch_bounds__(default_block_size, sm_multiplier) void advanced_spmv(
        const batch_dense::UniformBatch<const ValueType> alpha,
        const gko::batch_csr::UniformBatch<const ValueType> a,
        const batch_dense::UniformBatch<const ValueType> b,
        const batch_dense::UniformBatch<const ValueType> beta,
        const batch_dense::UniformBatch<ValueType> c)
{
    for (size_type ibatch = blockIdx.x; ibatch < a.num_batch;
         ibatch += gridDim.x) {
        const auto a_b = gko::batch::batch_entry(a, ibatch);
        const auto b_b = gko::batch::batch_entry(b, ibatch);
        const auto c_b = gko::batch::batch_entry(c, ibatch);
        const auto alpha_b = gko::batch::batch_entry(alpha, ibatch);
        const auto beta_b = gko::batch::batch_entry(beta, ibatch);
        advanced_spmv_kernel(alpha_b.values[0], a_b, b_b, beta_b.values[0],
                             c_b);
    }
}


/**
 * Scales one uniform CSR matrix two dense
 * vectors for row and column scaling respectively.
 *
 * One warp is assigned to each row.
 */
template <typename ValueType>
__device__ __forceinline__ void batch_scale(
    const gko::batch_dense::BatchEntry<const ValueType> &left_scale,
    const gko::batch_dense::BatchEntry<const ValueType> &right_scale,
    const gko::batch_csr::BatchEntry<ValueType> &a)
{
    constexpr auto warp_size = config::warp_size;
    const auto tile =
        group::tiled_partition<warp_size>(group::this_thread_block());
    const int tile_rank = threadIdx.x / warp_size;
    const int num_tiles = (blockDim.x - 1) / warp_size + 1;

    for (int i_row = tile_rank; i_row < a.num_rows; i_row += num_tiles) {
        const ValueType rowscale = left_scale.values[i_row];
        for (int iz = a.row_ptrs[i_row] + tile.thread_rank();
             iz < a.row_ptrs[i_row + 1]; iz += warp_size) {
            a.values[iz] *= rowscale * right_scale.values[a.col_idxs[iz]];
        }
    }
}


/**
 * Scales an entire batch of uniform CSR matrices with common dense
 * vectors for row and column scaling.
 *
 * One warp is assigned to each row.
 */
template <typename ValueType>
__global__ void uniform_batch_scale(
    const batch_dense::UniformBatch<const ValueType> left,
    const batch_dense::UniformBatch<const ValueType> right,
    const gko::batch_csr::UniformBatch<ValueType> a)
{
    for (size_type batch_id = blockIdx.x; batch_id < a.num_batch;
         batch_id += gridDim.x) {
        const auto a_b = gko::batch::batch_entry(a, batch_id);
        const auto left_b = gko::batch::batch_entry(left, batch_id);
        const auto right_b = gko::batch::batch_entry(right, batch_id);
        batch_scale(left_b, right_b, a_b);
    }
}
