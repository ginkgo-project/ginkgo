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
 *
 *   Scales the vectors in global or shared memory with a factor of alpha(alpha
 * is in global memory or shared memory or even the thread's local memory)
 *
 */
// template <typename ValueType>
//__device__ __forceinline__ void scale(const ValueType &alpha,
//    const int num_rows, ValueType *const x)
//{
//    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
//        x[row] = alpha * x[row];
//    }
//}
//
// template <typename ValueType>
//__global__ __launch_bounds__(default_block_size, sm_multiplier) void scale(
//    const gko::batch_dense::UniformBatch<const ValueType> alpha,
//    const gko::batch_dense::UniformBatch<ValueType> x)
//{
//    for (size_type ibatch = blockIdx.x; ibatch < x.num_batch;
//         ibatch += gridDim.x) {
//        const auto alpha_b = gko::batch::batch_entry(alpha, ibatch);
//        const auto x_b = gko::batch::batch_entry(x, ibatch);
//        scale(alpha_b.value[0], x_b.num_rows, x_b.values);
//    }
//}
//
//
// template <typename ValueType>
//__device__ __forceinline__ void add_scaled(
//    const ValueType &alpha, const int num_rows,
//	const ValueType *const x,
//    ValueType *const y)
//{
//    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
//        y[row] += alpha * x[row];
//    }
//}
//
// template <typename ValueType>
//__global__ __launch_bounds__(default_block_size, sm_multiplier) void
// add_scaled(
//    const gko::batch_dense::UniformBatch<const ValueType> alpha,
//    const gko::batch_dense::UniformBatch<const ValueType> x,
//    const gko::batch_dense::UniformBatch<ValueType> y)
//{
//    for (size_type ibatch = blockIdx.x; ibatch < x.num_batch;
//         ibatch += gridDim.x) {
//        const auto alpha_b = gko::batch::batch_entry(alpha, ibatch);
//        const auto x_b = gko::batch::batch_entry(x, ibatch);
//        const auto y_b = gko::batch::batch_entry(y, ibatch);
//        add_scaled(alpha_b.values[0], x_b.num_rows, x_b.values, y_b.values);
//    }
//}


/**
 * Computes the 2-norms of some column vectors in global or shared memory.
 *
 * @param x  A row-major multivector with nrhs columns.
 * @param result  Holds norm value for each vector in x.
 */
template <typename ValueType>
__device__ __forceinline__ void compute_norm2(
    group::thread_block_tile<config::warp_size> &warp_grp, const int num_rows,
    const ValueType *const x, remove_complex<ValueType> &result)
{
    using real_type = typename gko::remove_complex<ValueType>;
    real_type val = zero<real_type>();

    for (int r = warp_grp.thread_rank(); r < num_rows; r += warp_grp.size()) {
        val += squared_norm(x[r]);
    }

    // subwarp_grp level reduction
#pragma unroll
    for (int j = config::warp_size / 2; j > 0; j /= 2) {
        val += warp_grp.shfl_down(val, j);
    }

    if (warp_grp.thread_rank() == 0) {
        result = sqrt(val);
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
        if (threadIdx.x / config::warp_size == 0) {
            compute_norm2(group::tiled_partition<config::warp_size>(
                              group::this_thread_block()),
                          x_b.num_rows, x_b.values, r_b.values[0]);
        }
    }
}


/**
 * Copies the values of one multi-vector into another.
 *
 * Note that the output multi-vector should already have memory allocated
 * and stride set.
 */
template <typename ValueType>
__device__ __forceinline__ void copy(const int num_rows,
                                     const ValueType *const in,
                                     ValueType *const out)
{
    for (int iz = threadIdx.x; iz < num_rows; iz += blockDim.x) {
        out[iz] = in[iz];
    }
}


/**
 * Computes the dot product of some column vectors in global or shared memory.
 *
 * @param result  Holds dot product value for vector in x and y.
 */
template <typename ValueType>
__device__ __forceinline__ void compute_dot_product(
    group::thread_block_tile<config::warp_size> &warp_grp, const int num_rows,
    const ValueType *const x, const ValueType *const y, ValueType &result)
{
    ValueType val = zero<ValueType>();

    for (int r = warp_grp.thread_rank(); r < num_rows; r += warp_grp.size()) {
        val += conj(x[r]) * y[r];
    }

    // subwarp_grp level reduction
#pragma unroll
    for (int j = config::warp_size / 2; j > 0; j /= 2) {
        val += warp_grp.shfl_down(val, j);
    }

    if (warp_grp.thread_rank() == 0) {
        result = val;
    }
}


// clang-format off
template <typename ValueType>
__global__ __launch_bounds__(default_block_size, sm_multiplier)
    // clang-format on
    void compute_dot_product(
        const gko::batch_dense::UniformBatch<const ValueType> x,
        const gko::batch_dense::UniformBatch<const ValueType> y,
        const gko::batch_dense::UniformBatch<ValueType> result)
{
    for (size_type ibatch = blockIdx.x; ibatch < x.num_batch;
         ibatch += gridDim.x) {
        const auto x_b = gko::batch::batch_entry(x, ibatch);
        const auto y_b = gko::batch::batch_entry(y, ibatch);
        const auto r_b = gko::batch::batch_entry(result, ibatch);
        compute_dot_product(x_b.num_rows, x_b.values, y_b.values,
                            r_b.values[0]);
    }
}
