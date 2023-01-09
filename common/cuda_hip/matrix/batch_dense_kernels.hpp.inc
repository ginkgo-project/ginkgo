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


template <typename ValueType>
using BatchEntry = gko::batch_dense::BatchEntry<ValueType>;


template <typename ValueType>
__device__ __forceinline__ void single_matvec_kernel(
    const gko::batch_dense::BatchEntry<const ValueType>& a,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ c)
{
    constexpr auto tile_size = config::warp_size;

    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<tile_size>(thread_block);
    const auto subwarp_grp_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_subwarp_grps_per_block = ceildiv(blockDim.x, tile_size);

    for (int row = subwarp_grp_id; row < a.num_rows;
         row += num_subwarp_grps_per_block) {
        ValueType temp = zero<ValueType>();
        for (int j = subwarp_grp.thread_rank(); j < a.num_rhs;
             j += subwarp_grp.size()) {
            const ValueType val = a.values[row * a.stride + j];
            temp += val * b[j];
        }

#pragma unroll
        for (int i = static_cast<int>(tile_size) / 2; i > 0; i /= 2) {
            temp += subwarp_grp.shfl_down(temp, i);
        }

        if (subwarp_grp.thread_rank() == 0) {
            c[row] = temp;
        }
    }
}

template <typename ValueType>
__global__ __launch_bounds__(default_block_size, sm_multiplier) void mv(
    const gko::batch_dense::UniformBatch<const ValueType> a,
    const gko::batch_dense::UniformBatch<const ValueType> b,
    const gko::batch_dense::UniformBatch<ValueType> c)
{
    for (size_type ibatch = blockIdx.x; ibatch < a.num_batch;
         ibatch += gridDim.x) {
        const auto a_b = gko::batch::batch_entry(a, ibatch);
        const auto b_b = gko::batch::batch_entry(b, ibatch);
        const auto c_b = gko::batch::batch_entry(c, ibatch);
        single_matvec_kernel(a_b, b_b.values, c_b.values);
    }
}


template <typename ValueType>
__device__ __forceinline__ void single_advanced_matvec_kernel(
    const ValueType alpha,
    const gko::batch_dense::BatchEntry<const ValueType>& a,
    const ValueType* const __restrict__ b, const ValueType beta,
    ValueType* const __restrict__ c)
{
    constexpr auto tile_size = config::warp_size;

    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<tile_size>(thread_block);
    const auto subwarp_grp_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_subwarp_grps_per_block = ceildiv(blockDim.x, tile_size);

    for (int row = subwarp_grp_id; row < a.num_rows;
         row += num_subwarp_grps_per_block) {
        ValueType temp = zero<ValueType>();
        for (int j = subwarp_grp.thread_rank(); j < a.num_rhs;
             j += subwarp_grp.size()) {
            const ValueType val = a.values[row * a.stride + j];
            temp += alpha * val * b[j];
        }

#pragma unroll
        for (int i = static_cast<int>(tile_size) / 2; i > 0; i /= 2) {
            temp += subwarp_grp.shfl_down(temp, i);
        }

        if (subwarp_grp.thread_rank() == 0) {
            c[row] = temp + beta * c[row];
        }
    }
}

template <typename ValueType>
__global__
    __launch_bounds__(default_block_size, sm_multiplier) void advanced_mv(
        const gko::batch_dense::UniformBatch<const ValueType> alpha,
        const gko::batch_dense::UniformBatch<const ValueType> a,
        const gko::batch_dense::UniformBatch<const ValueType> b,
        const gko::batch_dense::UniformBatch<const ValueType> beta,
        const gko::batch_dense::UniformBatch<ValueType> c)
{
    for (size_type ibatch = blockIdx.x; ibatch < a.num_batch;
         ibatch += gridDim.x) {
        const auto a_b = gko::batch::batch_entry(a, ibatch);
        const auto b_b = gko::batch::batch_entry(b, ibatch);
        const auto c_b = gko::batch::batch_entry(c, ibatch);
        const auto alpha_b = gko::batch::batch_entry(alpha, ibatch);
        const auto beta_b = gko::batch::batch_entry(beta, ibatch);
        single_advanced_matvec_kernel(alpha_b.values[0], a_b, b_b.values,
                                      beta_b.values[0], c_b.values);
    }
}

/**
 * Scales the vectors in global or shared memory with a factor of alpha (alpha
 * is in global memory or shared memory)
 */
template <typename ValueType>
__device__ __forceinline__ void scale(const BatchEntry<const ValueType>& alpha,
                                      const BatchEntry<ValueType>& x)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = threadIdx.x; li < max_li; li += blockDim.x) {
        const int row = li / x.num_rhs;
        const int col = li % x.num_rhs;

        if (alpha.num_rhs == 1) {
            x.values[row * x.stride + col] =
                alpha.values[0] * x.values[row * x.stride + col];
        } else {
            x.values[row * x.stride + col] =
                alpha.values[col] * x.values[row * x.stride + col];
        }
    }
}

template <typename ValueType>
__global__ __launch_bounds__(default_block_size, sm_multiplier) void scale(
    const gko::batch_dense::UniformBatch<const ValueType> alpha,
    const gko::batch_dense::UniformBatch<ValueType> x)
{
    for (size_type ibatch = blockIdx.x; ibatch < x.num_batch;
         ibatch += gridDim.x) {
        const auto alpha_b = gko::batch::batch_entry(alpha, ibatch);
        const auto x_b = gko::batch::batch_entry(x, ibatch);
        scale(alpha_b, x_b);
    }
}

template <typename ValueType>
__device__ __forceinline__ void add_scaled(
    const gko::batch_dense::BatchEntry<const ValueType>& alpha,
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<ValueType>& y)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = threadIdx.x; li < max_li; li += blockDim.x) {
        const int row = li / x.num_rhs;
        const int col = li % x.num_rhs;

        if (alpha.num_rhs == 1) {
            y.values[row * y.stride + col] +=
                alpha.values[0] * x.values[row * x.stride + col];
        } else {
            y.values[row * y.stride + col] +=
                alpha.values[col] * x.values[row * x.stride + col];
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


template <typename ValueType>
__device__ __forceinline__ void add_scale(
    const gko::batch_dense::BatchEntry<const ValueType>& alpha,
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<const ValueType>& beta,
    const gko::batch_dense::BatchEntry<ValueType>& y)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = threadIdx.x; li < max_li; li += blockDim.x) {
        const int row = li / x.num_rhs;
        const int col = li % x.num_rhs;

        if (alpha.num_rhs == 1) {
            y.values[row * y.stride + col] =
                alpha.values[0] * x.values[row * x.stride + col] +
                beta.values[0] * y.values[row * y.stride + col];
        } else {
            y.values[row * y.stride + col] =
                alpha.values[col] * x.values[row * x.stride + col] +
                beta.values[col] * y.values[row * y.stride + col];
        }
    }
}

template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void add_scale(
    const gko::batch_dense::UniformBatch<const ValueType> alpha,
    const gko::batch_dense::UniformBatch<const ValueType> x,
    const gko::batch_dense::UniformBatch<const ValueType> beta,
    const gko::batch_dense::UniformBatch<ValueType> y)
{
    for (size_type ibatch = blockIdx.x; ibatch < x.num_batch;
         ibatch += gridDim.x) {
        const auto alpha_b = gko::batch::batch_entry(alpha, ibatch);
        const auto beta_b = gko::batch::batch_entry(beta, ibatch);
        const auto x_b = gko::batch::batch_entry(x, ibatch);
        const auto y_b = gko::batch::batch_entry(y, ibatch);
        add_scale(alpha_b, x_b, beta_b, y_b);
    }
}


namespace {

template <typename ValueType>
__device__ __forceinline__ void one_norm2(
    const gko::batch_dense::BatchEntry<const ValueType>& x, const int rhs_index,
    const gko::batch_dense::BatchEntry<remove_complex<ValueType>>& result,
    group::thread_block_tile<config::warp_size>& subwarp_grp)
{
    using real_type = typename gko::remove_complex<ValueType>;
    real_type val = zero<real_type>();

    for (int r = subwarp_grp.thread_rank(); r < x.num_rows;
         r += subwarp_grp.size()) {
        val += squared_norm(x.values[r * x.stride + rhs_index]);
    }

    // subwarp_grp level reduction
#pragma unroll
    for (int j = config::warp_size / 2; j > 0; j /= 2) {
        val += subwarp_grp.shfl_down(val, j);
    }

    if (subwarp_grp.thread_rank() == 0) {
        result.values[rhs_index] = sqrt(val);
    }
}

}  // namespace

/**
 * Computes the 2-norms of some column vectors in global or shared memory.
 *
 * @param x  A row-major multivector with nrhs columns.
 * @param result  Holds norm value for each vector in x.
 */
template <typename ValueType>
__device__ __forceinline__ void compute_norm2(
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<remove_complex<ValueType>>& result)
{
    constexpr auto tile_size = config::warp_size;
    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<tile_size>(thread_block);
    const auto subwarp_grp_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_subwarp_grps_per_block = ceildiv(blockDim.x, tile_size);

    for (int rhs_index = subwarp_grp_id; rhs_index < x.num_rhs;
         rhs_index += num_subwarp_grps_per_block) {
        one_norm2(x, rhs_index, result, subwarp_grp);
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
__device__ __forceinline__ void batch_scale(const int num_rows,
                                            const size_type stride,
                                            const int num_rhs,
                                            const ValueType* const scale_vec,
                                            ValueType* const a)
{
    for (int iz = threadIdx.x; iz < num_rows * num_rhs; iz += blockDim.x) {
        const int row = iz / num_rhs;
        const int col = iz % num_rhs;
        a[row * stride + col] *= scale_vec[row];
    }
}

template <typename ValueType>
__global__ void uniform_batch_scale(
    const int nrows, const size_type stride, const int nrhs,
    const size_type num_batch, const ValueType* const __restrict__ scale_vec,
    ValueType* const __restrict__ a)
{
    for (size_type batch_id = blockIdx.x; batch_id < num_batch;
         batch_id += gridDim.x) {
        // const auto a_b = gko::batch::batch_entry(a, batch_id);
        const auto a_ptr =
            gko::batch::batch_entry_ptr(a, stride, nrows, batch_id);
        const auto s_ptr =
            gko::batch::batch_entry_ptr(scale_vec, 1, nrows, batch_id);
        batch_scale(nrows, stride, nrhs, s_ptr, a_ptr);
    }
}

template <typename ValueType>
__device__ __forceinline__ void batch_scale(
    const int num_rows, const size_type stride, const int num_rhs,
    const ValueType* const left_scale_vec,
    const ValueType* const right_scale_vec, ValueType* const a)
{
    for (int iz = threadIdx.x; iz < num_rows * num_rhs; iz += blockDim.x) {
        const int row = iz / num_rhs;
        const int col = iz % num_rhs;
        a[row * stride + col] *= left_scale_vec[row] * right_scale_vec[col];
    }
}

template <typename ValueType>
__global__ void uniform_batch_scale(
    const int nrows, const size_type stride, const int nrhs,
    const size_type num_batch,
    const ValueType* const __restrict__ left_scale_vec,
    const ValueType* const __restrict__ right_scale_vec,
    ValueType* const __restrict__ a)
{
    for (size_type batch_id = blockIdx.x; batch_id < num_batch;
         batch_id += gridDim.x) {
        // const auto a_b = gko::batch::batch_entry(a, batch_id);
        const auto a_ptr =
            gko::batch::batch_entry_ptr(a, stride, nrows, batch_id);
        const auto ls_ptr =
            gko::batch::batch_entry_ptr(left_scale_vec, 1, nrows, batch_id);
        const auto rs_ptr =
            gko::batch::batch_entry_ptr(right_scale_vec, 1, nrhs, batch_id);
        batch_scale(nrows, stride, nrhs, ls_ptr, rs_ptr, a_ptr);
    }
}


/**
 * Copies the values of one multi-vector into another.
 *
 * Note that the output multi-vector should already have memory allocated
 * and stride set.
 */
template <typename ValueType>
__device__ __forceinline__ void copy(
    const gko::batch_dense::BatchEntry<const ValueType>& in,
    const gko::batch_dense::BatchEntry<ValueType>& out)
{
    for (int iz = threadIdx.x; iz < in.num_rows * in.num_rhs;
         iz += blockDim.x) {
        const int i = iz / in.num_rhs;
        const int j = iz % in.num_rhs;
        out.values[i * out.stride + j] = in.values[i * in.stride + j];
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size, sm_multiplier) void copy(
    const gko::batch_dense::UniformBatch<const ValueType> src,
    const gko::batch_dense::UniformBatch<ValueType> dst)
{
    for (size_type ibatch = blockIdx.x; ibatch < src.num_batch;
         ibatch += gridDim.x) {
        const auto dst_b = gko::batch::batch_entry(dst, ibatch);
        const auto src_b = gko::batch::batch_entry(src, ibatch);
        copy(src_b, dst_b);
    }
}


namespace {

template <typename ValueType>
__device__ __forceinline__ void one_dot(
    const BatchEntry<const ValueType>& x, const BatchEntry<const ValueType>& y,
    const int rhs_index, const BatchEntry<ValueType>& result,
    group::thread_block_tile<config::warp_size>& subwarp_grp)
{
    ValueType val = zero<ValueType>();

    for (int r = subwarp_grp.thread_rank(); r < x.num_rows;
         r += subwarp_grp.size()) {
        val += conj(x.values[r * x.stride + rhs_index]) *
               y.values[r * y.stride + rhs_index];
    }

    // subwarp_grp level reduction
#pragma unroll
    for (int j = config::warp_size / 2; j > 0; j /= 2) {
        val += subwarp_grp.shfl_down(val, j);
    }

    if (subwarp_grp.thread_rank() == 0) {
        result.values[rhs_index] = val;
    }
}

}  // namespace

/**
 * Computes the dot product of some column vectors in global or shared memory.
 *
 * @param result  Holds dot product value for vector in x and y.
 */
template <typename ValueType>
__device__ __forceinline__ void compute_dot_product(
    const BatchEntry<const ValueType>& x, const BatchEntry<const ValueType>& y,
    const BatchEntry<ValueType>& result)
{
    constexpr auto tile_size = config::warp_size;
    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<tile_size>(thread_block);
    const auto subwarp_grp_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_subwarp_grps_per_block = ceildiv(blockDim.x, tile_size);

    for (int rhs_index = subwarp_grp_id; rhs_index < x.num_rhs;
         rhs_index += num_subwarp_grps_per_block) {
        one_dot(x, y, rhs_index, result, subwarp_grp);
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
        compute_dot_product(x_b, y_b, r_b);
    }
}


template <typename Op, typename ValueType>
__device__ __forceinline__ void transpose_entry(const int src_nrows,
                                                const int src_ncols,
                                                const size_type src_stride,
                                                const ValueType* const src,
                                                const size_type dest_stride,
                                                ValueType* const dest, Op op)
{
    constexpr auto tile_size = config::warp_size;
    auto thread_block = group::this_thread_block();
    auto warp_grp = group::tiled_partition<tile_size>(thread_block);
    const auto warp_grp_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_tiles = ceildiv(blockDim.x, tile_size);

    for (int i_row = warp_grp_id; i_row < src_nrows; i_row += num_tiles) {
        for (int j = warp_grp.thread_rank(); j < src_ncols; j += tile_size) {
            dest[j * dest_stride + i_row] = op(src[i_row * src_stride + j]);
        }
    }
}

template <typename Op, typename ValueType>
__global__ void transpose(const int src_nrows, const int src_ncols,
                          const size_type src_stride,
                          const ValueType* const src,
                          const size_type dest_stride, ValueType* const dest,
                          Op op)
{
    const ValueType* const bsrc = src + blockIdx.x * src_stride * src_nrows;
    ValueType* const bdest = dest + blockIdx.x * dest_stride * src_ncols;
    transpose_entry(src_nrows, src_ncols, src_stride, bsrc, dest_stride, bdest,
                    op);
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void add_scaled_identity(
    const size_type num_batch, const int nrows, const int ncols,
    const size_type stride, ValueType* const __restrict__ values,
    const size_type a_stride, const ValueType* const __restrict__ alpha,
    const size_type b_stride, const ValueType* const __restrict__ beta)
{
    constexpr auto tile_size = config::warp_size;
    auto warp_grp =
        group::tiled_partition<tile_size>(group::this_thread_block());
    const auto warp_grp_id = static_cast<int>(threadIdx.x / tile_size);
    constexpr int num_tiles = default_block_size / tile_size;
    for (size_type ibatch = blockIdx.x; ibatch < num_batch;
         ibatch += gridDim.x) {
        auto values_batch = values + ibatch * stride * nrows;
        for (int row = warp_grp_id; row < nrows; row += num_tiles) {
            for (int col = warp_grp.thread_rank(); col < ncols;
                 col += tile_size) {
                values_batch[row * stride + col] *= beta[ibatch * b_stride];
                if (col == row) {
                    values_batch[row * stride + row] +=
                        alpha[ibatch * a_stride];
                }
            }
        }
    }
}
