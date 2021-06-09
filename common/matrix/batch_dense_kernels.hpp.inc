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
using BatchEntry = gko::batch_dense::BatchEntry<ValueType>;


/**
 *
 *   Scales the vectors in global or shared memory with a factor of alpha(alpha
 * is in global memory or shared memory or even the thread's local memory)
 *
 */
template <typename ValueType>
__device__ __forceinline__ void scale(const BatchEntry<const ValueType> &alpha,
                                      const BatchEntry<ValueType> &x)
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
    const gko::batch_dense::BatchEntry<const ValueType> &alpha,
    const gko::batch_dense::BatchEntry<const ValueType> &x,
    const gko::batch_dense::BatchEntry<ValueType> &y)
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
    using real_type = remove_complex<ValueType>;
    for (int col = 0; col < x.num_rhs; col++) {
        __shared__ UninitializedArray<real_type, default_block_size> work;
        work[threadIdx.x] = zero<real_type>();
        // first sum across block-length chunks
        for (int i = threadIdx.x; i < x.num_rows; i += blockDim.x) {
            work[i] += squared_norm(x.values[i * x.stride + col]);
        }
        __syncthreads();
        reduce(group::this_thread_block(), static_cast<real_type *>(work),
               [](const real_type &a, const real_type &b) { return a + b; });
        if (threadIdx.x == 0) {
            result.values[col] = sqrt(work[0]);
        }
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

/**
 * Copies the values of one multi-vector into another.
 *
 * Note that the output multi-vector should already have memory allocated
 * and stride set.
 */
template <typename ValueType>
__device__ __forceinline__ void copy(
    const gko::batch_dense::BatchEntry<const ValueType> &in,
    const gko::batch_dense::BatchEntry<ValueType> &out)
{
    for (int iz = threadIdx.x; iz < in.num_rows * in.num_rhs;
         iz += blockDim.x) {
        const int i = iz / in.num_rhs;
        const int j = iz % in.num_rhs;
        out.values[i * out.stride + j] = in.values[i * in.stride + j];
    }
}


/**
 * Computes the dot product of some column vectors in global or shared memory.
 *
 * @param result  Holds dot product value for vector in x and y.
 */
template <typename ValueType>
__device__ __forceinline__ void compute_dot_product(
    const BatchEntry<const ValueType> &x, const BatchEntry<const ValueType> &y,
    const BatchEntry<ValueType> &result)
{
    for (int col = 0; col < x.num_rhs; col++) {
        __shared__ UninitializedArray<ValueType, default_block_size> work;
        work[threadIdx.x] = zero<ValueType>();
        // first sum across block-length chunks
        for (int i = threadIdx.x; i < x.num_rows; i += blockDim.x) {
            work[i] += conj(x.values[i * x.stride + col]) *
                       y.values[i * y.stride + col];
        }
        __syncthreads();
        // next, reduce in the block
        reduce(group::this_thread_block(), static_cast<ValueType *>(work),
               [](const ValueType &a, const ValueType &b) { return a + b; });
        if (threadIdx.x == 0) {
            result.values[col] = work[0];
        }
    }
}

template <typename ValueType>
__global__ __launch_bounds__(
    default_block_size,
    sm_multiplier) void compute_dot_product(const gko::batch_dense::
                                                UniformBatch<const ValueType>
                                                    x,
                                            const gko::batch_dense::
                                                UniformBatch<const ValueType>
                                                    y,
                                            const gko::batch_dense::
                                                UniformBatch<ValueType>
                                                    result)
{
    for (size_type ibatch = blockIdx.x; ibatch < x.num_batch;
         ibatch += gridDim.x) {
        const auto x_b = gko::batch::batch_entry(x, ibatch);
        const auto y_b = gko::batch::batch_entry(y, ibatch);
        const auto r_b = gko::batch::batch_entry(result, ibatch);
        compute_dot_product(x_b, y_b, r_b);
    }
}


template <typename ValueType>
__device__ __forceinline__ void compute_norm2(
    const gko::batch_dense::BatchEntry<const ValueType> &x,
    const gko::batch_dense::BatchEntry<remove_complex<ValueType>> &result,
    const uint32 &converged)
{
    for (int c = threadIdx.x; c < x.num_rhs; c += blockDim.x) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        result.values[c] = zero<remove_complex<ValueType>>();

        for (int r = 0; r < x.num_rows; r++) {
            result.values[c] += squared_norm(x.values[r * x.stride + c]);
        }

        result.values[c] = sqrt(result.values[c]);
    }
    // TODO: put a parallel reduce here
}


template <typename ValueType>
__device__ __forceinline__ void add_scaled(
    const gko::batch_dense::BatchEntry<const ValueType> &alpha,
    const gko::batch_dense::BatchEntry<const ValueType> &x,
    const gko::batch_dense::BatchEntry<ValueType> &y, const uint32 &converged)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = threadIdx.x; li < max_li; li += blockDim.x) {
        const int row = li / x.num_rhs;
        const int col = li % x.num_rhs;

        const uint32 conv = converged & (1 << col);

        if (conv) {
            continue;
        }

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
__device__ __forceinline__ void scale(const BatchEntry<const ValueType> &alpha,
                                      const BatchEntry<ValueType> &x,
                                      const uint32 &converged)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = threadIdx.x; li < max_li; li += blockDim.x) {
        const int row = li / x.num_rhs;
        const int col = li % x.num_rhs;

        const uint32 conv = converged & (1 << col);

        if (conv) {
            continue;
        }

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
__device__ __forceinline__ void copy(
    const gko::batch_dense::BatchEntry<const ValueType> &src_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &dest_shared_entry,
    const uint32 &converged)
{
    for (int li = threadIdx.x;
         li < dest_shared_entry.num_rhs * dest_shared_entry.num_rows;
         li += blockDim.x) {
        const int r = li / dest_shared_entry.num_rhs;
        const int c = li % dest_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        dest_shared_entry.values[r * dest_shared_entry.stride + c] =
            src_shared_entry.values[r * src_shared_entry.stride + c];
    }
}


template <typename ValueType>
__device__ __forceinline__ void compute_dot_product(
    const BatchEntry<const ValueType> &x, const BatchEntry<const ValueType> &y,
    const BatchEntry<ValueType> &result, const uint32 &converged)
{
    for (int c = threadIdx.x; c < x.num_rhs; c += blockDim.x) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        result.values[c] = zero<ValueType>();

        for (int r = 0; r < x.num_rows; r++) {
            result.values[c] +=
                conj(x.values[r * x.stride + c]) * y.values[r * y.stride + c];
        }
    }
    // TODO: put a parallel reduce here
}
