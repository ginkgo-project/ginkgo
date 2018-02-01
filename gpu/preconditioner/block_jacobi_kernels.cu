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

#include "core/preconditioner/block_jacobi_kernels.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "gpu/base/math.hpp"
#include "gpu/base/shuffle.cuh"
#include "gpu/base/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace gpu {
namespace block_jacobi {


template <typename ValueType, typename IndexType>
void find_blocks(std::shared_ptr<const GpuExecutor> exec,
                 const matrix::Csr<ValueType, IndexType> *system_matrix,
                 uint32 max_block_size, size_type &num_blocks,
                 Array<IndexType> &block_pointers) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_FIND_BLOCKS_KERNEL);


namespace {


constexpr int cuda_warp_size = 32;


uint64 ceildiv(uint64 num, uint64 denom) { return (num - 1) / denom + 1; }


namespace warp {


// Computes a reduction using the binary operation `reduce_op` on a block of
// block_size threads (block_size <= MagmaWarpSize).
// Each thread contributes with one element `local_data`. The local thread
// element is always passed as the first parameter to the `reduce_op`.
// The function returns the result of the reduction on all threads.
//
// NOTE: The function is guarantied to return the correct value on all threads
// only if `reduce_op` is comutative (in addition to being asociative).
// Otherwise, the correct value is returned only to the first thread (use
// gpu::warp::shuffle to exchange it with the other threads).
template <int ws, typename ValueType, typename Operator>
__device__ __forceinline__ ValueType reduce(ValueType local_data,
                                            Operator reduce_op)
{
#pragma unroll
    for (int bitmask = 1; bitmask < ws; bitmask <<= 1) {
        const auto remote_data =
            gpu::warp::shuffle_xor(local_data, bitmask, ws);
        local_data = reduce_op(local_data, remote_data);
    }
    return local_data;
}


// Returns the index of the thread that has the element with the largest
// magnitude among all the threads in the block of block_size threads
// (block_size <= MagmaWarpSize) which called this function with is_pivoted set
// to false.
template <int ws, typename ValueType>
__device__ __forceinline__ int choose_pivot(ValueType local_data,
                                            bool is_pivoted)
{
    using real = remove_complex<ValueType>;
    real lmag = is_pivoted ? -one<real>() : abs(local_data);
    const auto pivot = reduce<ws>(threadIdx.x, [&](int lidx, int ridx) {
        const auto rmag = gpu::warp::shuffle(lmag, ridx, ws);
        if (rmag > lmag) {
            lmag = rmag;
            lidx = ridx;
        }
        return lidx;
    });
    // make sure everyone has the same pivot, as the above reduction operator is
    // not comutative
    return gpu::warp::shuffle(pivot, 0, ws);
}


// Applies a Gauss-Jordan elimination to a block_size-by-block_size matrix,
// (block_size <= MagmaWarpSize) using the element at position (key_r, key_c).
// Each of the block_size threads in the block supplies a single row of the
// matrix as row_reg argument when calling this function.
template <int mps, int ws, typename ValueType>
__device__ __forceinline__ void apply_gauss_jordan_step(int key_r, int key_c,
                                                        ValueType *row)
{
    auto key_col = gpu::warp::shuffle(row[key_c], key_r, ws);
    if (key_col == zero<ValueType>()) {
        return;  // TODO(Goran): report error here!
    }
    if (threadIdx.x == key_r) {
        key_col = one<ValueType>() / key_col;
    } else {
        key_col = -row[key_c] / key_col;
    }
#pragma unroll
    for (int i = 0; i < mps; ++i) {
        const auto key_row_elem = gpu::warp::shuffle(row[i], key_r, ws);
        if (threadIdx.x == key_r) {
            row[i] = zero<ValueType>();
        }
        row[i] += key_col * key_row_elem;
    }
    row[key_c] = key_col;
}


template <int mps, int ws, typename ValueType>
__device__ __forceinline__ void invert_using_gauss_jordan(ValueType *row,
                                                          int *perm, int *iperm,
                                                          int size)
{
    // prevent rows after real size to become pivots.
    auto pivoted = threadIdx.x >= size;
#pragma unroll
    for (int i = 0; i < mps; ++i) {
        if (i >= size) {
            break;
        }
        const auto piv = choose_pivot<ws>(row[i], pivoted);
        if (threadIdx.x == piv) {
            // I'm selected at step i, so my result needs to go to output row i.
            *perm = i;
            pivoted = true;
        }
        if (threadIdx.x == i) {
            *iperm = piv;
        }
        apply_gauss_jordan_step<mps, ws>(piv, i, row);
    }
}


}  // namespace warp


namespace device {


template <int mbs, int ws, int wpb, typename ValueType, typename IndexType>
__device__ __forceinline__ void extract_transposed_diag_blocks(
    size_type num_rows, const IndexType *__restrict__ Arow,
    const IndexType *__restrict__ Acol, const ValueType *__restrict__ Aval,
    const IndexType *__restrict__ block_ptrs, size_type num_blocks,
    ValueType *__restrict__ B, int binc, ValueType *sB)
{
    const int bpw = cuda_warp_size / ws;
    const int tid = threadIdx.y * ws + threadIdx.x;
    IndexType bid = blockIdx.x * wpb * bpw + threadIdx.z * bpw;
    auto bstart = block_ptrs[bid];
    IndexType bsize = 0;
#pragma unroll
    for (int b = 0; b < bpw; ++b, ++bid) {
        if (bid >= num_blocks) {
            break;
        }
        bstart += bsize;
        bsize = block_ptrs[bid + 1] - bstart;
#pragma unroll
        for (int i = 0; i < mbs; ++i) {
            if (i >= bsize) {
                break;
            }
            if (threadIdx.y == b && threadIdx.x < mbs) {
                sB[threadIdx.x] = zero<ValueType>();
            }
            const auto row = bstart + i;
            const auto rstart = Arow[row] + tid;
            const auto rend = Arow[row + 1];
            for (auto j = rstart; j < rend; j += cuda_warp_size) {
                const auto col = Acol[j] - bstart;
                if (col >= bsize) {
                    break;
                }
                if (col >= 0) {
                    sB[col] = Aval[j];
                }
            }
            __threadfence_block();
            if (threadIdx.y == b && threadIdx.x < bsize) {
                B[i * binc] = sB[threadIdx.x];
            }
        }
    }
}


template <int mbs, int ws, int wpb, typename ValueType, typename IndexType>
__device__ __forceinline__ void insert_diag_blocks_trans(
    int rperm, int cperm, const ValueType *__restrict__ B, int binc,
    const IndexType *__restrict__ block_ptrs,
    ValueType *__restrict__ block_data, size_type padding, size_type num_blocks)
{
    const int bpw = cuda_warp_size / ws;
    const size_type bid =
        blockIdx.x * wpb * bpw + threadIdx.z * bpw + threadIdx.y;
    const IndexType bstart = bid < num_blocks ? block_ptrs[bid] : 0;
    const IndexType bsize = bid < num_blocks ? block_ptrs[bid + 1] - bstart : 0;
#pragma unroll
    for (int i = 0; i < mbs; ++i) {
        if (i >= bsize) {
            break;
        }
        const auto idx = gpu::warp::shuffle(cperm, i, ws);
        const auto rstart = (bstart + idx) * padding;
        if (bid < num_blocks && threadIdx.x < bsize) {
            block_data[rstart + rperm] = B[i * binc];
        }
    }
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void __launch_bounds__(warps_per_block *cuda_warp_size)
    generate(size_type num_rows, const IndexType *__restrict__ row_ptrs,
             const IndexType *__restrict__ col_idxs,
             const ValueType *__restrict__ values,
             ValueType *__restrict__ block_data, size_type padding,
             const IndexType *__restrict__ block_ptrs, size_type num_blocks)
{
    const int blocks_per_warp = cuda_warp_size / subwarp_size;
    const size_type bid = blockIdx.x * warps_per_block * blocks_per_warp +
                          threadIdx.z * blocks_per_warp + threadIdx.y;
    const int block_size =
        (bid < num_blocks) ? block_ptrs[bid + 1] - block_ptrs[bid] : 0;
    int perm = threadIdx.x;
    int iperm = threadIdx.x;
    ValueType row[max_block_size];
    __shared__ UninitializedArray<ValueType, max_block_size * warps_per_block>
        sM;

    extract_transposed_diag_blocks<max_block_size, subwarp_size,
                                   warps_per_block>(
        num_rows, row_ptrs, col_idxs, values, block_ptrs, num_blocks, row, 1,
        sM + threadIdx.z * max_block_size);
    if (bid < num_blocks) {
        warp::invert_using_gauss_jordan<max_block_size, subwarp_size>(
            row, &perm, &iperm, block_size);
    }
    insert_diag_blocks_trans<max_block_size, subwarp_size, warps_per_block>(
        perm, iperm, row, 1, block_ptrs, block_data, padding, num_blocks);
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void __launch_bounds__(warps_per_block *cuda_warp_size)
    apply(const ValueType *__restrict__ block_data, int32 padding,
          const IndexType *__restrict__ block_ptrs, size_type num_blocks,
          const ValueType *__restrict__ b, int32 b_padding,
          ValueType *__restrict__ x, int32 x_padding)
{
    const int bpw = cuda_warp_size / subwarp_size;
    const IndexType bid =
        blockIdx.x * warps_per_block * bpw + threadIdx.z * bpw + threadIdx.y;
    if (bid >= num_blocks) {
        return;
    }
    const auto bstart = block_ptrs[bid];
    const auto bsize = block_ptrs[bid + 1] - bstart;
    auto rstart = bstart * padding;
    ValueType v = zero<ValueType>();
    if (threadIdx.x < bsize) {
        v = b[(bstart + threadIdx.x) * b_padding];
    }
    auto a = zero<ValueType>();
    for (int i = 0; i < bsize; ++i) {
        if (threadIdx.x < bsize) {
            a = block_data[rstart + threadIdx.x];
        }
        auto out = warp::reduce<subwarp_size>(
            a * v, [](ValueType x, ValueType y) { return x + y; });
        if (threadIdx.x == 0) {
            x[(bstart + i) * x_padding] = out;
        }
        rstart += padding;
    }
}


}  // namespace device


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
void launch_generate_kernel(const matrix::Csr<ValueType, IndexType> *mtx,
                            ValueType *block_data, size_type padding,
                            const IndexType *block_ptrs, size_type num_blocks)
{
    const int blocks_per_warp = cuda_warp_size / subwarp_size;
    const dim3 grid_size(ceildiv(num_blocks, warps_per_block * blocks_per_warp),
                         1, 1);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    device::generate<max_block_size, subwarp_size, warps_per_block>
        <<<grid_size, block_size, 0, 0>>>(
            mtx->get_num_rows(), mtx->get_const_row_ptrs(),
            mtx->get_const_col_idxs(), as_cuda_type(mtx->get_const_values()),
            as_cuda_type(block_data), padding, block_ptrs, num_blocks);
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
void launch_apply_kernel(size_type num_blocks, const IndexType *block_pointers,
                         const ValueType *blocks, size_type block_padding,
                         const ValueType *b, size_type b_padding, ValueType *x,
                         size_type x_padding)
{
    const int blocks_per_warp = cuda_warp_size / subwarp_size;
    const dim3 grid_size(ceildiv(num_blocks, warps_per_block * blocks_per_warp),
                         1, 1);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    device::apply<max_block_size, subwarp_size, warps_per_block>
        <<<grid_size, block_size, 0, 0>>>(
            as_cuda_type(blocks), block_padding, block_pointers, num_blocks,
            as_cuda_type(b), b_padding, as_cuda_type(x), x_padding);
}


template <int max_block_size, int subwarp_size = max_block_size>
struct select_and_launch {
    template <int warps_per_block, typename... Args>
    static void generate(int runtime_max_block_size, Args &&... args)
    {
        if (runtime_max_block_size == max_block_size) {
            launch_generate_kernel<max_block_size, subwarp_size,
                                   warps_per_block>(
                std::forward<Args>(args)...);
        } else {
            constexpr int new_subwarp_size =
                (max_block_size - 1 == subwarp_size / 2) ? subwarp_size / 2
                                                         : subwarp_size;
            return select_and_launch<max_block_size - 1, new_subwarp_size>::
                template generate<warps_per_block>(runtime_max_block_size,
                                                   std::forward<Args>(args)...);
        }
    }

    template <int warps_per_block, typename... Args>
    static void apply(int runtime_max_block_size, Args &&... args)
    {
        if (runtime_max_block_size == max_block_size) {
            launch_apply_kernel<max_block_size, subwarp_size, warps_per_block>(
                std::forward<Args>(args)...);
        } else {
            constexpr int new_subwarp_size =
                (max_block_size - 1 == subwarp_size / 2) ? subwarp_size / 2
                                                         : subwarp_size;
            return select_and_launch<max_block_size - 1, new_subwarp_size>::
                template apply<warps_per_block>(runtime_max_block_size,
                                                std::forward<Args>(args)...);
        }
    }
};


template <>
struct select_and_launch<0, 0> {
    template <int warps_per_block, typename... Args>
    static void generate(Args &&...)
    {
        throw "TODO";
    }

    template <int warps_per_block, typename... Args>
    static void apply(Args &&...)
    {
        throw "TODO";
    }
};


}  // namespace


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const GpuExecutor> exec,
              const matrix::Csr<ValueType, IndexType> *system_matrix,
              size_type num_blocks, uint32 max_block_size, size_type padding,
              const Array<IndexType> &block_pointers, Array<ValueType> &blocks)
{
    const int warps_per_block = 4;
    const int block_size_limit = cuda_warp_size;
    select_and_launch<block_size_limit>::generate<warps_per_block>(
        max_block_size, system_matrix, blocks.get_data(), padding,
        block_pointers.get_const_data(), num_blocks);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_GENERATE_KERNEL);


template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const GpuExecutor> exec, size_type num_blocks,
           uint32 max_block_size, size_type padding,
           const Array<IndexType> &block_pointers,
           const Array<ValueType> &blocks,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *x)
{
    // TODO: do this efficiently
    auto tmp = matrix::Dense<ValueType>::create_with_config_of(x);
    simple_apply(exec, num_blocks, max_block_size, padding, block_pointers,
                 blocks, b, static_cast<matrix::Dense<ValueType> *>(tmp.get()));
    x->scale(beta);
    x->add_scaled(alpha, tmp.get());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void simple_apply(std::shared_ptr<const GpuExecutor> exec, size_type num_blocks,
                  uint32 max_block_size, size_type padding,
                  const Array<IndexType> &block_pointers,
                  const Array<ValueType> &blocks,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *x)
{
    const int warps_per_block = 4;
    const int block_size_limit = cuda_warp_size;
    // TODO write a special kernels for multiple RHS
    for (size_type col = 0; col < b->get_num_cols(); ++col) {
        select_and_launch<block_size_limit>::apply<warps_per_block>(
            max_block_size, num_blocks, block_pointers.get_const_data(),
            blocks.get_const_data(), padding, b->get_const_values() + col,
            b->get_padding(), x->get_values() + col, x->get_padding());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_SIMPLE_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const GpuExecutor> exec,
                      size_type num_blocks,
                      const Array<IndexType> &block_pointers,
                      const Array<ValueType> &blocks, size_type block_padding,
                      ValueType *result_values,
                      size_type result_padding) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_CONVERT_TO_DENSE_KERNEL);


}  // namespace block_jacobi
}  // namespace gpu
}  // namespace kernels
}  // namespace gko
