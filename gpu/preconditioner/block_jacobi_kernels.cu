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
#include "core/synthesizer/implementation_selection.hpp"
#include "gpu/base/math.hpp"
#include "gpu/components/reduction.cuh"
#include "gpu/components/uninitialized_array.hpp"


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
namespace warp {


/**
 * @internal
 *
 * Applies a Gauss-Jordan transformation (single step of Gauss-Jordan
 * elimination) to a `max_problem_size`-by-`max_problem_size` matrix using
 * `subwarp_size` threads (restrictions from `gpu::warp::reduce` apply, and
 * `max_problem_size` must not be greater than `subwarp_size`.
 * Each thread contributes one `row` of the matrix, and the routine uses warp
 * shuffles to exchange data between rows. The transform is performed by using
 * the `key_row`-th row and `key_col`-th column of the matrix.
 */
template <int max_problem_size, int subwarp_size, typename ValueType>
__device__ __forceinline__ void apply_gauss_jordan_transform(int key_row,
                                                             int key_col,
                                                             ValueType *row)
{
    auto key_col_elem = gpu::warp::shuffle(row[key_col], key_row, subwarp_size);
    if (key_col_elem == zero<ValueType>()) {
        // TODO: implement error handling for GPUs to be able to properly
        //       report it here
        return;
    }
    if (threadIdx.x == key_row) {
        key_col_elem = one<ValueType>() / key_col_elem;
    } else {
        key_col_elem = -row[key_col] / key_col_elem;
    }
#pragma unroll
    for (int i = 0; i < max_problem_size; ++i) {
        const auto key_row_elem =
            gpu::warp::shuffle(row[i], key_row, subwarp_size);
        if (threadIdx.x == key_row) {
            row[i] = zero<ValueType>();
        }
        row[i] += key_col_elem * key_row_elem;
    }
    row[key_col] = key_col_elem;
}


/**
 * @internal
 *
 * Inverts a matrix using Gauss-Jordan elimination. The inversion is
 * done in-place, so the original matrix will be overridden with the inverse.
 * The inversion routine uses implicit pivoting, so the returned matrix will be
 * a permuted inverse (from both sides). To obtain the correct inverse, the
 * rows of the result should be permuted with \f$P\f$, and the columns with
 * \f$ P^T \f$ (i.e.
 * \f$ A^{-1} = P X P \f$, where \f$ X \f$ is the returned matrix). These
 * permutation matrices are returned compressed as vectors `perm` and `tperm`,
 * respectively. `i`-th value of each of the vectors is returned to sub-warp
 * thread with index `i`.
 *
 * @tparam max_problem_size  the maximum problem size that will be passed to the
 *                           inversion routine (a tighter bound results in
 *                           faster code
 * @tparam subwarp_size  the size of the sub-warp used to invert the block,
 *                       cannot be smaller than `max_problem_size`
 * @tparam ValueType  type of values stored in the matrix
 *
 * @param problem_size  the actual size of the matrix (cannot be larger than
 *                      max_problem_size)
 * @param row  a pointer to the matrix row (i-th thread in the subwarp should
 *             pass the pointer to the i-th row), has to have at least
 *             max_problem_size elements
 * @param perm  a value to hold an element of permutation matrix \f$ P \f$
 * @param tperm  a value to hold an element of permutation matrix \f$ P^T \f$
 */
template <int max_problem_size, int subwarp_size, typename ValueType>
__device__ __forceinline__ void invert_block(int problem_size, ValueType *row,
                                             int &perm, int &tperm)
{
    static_assert(max_problem_size <= subwarp_size,
                  "max_problem_size cannot be larger than subwarp_size");
    // prevent rows after problem_size to become pivots
    auto pivoted = threadIdx.x >= problem_size;
#pragma unroll
    for (int i = 0; i < max_problem_size; ++i) {
        if (i >= problem_size) {
            break;
        }
        const auto piv = gpu::warp::choose_pivot<subwarp_size>(row[i], pivoted);
        if (threadIdx.x == piv) {
            perm = i;
            pivoted = true;
        }
        if (threadIdx.x == i) {
            tperm = piv;
        }
        apply_gauss_jordan_transform<max_problem_size, subwarp_size>(piv, i,
                                                                     row);
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
    const int bpw = warp_size / ws;
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
            for (auto j = rstart; j < rend; j += warp_size) {
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
    const int bpw = warp_size / ws;
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
__global__ void __launch_bounds__(warps_per_block *warp_size)
    generate(size_type num_rows, const IndexType *__restrict__ row_ptrs,
             const IndexType *__restrict__ col_idxs,
             const ValueType *__restrict__ values,
             ValueType *__restrict__ block_data, size_type padding,
             const IndexType *__restrict__ block_ptrs, size_type num_blocks)
{
    const int blocks_per_warp = warp_size / subwarp_size;
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
        warp::invert_block<max_block_size, subwarp_size>(block_size, row, perm,
                                                         iperm);
    }
    insert_diag_blocks_trans<max_block_size, subwarp_size, warps_per_block>(
        perm, iperm, row, 1, block_ptrs, block_data, padding, num_blocks);
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void __launch_bounds__(warps_per_block *warp_size)
    apply(const ValueType *__restrict__ block_data, int32 padding,
          const IndexType *__restrict__ block_ptrs, size_type num_blocks,
          const ValueType *__restrict__ b, int32 b_padding,
          ValueType *__restrict__ x, int32 x_padding)
{
    const int bpw = warp_size / subwarp_size;
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
        auto out = gpu::warp::reduce<subwarp_size>(
            a * v, [](ValueType x, ValueType y) { return x + y; });
        if (threadIdx.x == 0) {
            x[(bstart + i) * x_padding] = out;
        }
        rstart += padding;
    }
}


}  // namespace device


constexpr int get_larger_power(int value, int guess = 1)
{
    return guess >= value ? guess : get_larger_power(value, guess << 1);
}


template <int warps_per_block, int max_block_size, typename ValueType,
          typename IndexType>
void generate(syn::compile_int_list<max_block_size>,
              const matrix::Csr<ValueType, IndexType> *mtx,
              ValueType *block_data, size_type padding,
              const IndexType *block_ptrs, size_type num_blocks)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    constexpr int blocks_per_warp = warp_size / subwarp_size;
    const dim3 grid_size(ceildiv(num_blocks, warps_per_block * blocks_per_warp),
                         1, 1);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    device::generate<max_block_size, subwarp_size, warps_per_block>
        <<<grid_size, block_size, 0, 0>>>(
            mtx->get_num_rows(), mtx->get_const_row_ptrs(),
            mtx->get_const_col_idxs(), as_cuda_type(mtx->get_const_values()),
            as_cuda_type(block_data), padding, block_ptrs, num_blocks);
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_generate, generate);


template <int warps_per_block, int max_block_size, typename ValueType,
          typename IndexType>
void apply(syn::compile_int_list<max_block_size>, size_type num_blocks,
           const IndexType *block_pointers, const ValueType *blocks,
           size_type block_padding, const ValueType *b, size_type b_padding,
           ValueType *x, size_type x_padding)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    const int blocks_per_warp = warp_size / subwarp_size;
    const dim3 grid_size(ceildiv(num_blocks, warps_per_block * blocks_per_warp),
                         1, 1);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    device::apply<max_block_size, subwarp_size, warps_per_block>
        <<<grid_size, block_size, 0, 0>>>(
            as_cuda_type(blocks), block_padding, block_pointers, num_blocks,
            as_cuda_type(b), b_padding, as_cuda_type(x), x_padding);
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_apply, apply);


}  // namespace


using compiled_kernels = syn::compile_int_list<1, 3, 16, 32>;


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const GpuExecutor> exec,
              const matrix::Csr<ValueType, IndexType> *system_matrix,
              size_type num_blocks, uint32 max_block_size, size_type padding,
              const Array<IndexType> &block_pointers, Array<ValueType> &blocks)
{
    const int warps_per_block = 4;
    select_generate(compiled_kernels(),
                    [&](int compiled_block_size) {
                        return max_block_size <= compiled_block_size;
                    },
                    syn::compile_int_list<warps_per_block>(),
                    syn::compile_type_list<>(), system_matrix,
                    blocks.get_data(), padding, block_pointers.get_const_data(),
                    num_blocks);
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
                 blocks, b, tmp.get());
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
    // TODO: write a special kernel for multiple RHS
    for (size_type col = 0; col < b->get_num_cols(); ++col) {
        select_apply(compiled_kernels(),
                     [&](int compiled_block_size) {
                         return max_block_size <= compiled_block_size;
                     },
                     syn::compile_int_list<warps_per_block>(),
                     syn::compile_type_list<>(), num_blocks,
                     block_pointers.get_const_data(), blocks.get_const_data(),
                     padding, b->get_const_values() + col, b->get_padding(),
                     x->get_values() + col, x->get_padding());
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
