/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include "core/preconditioner/jacobi_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/base/extended_float.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/diagonal_block_manipulation.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"
#include "cuda/components/warp_blas.cuh"


namespace gko {
namespace kernels {
namespace cuda {


/**
 * A compile-time list of block sizes for which dedicated generate and apply
 * kernels should be compiled.
 */
using compiled_kernels = syn::value_list<int, 1, 13, 16, 32>;


namespace {


template <int max_block_size, typename ReducedType, typename Group,
          typename ValueType, typename IndexType>
__device__ __forceinline__ bool validate_precision_reduction_feasibility(
    Group &__restrict__ group, IndexType block_size,
    ValueType *__restrict__ row, ValueType *__restrict__ work, size_type stride)
{
    using gko::detail::float_traits;
    // save original data and reduce precision
    if (group.thread_rank() < block_size) {
#pragma unroll
        for (auto i = 0u; i < max_block_size; ++i) {
            if (i >= block_size) {
                break;
            }
            work[i * stride + group.thread_rank()] = row[i];
            row[i] = static_cast<ValueType>(static_cast<ReducedType>(row[i]));
        }
    }

    // compute the condition number
    auto perm = group.thread_rank();
    auto trans_perm = perm;
    auto block_cond = compute_infinity_norm<max_block_size>(group, block_size,
                                                            block_size, row);
    auto succeeded =
        invert_block<max_block_size>(group, block_size, row, perm, trans_perm);
    block_cond *= compute_infinity_norm<max_block_size>(group, block_size,
                                                        block_size, row);

    // restore original data
    if (group.thread_rank() < block_size) {
#pragma unroll
        for (auto i = 0u; i < max_block_size; ++i) {
            if (i >= block_size) {
                break;
            }
            row[i] = work[i * stride + group.thread_rank()];
        }
    }

    return succeeded && block_cond >= 1.0 &&
           block_cond * float_traits<remove_complex<ValueType>>::eps < 1e-3;
}


}  // namespace


namespace kernel {
namespace {


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void __launch_bounds__(warps_per_block *cuda_config::warp_size)
    generate(size_type num_rows, const IndexType *__restrict__ row_ptrs,
             const IndexType *__restrict__ col_idxs,
             const ValueType *__restrict__ values,
             ValueType *__restrict__ block_data,
             preconditioner::block_interleaved_storage_scheme<IndexType>
                 storage_scheme,
             const IndexType *__restrict__ block_ptrs, size_type num_blocks)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto block = group::this_thread_block();
    ValueType row[max_block_size];
    __shared__ UninitializedArray<ValueType, max_block_size * warps_per_block>
        workspace;
    csr::extract_transposed_diag_blocks<max_block_size, warps_per_block>(
        block, cuda_config::warp_size / subwarp_size, row_ptrs, col_idxs,
        values, block_ptrs, num_blocks, row, 1,
        workspace + threadIdx.z * max_block_size);
    const auto subwarp = group::tiled_partition<subwarp_size>(block);
    if (block_id < num_blocks) {
        const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];
        auto perm = subwarp.thread_rank();
        auto trans_perm = subwarp.thread_rank();
        invert_block<max_block_size>(subwarp, block_size, row, perm,
                                     trans_perm);
        copy_matrix<max_block_size, and_transpose>(
            subwarp, block_size, row, 1, perm, trans_perm,
            block_data + storage_scheme.get_global_block_offset(block_id),
            storage_scheme.get_stride());
    }
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void
__launch_bounds__(warps_per_block *cuda_config::warp_size) adaptive_generate(
    size_type num_rows, const IndexType *__restrict__ row_ptrs,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values, remove_complex<ValueType> accuracy,
    ValueType *__restrict__ block_data,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    remove_complex<ValueType> *__restrict__ conditioning,
    precision_reduction *__restrict__ block_precisions,
    const IndexType *__restrict__ block_ptrs, size_type num_blocks)
{
    // extract blocks
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto block = group::this_thread_block();
    ValueType row[max_block_size];
    __shared__ UninitializedArray<ValueType, max_block_size * warps_per_block>
        workspace;
    csr::extract_transposed_diag_blocks<max_block_size, warps_per_block>(
        block, cuda_config::warp_size / subwarp_size, row_ptrs, col_idxs,
        values, block_ptrs, num_blocks, row, 1,
        workspace + threadIdx.z * max_block_size);

    // compute inverse and figure out the correct precision
    const auto subwarp = group::tiled_partition<subwarp_size>(block);
    const auto block_size =
        block_id < num_blocks ? block_ptrs[block_id + 1] - block_ptrs[block_id]
                              : 0;
    auto perm = subwarp.thread_rank();
    auto trans_perm = subwarp.thread_rank();
    auto prec_descriptor = ~uint32{};
    if (block_id < num_blocks) {
        auto block_cond = compute_infinity_norm<max_block_size>(
            subwarp, block_size, block_size, row);
        invert_block<max_block_size>(subwarp, block_size, row, perm,
                                     trans_perm);
        block_cond *= compute_infinity_norm<max_block_size>(subwarp, block_size,
                                                            block_size, row);
        conditioning[block_id] = block_cond;
        const auto prec = block_precisions[block_id];
        prec_descriptor =
            preconditioner::detail::precision_reduction_descriptor::singleton(
                prec);
        if (prec == precision_reduction::autodetect()) {
            using preconditioner::detail::get_supported_storage_reductions;
            prec_descriptor = get_supported_storage_reductions<ValueType>(
                accuracy, block_cond,
                [&subwarp, &block_size, &row, &block_data, &storage_scheme,
                 &block_id] {
                    using target = reduce_precision<ValueType>;
                    return validate_precision_reduction_feasibility<
                        max_block_size, target>(
                        subwarp, block_size, row,
                        block_data +
                            storage_scheme.get_global_block_offset(block_id),
                        storage_scheme.get_stride());
                },
                [&subwarp, &block_size, &row, &block_data, &storage_scheme,
                 &block_id] {
                    using target =
                        reduce_precision<reduce_precision<ValueType>>;
                    return validate_precision_reduction_feasibility<
                        max_block_size, target>(
                        subwarp, block_size, row,
                        block_data +
                            storage_scheme.get_global_block_offset(block_id),
                        storage_scheme.get_stride());
                });
        }
    }

    // make sure all blocks in the group have the same precision
    const auto warp = group::tiled_partition<cuda_config::warp_size>(block);
    const auto prec =
        preconditioner::detail::get_optimal_storage_reduction(reduce(
            warp, prec_descriptor, [](uint32 x, uint32 y) { return x & y; }));

    // store the block back into memory
    if (block_id < num_blocks) {
        block_precisions[block_id] = prec;
        GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(
            ValueType, prec,
            copy_matrix<max_block_size, and_transpose>(
                subwarp, block_size, row, 1, perm, trans_perm,
                reinterpret_cast<resolved_precision *>(
                    block_data + storage_scheme.get_group_offset(block_id)) +
                    storage_scheme.get_block_offset(block_id),
                storage_scheme.get_stride()));
    }
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void __launch_bounds__(warps_per_block *cuda_config::warp_size)
    apply(const ValueType *__restrict__ blocks,
          preconditioner::block_interleaved_storage_scheme<IndexType>
              storage_scheme,
          const IndexType *__restrict__ block_ptrs, size_type num_blocks,
          const ValueType *__restrict__ b, int32 b_stride,
          ValueType *__restrict__ x, int32 x_stride)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    if (block_id >= num_blocks) {
        return;
    }
    const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];
    ValueType v = zero<ValueType>();
    if (subwarp.thread_rank() < block_size) {
        v = b[(block_ptrs[block_id] + subwarp.thread_rank()) * b_stride];
    }
    multiply_vec<max_block_size>(
        subwarp, block_size, v,
        blocks + storage_scheme.get_global_block_offset(block_id) +
            subwarp.thread_rank(),
        storage_scheme.get_stride(), x + block_ptrs[block_id] * x_stride,
        x_stride,
        [](ValueType &result, const ValueType &out) { result = out; });
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void __launch_bounds__(warps_per_block *cuda_config::warp_size)
    advanced_apply(const ValueType *__restrict__ blocks,
                   preconditioner::block_interleaved_storage_scheme<IndexType>
                       storage_scheme,
                   const IndexType *__restrict__ block_ptrs,
                   size_type num_blocks, const ValueType *__restrict__ alpha,
                   const ValueType *__restrict__ b, int32 b_stride,
                   ValueType *__restrict__ x, int32 x_stride)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    if (block_id >= num_blocks) {
        return;
    }
    const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];
    ValueType v = zero<ValueType>();
    if (subwarp.thread_rank() < block_size) {
        v = alpha[0] *
            b[(block_ptrs[block_id] + subwarp.thread_rank()) * b_stride];
    }
    multiply_vec<max_block_size>(
        subwarp, block_size, v,
        blocks + storage_scheme.get_global_block_offset(block_id) +
            subwarp.thread_rank(),
        storage_scheme.get_stride(), x + block_ptrs[block_id] * x_stride,
        x_stride,
        [](ValueType &result, const ValueType &out) { result += out; });
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void __launch_bounds__(warps_per_block *cuda_config::warp_size)
    adaptive_apply(const ValueType *__restrict__ blocks,
                   preconditioner::block_interleaved_storage_scheme<IndexType>
                       storage_scheme,
                   const precision_reduction *__restrict__ block_precisions,
                   const IndexType *__restrict__ block_ptrs,
                   size_type num_blocks, const ValueType *__restrict__ b,
                   int32 b_stride, ValueType *__restrict__ x, int32 x_stride)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    if (block_id >= num_blocks) {
        return;
    }
    const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];
    ValueType v = zero<ValueType>();
    if (subwarp.thread_rank() < block_size) {
        v = b[(block_ptrs[block_id] + subwarp.thread_rank()) * b_stride];
    }
    GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(
        ValueType, block_precisions[block_id],
        multiply_vec<max_block_size>(
            subwarp, block_size, v,
            reinterpret_cast<const resolved_precision *>(
                blocks + storage_scheme.get_group_offset(block_id)) +
                storage_scheme.get_block_offset(block_id) +
                subwarp.thread_rank(),
            storage_scheme.get_stride(), x + block_ptrs[block_id] * x_stride,
            x_stride,
            [](ValueType &result, const ValueType &out) { result = out; }));
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void __launch_bounds__(warps_per_block *cuda_config::warp_size)
    advanced_adaptive_apply(
        const ValueType *__restrict__ blocks,
        preconditioner::block_interleaved_storage_scheme<IndexType>
            storage_scheme,
        const precision_reduction *__restrict__ block_precisions,
        const IndexType *__restrict__ block_ptrs, size_type num_blocks,
        const ValueType *__restrict__ alpha, const ValueType *__restrict__ b,
        int32 b_stride, ValueType *__restrict__ x, int32 x_stride)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    if (block_id >= num_blocks) {
        return;
    }
    const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];
    auto alpha_val = alpha == nullptr ? one<ValueType>() : alpha[0];
    ValueType v = zero<ValueType>();
    if (subwarp.thread_rank() < block_size) {
        v = alpha[0] *
            b[(block_ptrs[block_id] + subwarp.thread_rank()) * b_stride];
    }
    GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(
        ValueType, block_precisions[block_id],
        multiply_vec<max_block_size>(
            subwarp, block_size, v,
            reinterpret_cast<const resolved_precision *>(
                blocks + storage_scheme.get_group_offset(block_id)) +
                storage_scheme.get_block_offset(block_id) +
                subwarp.thread_rank(),
            storage_scheme.get_stride(), x + block_ptrs[block_id] * x_stride,
            x_stride,
            [](ValueType &result, const ValueType &out) { result += out; }));
}


}  // namespace
}  // namespace kernel


namespace {


// a total of 32 warps (1024 threads)
constexpr int default_block_size = 32;
// with current architectures, at most 32 warps can be scheduled per SM (and
// current GPUs have at most 84 SMs)
constexpr int default_grid_size = 32 * 32 * 128;


constexpr int get_larger_power(int value, int guess = 1)
{
    return guess >= value ? guess : get_larger_power(value, guess << 1);
}


template <int warps_per_block, int max_block_size, typename ValueType,
          typename IndexType>
void generate(syn::value_list<int, max_block_size>,
              const matrix::Csr<ValueType, IndexType> *mtx,
              remove_complex<ValueType> accuracy, ValueType *block_data,
              const preconditioner::block_interleaved_storage_scheme<IndexType>
                  &storage_scheme,
              remove_complex<ValueType> *conditioning,
              precision_reduction *block_precisions,
              const IndexType *block_ptrs, size_type num_blocks)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    constexpr int blocks_per_warp = cuda_config::warp_size / subwarp_size;
    const dim3 grid_size(ceildiv(num_blocks, warps_per_block * blocks_per_warp),
                         1, 1);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    if (block_precisions) {
        kernel::adaptive_generate<max_block_size, subwarp_size, warps_per_block>
            <<<grid_size, block_size, 0, 0>>>(
                mtx->get_size()[0], mtx->get_const_row_ptrs(),
                mtx->get_const_col_idxs(),
                as_cuda_type(mtx->get_const_values()), as_cuda_type(accuracy),
                as_cuda_type(block_data), storage_scheme,
                as_cuda_type(conditioning), block_precisions, block_ptrs,
                num_blocks);
    } else {
        kernel::generate<max_block_size, subwarp_size, warps_per_block>
            <<<grid_size, block_size, 0, 0>>>(
                mtx->get_size()[0], mtx->get_const_row_ptrs(),
                mtx->get_const_col_idxs(),
                as_cuda_type(mtx->get_const_values()), as_cuda_type(block_data),
                storage_scheme, block_ptrs, num_blocks);
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_generate, generate);


template <int warps_per_block, int max_block_size, typename ValueType,
          typename IndexType>
void apply(syn::value_list<int, max_block_size>, size_type num_blocks,
           const precision_reduction *block_precisions,
           const IndexType *block_pointers, const ValueType *blocks,
           const preconditioner::block_interleaved_storage_scheme<IndexType>
               &storage_scheme,
           const ValueType *b, size_type b_stride, ValueType *x,
           size_type x_stride)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    constexpr int blocks_per_warp = cuda_config::warp_size / subwarp_size;
    const dim3 grid_size(ceildiv(num_blocks, warps_per_block * blocks_per_warp),
                         1, 1);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    if (block_precisions) {
        kernel::adaptive_apply<max_block_size, subwarp_size, warps_per_block>
            <<<grid_size, block_size, 0, 0>>>(
                as_cuda_type(blocks), storage_scheme, block_precisions,
                block_pointers, num_blocks, as_cuda_type(b), b_stride,
                as_cuda_type(x), x_stride);
    } else {
        kernel::apply<max_block_size, subwarp_size, warps_per_block>
            <<<grid_size, block_size, 0, 0>>>(
                as_cuda_type(blocks), storage_scheme, block_pointers,
                num_blocks, as_cuda_type(b), b_stride, as_cuda_type(x),
                x_stride);
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_apply, apply);


template <int warps_per_block, int max_block_size, typename ValueType,
          typename IndexType>
void advanced_apply(
    syn::value_list<int, max_block_size>, size_type num_blocks,
    const precision_reduction *block_precisions,
    const IndexType *block_pointers, const ValueType *blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>
        &storage_scheme,
    const ValueType *alpha, const ValueType *b, size_type b_stride,
    ValueType *x, size_type x_stride)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    constexpr int blocks_per_warp = cuda_config::warp_size / subwarp_size;
    const dim3 grid_size(ceildiv(num_blocks, warps_per_block * blocks_per_warp),
                         1, 1);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    if (block_precisions) {
        kernel::advanced_adaptive_apply<max_block_size, subwarp_size,
                                        warps_per_block>
            <<<grid_size, block_size, 0, 0>>>(
                as_cuda_type(blocks), storage_scheme, block_precisions,
                block_pointers, num_blocks, as_cuda_type(alpha),
                as_cuda_type(b), b_stride, as_cuda_type(x), x_stride);
    } else {
        kernel::advanced_apply<max_block_size, subwarp_size, warps_per_block>
            <<<grid_size, block_size, 0, 0>>>(
                as_cuda_type(blocks), storage_scheme, block_pointers,
                num_blocks, as_cuda_type(alpha), as_cuda_type(b), b_stride,
                as_cuda_type(x), x_stride);
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_advanced_apply, advanced_apply);


template <int warps_per_block>
__global__
__launch_bounds__(warps_per_block *cuda_config::warp_size) void duplicate_array(
    const precision_reduction *__restrict__ source, size_type source_size,
    precision_reduction *__restrict__ dest, size_type dest_size)
{
    auto grid = group::this_grid();
    if (grid.thread_rank() >= dest_size) {
        return;
    }
    for (auto i = grid.thread_rank(); i < dest_size; i += grid.size()) {
        dest[i] = source[i % source_size];
    }
}


template <typename IndexType>
__global__ void compare_adjacent_rows(size_type num_rows, int32 max_block_size,
                                      const IndexType *__restrict__ row_ptrs,
                                      const IndexType *__restrict__ col_idx,
                                      bool *__restrict__ matching_next_row)
{
    const auto global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    const auto local_tid = threadIdx.x % cuda_config::warp_size;
    const auto warp_id = global_tid / cuda_config::warp_size;
    const auto warp = group::tiled_partition<cuda_config::warp_size>(
        group::this_thread_block());

    if (warp_id >= num_rows - 1) {
        return;
    }

    const auto curr_row_start = row_ptrs[warp_id];
    const auto next_row_start = row_ptrs[warp_id + 1];
    const auto next_row_end = row_ptrs[warp_id + 2];

    const auto nz_this_row = next_row_end - next_row_start;
    const auto nz_prev_row = next_row_start - curr_row_start;

    if (nz_this_row != nz_prev_row) {
        matching_next_row[warp_id] = false;
        return;
    }
    size_type steps = ceildiv(nz_this_row, cuda_config::warp_size);
    for (size_type i = 0; i < steps; i++) {
        auto j = local_tid + i * cuda_config::warp_size;
        auto prev_col = (curr_row_start + j < next_row_start)
                            ? col_idx[curr_row_start + j]
                            : 0;
        auto this_col = (curr_row_start + j < next_row_start)
                            ? col_idx[next_row_start + j]
                            : 0;
        if (warp.any(prev_col != this_col)) {
            matching_next_row[warp_id] = false;
            return;
        }
    }
    matching_next_row[warp_id] = true;
}


template <typename IndexType>
__global__ void generate_natural_block_pointer(
    size_type num_rows, int32 max_block_size,
    const bool *__restrict__ matching_next_row,
    IndexType *__restrict__ block_ptrs, size_type *__restrict__ num_blocks_arr)
{
    block_ptrs[0] = 0;
    if (num_rows == 0) {
        return;
    }
    size_type num_blocks = 1;
    int32 current_block_size = 1;
    for (size_type i = 1; i < num_rows; ++i) {
        if ((matching_next_row[i]) && (current_block_size < max_block_size)) {
            ++current_block_size;
        } else {
            block_ptrs[num_blocks] =
                block_ptrs[num_blocks - 1] + current_block_size;
            ++num_blocks;
            current_block_size = 1;
        }
    }
    block_ptrs[num_blocks] = block_ptrs[num_blocks - 1] + current_block_size;
    num_blocks_arr[0] = num_blocks;
}


template <typename ValueType, typename IndexType>
size_type find_natural_blocks(std::shared_ptr<const CudaExecutor> exec,
                              const matrix::Csr<ValueType, IndexType> *mtx,
                              int32 max_block_size,
                              IndexType *__restrict__ block_ptrs)
{
    Array<size_type> nums(exec, 1);

    Array<bool> matching_next_row(exec, mtx->get_size()[0]);

    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(mtx->get_size()[0] * cuda_config::warp_size, block_size.x), 1,
        1);
    compare_adjacent_rows<<<grid_size, block_size, 0, 0>>>(
        mtx->get_size()[0], max_block_size, mtx->get_const_row_ptrs(),
        mtx->get_const_col_idxs(), matching_next_row.get_data());
    generate_natural_block_pointer<<<1, 1, 0, 0>>>(
        mtx->get_size()[0], max_block_size, matching_next_row.get_const_data(),
        block_ptrs, nums.get_data());
    nums.set_executor(exec->get_master());
    return nums.get_const_data()[0];
}


template <typename IndexType>
__global__ void agglomerate_supervariables_kernel(
    int32 max_block_size, size_type num_natural_blocks,
    IndexType *__restrict__ block_ptrs, size_type *__restrict__ num_blocks_arr)
{
    num_blocks_arr[0] = 0;
    if (num_natural_blocks == 0) {
        return;
    }
    size_type num_blocks = 1;
    int32 current_block_size = block_ptrs[1] - block_ptrs[0];
    for (size_type i = 1; i < num_natural_blocks; ++i) {
        const int32 block_size = block_ptrs[i + 1] - block_ptrs[i];
        if (current_block_size + block_size <= max_block_size) {
            current_block_size += block_size;
        } else {
            block_ptrs[num_blocks] = block_ptrs[i];
            ++num_blocks;
            current_block_size = block_size;
        }
    }
    block_ptrs[num_blocks] = block_ptrs[num_natural_blocks];
    num_blocks_arr[0] = num_blocks;
}


template <typename IndexType>
inline size_type agglomerate_supervariables(
    std::shared_ptr<const CudaExecutor> exec, int32 max_block_size,
    size_type num_natural_blocks, IndexType *block_ptrs)
{
    Array<size_type> nums(exec, 1);

    agglomerate_supervariables_kernel<<<1, 1, 0, 0>>>(
        max_block_size, num_natural_blocks, block_ptrs, nums.get_data());

    nums.set_executor(exec->get_master());
    return nums.get_const_data()[0];
}


}  // namespace


/**
 * @brief The Jacobi preconditioner.
 * @ref Jacobi
 * @ingroup jacobi
 */
namespace jacobi {


void initialize_precisions(std::shared_ptr<const CudaExecutor> exec,
                           const Array<precision_reduction> &source,
                           Array<precision_reduction> &precisions)
{
    const auto block_size = default_block_size * cuda_config::warp_size;
    const auto grid_size = min(
        default_grid_size,
        static_cast<int32>(ceildiv(precisions.get_num_elems(), block_size)));
    duplicate_array<default_block_size><<<grid_size, block_size>>>(
        source.get_const_data(), source.get_num_elems(), precisions.get_data(),
        precisions.get_num_elems());
}


template <typename ValueType, typename IndexType>
void find_blocks(std::shared_ptr<const CudaExecutor> exec,
                 const matrix::Csr<ValueType, IndexType> *system_matrix,
                 uint32 max_block_size, size_type &num_blocks,
                 Array<IndexType> &block_pointers)
{
    auto num_natural_blocks = find_natural_blocks(
        exec, system_matrix, max_block_size, block_pointers.get_data());
    num_blocks = agglomerate_supervariables(
        exec, max_block_size, num_natural_blocks, block_pointers.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_FIND_BLOCKS_KERNEL);


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const CudaExecutor> exec,
              const matrix::Csr<ValueType, IndexType> *system_matrix,
              size_type num_blocks, uint32 max_block_size,
              remove_complex<ValueType> accuracy,
              const preconditioner::block_interleaved_storage_scheme<IndexType>
                  &storage_scheme,
              Array<remove_complex<ValueType>> &conditioning,
              Array<precision_reduction> &block_precisions,
              const Array<IndexType> &block_pointers, Array<ValueType> &blocks)
{
    select_generate(compiled_kernels(),
                    [&](int compiled_block_size) {
                        return max_block_size <= compiled_block_size;
                    },
                    syn::value_list<int, cuda_config::min_warps_per_block>(),
                    syn::type_list<>(), system_matrix, accuracy,
                    blocks.get_data(), storage_scheme, conditioning.get_data(),
                    block_precisions.get_data(),
                    block_pointers.get_const_data(), num_blocks);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_GENERATE_KERNEL);


template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const CudaExecutor> exec, size_type num_blocks,
           uint32 max_block_size,
           const preconditioner::block_interleaved_storage_scheme<IndexType>
               &storage_scheme,
           const Array<precision_reduction> &block_precisions,
           const Array<IndexType> &block_pointers,
           const Array<ValueType> &blocks,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *x)
{
    // TODO: write a special kernel for multiple RHS
    dense::scale(exec, beta, x);
    for (size_type col = 0; col < b->get_size()[1]; ++col) {
        select_advanced_apply(
            compiled_kernels(),
            [&](int compiled_block_size) {
                return max_block_size <= compiled_block_size;
            },
            syn::value_list<int, cuda_config::min_warps_per_block>(),
            syn::type_list<>(), num_blocks, block_precisions.get_const_data(),
            block_pointers.get_const_data(), blocks.get_const_data(),
            storage_scheme, alpha->get_const_values(),
            b->get_const_values() + col, b->get_stride(), x->get_values() + col,
            x->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_JACOBI_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void simple_apply(
    std::shared_ptr<const CudaExecutor> exec, size_type num_blocks,
    uint32 max_block_size,
    const preconditioner::block_interleaved_storage_scheme<IndexType>
        &storage_scheme,
    const Array<precision_reduction> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *x)
{
    // TODO: write a special kernel for multiple RHS
    for (size_type col = 0; col < b->get_size()[1]; ++col) {
        select_apply(compiled_kernels(),
                     [&](int compiled_block_size) {
                         return max_block_size <= compiled_block_size;
                     },
                     syn::value_list<int, cuda_config::min_warps_per_block>(),
                     syn::type_list<>(), num_blocks,
                     block_precisions.get_const_data(),
                     block_pointers.get_const_data(), blocks.get_const_data(),
                     storage_scheme, b->get_const_values() + col,
                     b->get_stride(), x->get_values() + col, x->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_SIMPLE_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(
    std::shared_ptr<const CudaExecutor> exec, size_type num_blocks,
    const Array<precision_reduction> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>
        &storage_scheme,
    ValueType *result_values, size_type result_stride) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_CONVERT_TO_DENSE_KERNEL);


}  // namespace jacobi
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
