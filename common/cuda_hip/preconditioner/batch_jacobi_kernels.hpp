// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_PRECONDITIONER_BATCH_JACOBI_KERNELS_HPP_
#define GKO_COMMON_CUDA_HIP_PRECONDITIONER_BATCH_JACOBI_KERNELS_HPP_


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>

#include "common/cuda_hip/base/batch_multi_vector_kernels.hpp"
#include "common/cuda_hip/base/batch_struct.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/matrix/batch_csr_kernels.hpp"
#include "common/cuda_hip/matrix/batch_dense_kernels.hpp"
#include "common/cuda_hip/matrix/batch_ell_kernels.hpp"
#include "common/cuda_hip/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace batch_single_kernels {


__global__ void compute_block_storage_kernel(
    const gko::size_type num_blocks,
    const int* const __restrict__ block_pointers,
    int* const __restrict__ blocks_cumulative_offsets)
{
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < num_blocks; i += blockDim.x * gridDim.x) {
        const auto bsize = block_pointers[i + 1] - block_pointers[i];
        blocks_cumulative_offsets[i] = bsize * bsize;
    }
}


__global__ __launch_bounds__(default_block_size) void find_row_block_map_kernel(
    const gko::size_type num_blocks,
    const int* const __restrict__ block_pointers,
    int* const __restrict__ map_block_to_row)
{
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int block_idx = tid; block_idx < num_blocks;
         block_idx += blockDim.x * gridDim.x) {
        for (int i = block_pointers[block_idx];
             i < block_pointers[block_idx + 1]; i++) {
            map_block_to_row[i] = block_idx;  // uncoalesced accesses
        }
    }
}


__global__
__launch_bounds__(default_block_size) void extract_common_block_pattern_kernel(
    const int nrows, const int* const __restrict__ sys_row_ptrs,
    const int* const __restrict__ sys_col_idxs, const gko::size_type num_blocks,
    const int* const __restrict__ blocks_cumulative_offsets,
    const int* const __restrict__ block_pointers,
    const int* const __restrict__ map_block_to_row, int* const blocks_pattern)
{
    constexpr auto tile_size =
        config::warp_size;  // use full warp for coalesced memory accesses
    auto thread_block = group::this_thread_block();
    auto warp_grp = group::tiled_partition<tile_size>(thread_block);
    const int warp_id_in_grid = thread::get_subwarp_id_flat<tile_size, int>();
    const int total_num_warps_in_grid =
        thread::get_subwarp_num_flat<tile_size, int>();
    const int id_within_warp = warp_grp.thread_rank();

    // one warp per row of the matrix
    for (int row_idx = warp_id_in_grid; row_idx < nrows;
         row_idx += total_num_warps_in_grid) {
        const int block_idx = map_block_to_row[row_idx];
        const int idx_start = block_pointers[block_idx];
        const int idx_end = block_pointers[block_idx + 1];
        int* __restrict__ pattern_ptr =
            blocks_pattern + gko::detail::batch_jacobi::get_block_offset(
                                 block_idx, blocks_cumulative_offsets);
        const auto stride =
            gko::detail::batch_jacobi::get_stride(block_idx, block_pointers);

        for (int i = sys_row_ptrs[row_idx] + id_within_warp;
             i < sys_row_ptrs[row_idx + 1]; i += tile_size) {
            const int col_idx = sys_col_idxs[i];  // coalesced accesses

            if (col_idx >= idx_start && col_idx < idx_end) {
                // element at (row_idx, col_idx) is part of the diagonal block
                // store it into the pattern
                const int dense_block_row = row_idx - idx_start;
                const int dense_block_col = col_idx - idx_start;

                // The pattern is stored in row-major order
                pattern_ptr[dense_block_row * stride + dense_block_col] =
                    i;  // coalesced accesses
            }
        }
    }
}


template <typename Group, typename ValueType>
__device__ __forceinline__ int choose_pivot(
    Group subwarp_grp, const int block_size,
    const ValueType* const __restrict__ block_row, const int& perm, const int k)
{
    auto my_abs_ele = abs(block_row[k]);
    if (perm > -1) {
        my_abs_ele = -one<remove_complex<ValueType>>();
    }
    if (subwarp_grp.thread_rank() >= block_size) {
        my_abs_ele = -one<remove_complex<ValueType>>();
    }
    subwarp_grp.sync();
    int my_piv_idx = subwarp_grp.thread_rank();
#pragma unroll
    for (int a = subwarp_grp.size() / 2; a > 0; a /= 2) {
        const auto abs_ele_other = subwarp_grp.shfl_down(my_abs_ele, a);
        const int piv_idx_other = subwarp_grp.shfl_down(my_piv_idx, a);

        if (my_abs_ele < abs_ele_other) {
            my_abs_ele = abs_ele_other;
            my_piv_idx = piv_idx_other;
        }
    }
    subwarp_grp.sync();
    const int ipiv = subwarp_grp.shfl(my_piv_idx, 0);
    return ipiv;
}


template <typename Group, typename ValueType>
__device__ __forceinline__ void invert_dense_block(Group subwarp_grp,
                                                   const int block_size,
                                                   ValueType* const block_row,
                                                   int& perm)
{
    // Gauss Jordan Elimination with implicit pivoting
    for (int k = 0; k < block_size; k++) {
        // implicit pivoting
        const int ipiv = choose_pivot(subwarp_grp, block_size, block_row, perm,
                                      k);  // pivot index
        if (subwarp_grp.thread_rank() == ipiv) {
            perm = k;
        }
        const ValueType d =
            (subwarp_grp.shfl(block_row[k], ipiv) == zero<ValueType>())
                ? one<ValueType>()
                : subwarp_grp.shfl(block_row[k], ipiv);
        // scale kth col
        block_row[k] /= -d;
        if (subwarp_grp.thread_rank() == ipiv) {
            block_row[k] = zero<ValueType>();
        }
        const ValueType row_val = block_row[k];
        // rank-1 update
        for (int col = 0; col < block_size; col++) {
            const ValueType col_val = subwarp_grp.shfl(block_row[col], ipiv);
            block_row[col] += row_val * col_val;
        }
        // Computations for the threads of the subwarp having local id >=
        // block_size are meaningless.

        // scale ipiv th row
        if (subwarp_grp.thread_rank() == ipiv) {
            for (int i = 0; i < block_size; i++) {
                block_row[i] /= d;
            }
            block_row[k] = one<ValueType>() / d;
        }
    }
}


template <int subwarp_size, typename ValueType>
__global__
__launch_bounds__(default_block_size) void compute_block_jacobi_kernel(
    const gko::size_type nbatch, const int nnz, const ValueType* const A_vals,
    const gko::size_type num_blocks,
    const int* const __restrict__ blocks_cumulative_offsets,
    const int* const __restrict__ block_pointers,
    const int* const blocks_pattern, ValueType* const blocks)
{
    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<subwarp_size>(thread_block);
    const int subwarp_id_in_grid =
        thread::get_subwarp_id_flat<subwarp_size, int>();
    const int total_num_subwarps_in_grid =
        thread::get_subwarp_num_flat<subwarp_size, int>();
    const int id_within_subwarp = subwarp_grp.thread_rank();

    // one subwarp per small diagonal block
    for (size_type i = subwarp_id_in_grid; i < nbatch * num_blocks;
         i += total_num_subwarps_in_grid) {
        const auto batch_idx = i / num_blocks;
        const auto block_idx = i % num_blocks;

        ValueType block_row[subwarp_size];
        const auto block_size =
            block_pointers[block_idx + 1] - block_pointers[block_idx];
        assert(block_size <= subwarp_size);

        const int* __restrict__ current_block_pattern =
            blocks_pattern + gko::detail::batch_jacobi::get_block_offset(
                                 block_idx, blocks_cumulative_offsets);
        ValueType* __restrict__ current_block_data =
            blocks +
            gko::detail::batch_jacobi::get_global_block_offset(
                batch_idx, num_blocks, block_idx, blocks_cumulative_offsets);
        const auto stride =
            gko::detail::batch_jacobi::get_stride(block_idx, block_pointers);

        // each thread of the subwarp stores the column of the dense block/row
        // of the transposed block in its local memory
        if (id_within_subwarp < block_size) {
            for (int a = 0; a < block_size; a++) {
                const auto idx = current_block_pattern
                    [a * gko::detail::batch_jacobi::get_stride(block_idx,
                                                               block_pointers) +
                     id_within_subwarp];  // coalesced
                                          // accesses
                ValueType val_to_fill = zero<ValueType>();
                if (idx >= 0) {
                    assert(idx < nnz);
                    val_to_fill = A_vals[idx + nnz * batch_idx];
                }
                block_row[a] = val_to_fill;
            }
        }

        int perm = -1;

        // invert
        invert_dense_block(subwarp_grp, block_size, block_row,
                           perm);  // invert the transpose of the dense block.
        // Note: Each thread of the subwarp has a row of the block to be
        // inverted. (local id: 0 thread has 0th row, 1st thread has 1st row and
        // so on..)
        // If block_size < subwarp_size, then threads with local id >=
        // block_size do not mean anything. Also, values in the block_row for
        // index >= block_size are meaningless

        subwarp_grp.sync();

        // write back the transpose of the transposed inverse matrix to block
        // array
        for (int a = 0; a < block_size; a++) {
            const int col_inv_transposed_mat = a;
            const int col = subwarp_grp.shfl(perm, a);  // column permutation
            const int row_inv_transposed_mat =
                perm;  // accumulated row swaps during pivoting
            const auto val_to_write = block_row[col];

            const int row_diag_block = col_inv_transposed_mat;
            const int col_diag_block = row_inv_transposed_mat;

            if (id_within_subwarp < block_size) {
                current_block_data[row_diag_block * stride + col_diag_block] =
                    val_to_write;  // non-coalesced accesses due to pivoting
            }
        }
    }
}


}  // namespace batch_single_kernels
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif
