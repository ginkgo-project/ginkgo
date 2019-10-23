#include "hip/hip_runtime.h"
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
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The Jacobi preconditioner namespace.
 * @ref Jacobi
 * @ingroup jacobi
 */
namespace jacobi {
namespace {


// a total of 32 warps (1024 threads)
constexpr int default_block_size = 32;
// with current architectures, at most 32 warps can be scheduled per SM (and
// current GPUs have at most 84 SMs)
constexpr int default_grid_size = 32 * 32 * 128;


template <int warps_per_block>
__global__
__launch_bounds__(warps_per_block *hip_config::warp_size) void duplicate_array(
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
    const auto local_tid = threadIdx.x % hip_config::warp_size;
    const auto warp_id = global_tid / hip_config::warp_size;
    const auto warp = group::tiled_partition<hip_config::warp_size>(
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
    size_type steps = ceildiv(nz_this_row, hip_config::warp_size);
    for (size_type i = 0; i < steps; i++) {
        auto j = local_tid + i * hip_config::warp_size;
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
size_type find_natural_blocks(std::shared_ptr<const HipExecutor> exec,
                              const matrix::Csr<ValueType, IndexType> *mtx,
                              int32 max_block_size,
                              IndexType *__restrict__ block_ptrs)
{
    Array<size_type> nums(exec, 1);

    Array<bool> matching_next_row(exec, mtx->get_size()[0]);

    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(mtx->get_size()[0] * hip_config::warp_size, block_size.x), 1,
        1);
    hipLaunchKernelGGL(compare_adjacent_rows, dim3(grid_size), dim3(block_size), 0, 0, 
        mtx->get_size()[0], max_block_size, mtx->get_const_row_ptrs(),
        mtx->get_const_col_idxs(), matching_next_row.get_data());
    hipLaunchKernelGGL(generate_natural_block_pointer, dim3(1), dim3(1), 0, 0, 
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
    std::shared_ptr<const HipExecutor> exec, int32 max_block_size,
    size_type num_natural_blocks, IndexType *block_ptrs)
{
    Array<size_type> nums(exec, 1);

    hipLaunchKernelGGL(agglomerate_supervariables_kernel, dim3(1), dim3(1), 0, 0, 
        max_block_size, num_natural_blocks, block_ptrs, nums.get_data());

    nums.set_executor(exec->get_master());
    return nums.get_const_data()[0];
}


}  // namespace


void initialize_precisions(std::shared_ptr<const HipExecutor> exec,
                           const Array<precision_reduction> &source,
                           Array<precision_reduction> &precisions)
{
    const auto block_size = default_block_size * hip_config::warp_size;
    const auto grid_size = min(
        default_grid_size,
        static_cast<int32>(ceildiv(precisions.get_num_elems(), block_size)));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(duplicate_array<default_block_size>), dim3(grid_size), dim3(block_size), 0, 0, 
        source.get_const_data(), source.get_num_elems(), precisions.get_data(),
        precisions.get_num_elems());
}


template <typename ValueType, typename IndexType>
void find_blocks(std::shared_ptr<const HipExecutor> exec,
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
void convert_to_dense(
    std::shared_ptr<const HipExecutor> exec, size_type num_blocks,
    const Array<precision_reduction> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>
        &storage_scheme,
    ValueType *result_values, size_type result_stride) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_CONVERT_TO_DENSE_KERNEL);


}  // namespace jacobi
}  // namespace hip
}  // namespace kernels
}  // namespace gko
