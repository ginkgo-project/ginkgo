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
#include "gpu/components/diagonal_block_manipulation.cuh"
#include "gpu/components/thread_ids.cuh"
#include "gpu/components/uninitialized_array.hpp"
#include "gpu/components/warp_blas.cuh"


namespace gko {
namespace kernels {
namespace gpu {
namespace kernel {
namespace {

template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void __launch_bounds__(warps_per_block *cuda_config::warp_size)
    generate(size_type num_rows, const IndexType *__restrict__ row_ptrs,
             const IndexType *__restrict__ col_idxs,
             const ValueType *__restrict__ values,
             ValueType *__restrict__ block_data, size_type stride,
             const IndexType *__restrict__ block_ptrs, size_type num_blocks)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    ValueType row[max_block_size];
    __shared__ UninitializedArray<ValueType, max_block_size * warps_per_block>
        workspace;
    device::csr::extract_transposed_diag_blocks<max_block_size, subwarp_size,
                                                warps_per_block>(
        row_ptrs, col_idxs, values, block_ptrs, num_blocks, row, 1,
        workspace + threadIdx.z * max_block_size);
    if (block_id < num_blocks) {
        const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];
        auto perm = threadIdx.x;
        auto trans_perm = threadIdx.x;
        warp::invert_block<max_block_size, subwarp_size>(block_size, row, perm,
                                                         trans_perm);
        warp::copy_matrix<max_block_size, subwarp_size>(
            block_size, row, 1, perm, trans_perm,
            block_data + (block_ptrs[block_id] * stride), stride);
    }
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void __launch_bounds__(warps_per_block *cuda_config::warp_size)
    apply(const ValueType *__restrict__ blocks, int32 stride,
          const IndexType *__restrict__ block_ptrs, size_type num_blocks,
          const ValueType *__restrict__ b, int32 b_stride,
          ValueType *__restrict__ x, int32 x_stride)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    if (block_id >= num_blocks) {
        return;
    }
    const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];
    ValueType v = zero<ValueType>();
    if (threadIdx.x < block_size) {
        v = b[(block_ptrs[block_id] + threadIdx.x) * b_stride];
    }
    warp::multiply_transposed_vec<max_block_size, subwarp_size>(
        block_size, v, blocks + block_ptrs[block_id] * stride + threadIdx.x,
        stride, x + block_ptrs[block_id] * x_stride, x_stride);
}


}  // namespace
}  // namespace kernel


namespace {

constexpr int default_block_size = 32;

constexpr int get_larger_power(int value, int guess = 1)
{
    return guess >= value ? guess : get_larger_power(value, guess << 1);
}


template <int warps_per_block, int max_block_size, typename ValueType,
          typename IndexType>
void generate(syn::compile_int_list<max_block_size>,
              const matrix::Csr<ValueType, IndexType> *mtx,
              ValueType *block_data, size_type stride,
              const IndexType *block_ptrs, size_type num_blocks)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    constexpr int blocks_per_warp = cuda_config::warp_size / subwarp_size;
    const dim3 grid_size(ceildiv(num_blocks, warps_per_block * blocks_per_warp),
                         1, 1);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    kernel::generate<max_block_size, subwarp_size, warps_per_block>
        <<<grid_size, block_size, 0, 0>>>(
            mtx->get_size().num_rows, mtx->get_const_row_ptrs(),
            mtx->get_const_col_idxs(), as_cuda_type(mtx->get_const_values()),
            as_cuda_type(block_data), stride, block_ptrs, num_blocks);
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_generate, generate);


template <int warps_per_block, int max_block_size, typename ValueType,
          typename IndexType>
void apply(syn::compile_int_list<max_block_size>, size_type num_blocks,
           const IndexType *block_pointers, const ValueType *blocks,
           size_type block_stride, const ValueType *b, size_type b_stride,
           ValueType *x, size_type x_stride)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    constexpr int blocks_per_warp = cuda_config::warp_size / subwarp_size;
    const dim3 grid_size(ceildiv(num_blocks, warps_per_block * blocks_per_warp),
                         1, 1);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    kernel::apply<max_block_size, subwarp_size, warps_per_block>
        <<<grid_size, block_size, 0, 0>>>(
            as_cuda_type(blocks), block_stride, block_pointers, num_blocks,
            as_cuda_type(b), b_stride, as_cuda_type(x), x_stride);
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_apply, apply);


using compiled_kernels = syn::compile_int_list<1, 13, 16, 32>;


template <typename IndexType>
__global__ void compare_adjacent_rows(size_type num_rows, int32 max_block_size,
                                      const IndexType *__restrict__ row_ptrs,
                                      const IndexType *__restrict__ col_idx,
                                      bool *matching_prev_row)
{
    const auto global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    const auto local_tid = threadIdx.x % cuda_config::warp_size;
    const auto warp_id = global_tid / cuda_config::warp_size;

    if (warp_id >= num_rows - 1) {
        return;
    }

    const auto prev_row_start = row_ptrs[warp_id];
    const auto curr_row_start = row_ptrs[warp_id + 1];
    const auto next_row_start = row_ptrs[warp_id + 2];

    const auto nz_this_row = next_row_start - curr_row_start;
    const auto nz_prev_row = curr_row_start - prev_row_start;

    if (nz_this_row != nz_prev_row) {
        matching_prev_row[warp_id] = false;
        return;
    }
    size_type steps = ceildiv(nz_this_row, cuda_config::warp_size);
    for (size_type i = 0; i < steps; i++) {
        auto j = local_tid + i * cuda_config::warp_size;
        auto prev_col = (prev_row_start + j < curr_row_start)
                            ? col_idx[prev_row_start + j]
                            : 0;
        auto this_col = (prev_row_start + j < curr_row_start)
                            ? col_idx[curr_row_start + j]
                            : 0;
        if (warp::any(prev_col != this_col)) {
            matching_prev_row[warp_id] = false;
            return;
        }
    }
    matching_prev_row[warp_id] = true;
}


template <typename IndexType>
__global__ void generate_natural_block_pointer(
    size_type num_rows, int32 max_block_size,
    const bool *__restrict__ matching_prev_row, IndexType *block_ptrs,
    size_type *num_blocks_arr)
{
    block_ptrs[0] = 0;
    if (num_rows == 0) {
        return;
    }
    size_type num_blocks = 1;
    int32 current_block_size = 1;
    for (size_type i = 1; i < num_rows; ++i) {
        if ((matching_prev_row[i]) && (current_block_size < max_block_size)) {
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
size_type find_natural_blocks(std::shared_ptr<const GpuExecutor> exec,
                              const matrix::Csr<ValueType, IndexType> *mtx,
                              int32 max_block_size, IndexType *block_ptrs)
{
    Array<size_type> nums(exec, 1);

    Array<bool> matching_prev_row(exec, mtx->get_size().num_rows);

    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(mtx->get_size().num_rows * cuda_config::warp_size,
                block_size.x),
        1, 1);
    compare_adjacent_rows<<<grid_size, block_size, 0, 0>>>(
        mtx->get_size().num_rows, max_block_size, mtx->get_const_row_ptrs(),
        mtx->get_const_col_idxs(), matching_prev_row.get_data());
    generate_natural_block_pointer<<<1, 1, 0, 0>>>(
        mtx->get_size().num_rows, max_block_size,
        matching_prev_row.get_const_data(), block_ptrs, nums.get_data());
    nums.set_executor(exec->get_master());
    return nums.get_const_data()[0];
}


template <typename IndexType>
__global__ void agglomerate_supervariables_kernel(int32 max_block_size,
                                                  size_type num_natural_blocks,
                                                  IndexType *block_ptrs,
                                                  size_type *num_blocks_arr)
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
    std::shared_ptr<const GpuExecutor> exec, int32 max_block_size,
    size_type num_natural_blocks, IndexType *block_ptrs)
{
    Array<size_type> nums(exec, 1);

    agglomerate_supervariables_kernel<<<1, 1, 0, 0>>>(
        max_block_size, num_natural_blocks, block_ptrs, nums.get_data());

    nums.set_executor(exec->get_master());
    return nums.get_const_data()[0];
}


}  // namespace


namespace block_jacobi {


template <typename ValueType, typename IndexType>
void find_blocks(std::shared_ptr<const GpuExecutor> exec,
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
    GKO_DECLARE_BLOCK_JACOBI_FIND_BLOCKS_KERNEL);


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const GpuExecutor> exec,
              const matrix::Csr<ValueType, IndexType> *system_matrix,
              size_type num_blocks, uint32 max_block_size, size_type stride,
              const Array<IndexType> &block_pointers, Array<ValueType> &blocks)
{
    select_generate(compiled_kernels(),
                    [&](int compiled_block_size) {
                        return max_block_size <= compiled_block_size;
                    },
                    syn::compile_int_list<cuda_config::min_warps_per_block>(),
                    syn::compile_type_list<>(), system_matrix,
                    blocks.get_data(), stride, block_pointers.get_const_data(),
                    num_blocks);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_GENERATE_KERNEL);


template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const GpuExecutor> exec, size_type num_blocks,
           uint32 max_block_size, size_type stride,
           const Array<IndexType> &block_pointers,
           const Array<ValueType> &blocks,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *x)
{
    using Vec = matrix::Dense<ValueType>;
    // TODO: do this efficiently
    auto tmp = x->clone();
    simple_apply(exec, num_blocks, max_block_size, stride, block_pointers,
                 blocks, b, static_cast<Vec *>(tmp.get()));
    x->scale(beta);
    x->add_scaled(alpha, tmp.get());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void simple_apply(std::shared_ptr<const GpuExecutor> exec, size_type num_blocks,
                  uint32 max_block_size, size_type stride,
                  const Array<IndexType> &block_pointers,
                  const Array<ValueType> &blocks,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *x)
{
    // TODO: write a special kernel for multiple RHS
    for (size_type col = 0; col < b->get_size().num_cols; ++col) {
        select_apply(compiled_kernels(),
                     [&](int compiled_block_size) {
                         return max_block_size <= compiled_block_size;
                     },
                     syn::compile_int_list<cuda_config::min_warps_per_block>(),
                     syn::compile_type_list<>(), num_blocks,
                     block_pointers.get_const_data(), blocks.get_const_data(),
                     stride, b->get_const_values() + col, b->get_stride(),
                     x->get_values() + col, x->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_SIMPLE_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const GpuExecutor> exec,
                      size_type num_blocks,
                      const Array<IndexType> &block_pointers,
                      const Array<ValueType> &blocks, size_type block_stride,
                      ValueType *result_values,
                      size_type result_stride) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_CONVERT_TO_DENSE_KERNEL);


}  // namespace block_jacobi


namespace adaptive_block_jacobi {


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const GpuExecutor> exec,
              const matrix::Csr<ValueType, IndexType> *system_matrix,
              size_type num_blocks, uint32 max_block_size, size_type stride,
              Array<precision<ValueType, IndexType>> &block_precisions,
              const Array<IndexType> &block_pointers,
              Array<ValueType> &blocks) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ADAPTIVE_BLOCK_JACOBI_GENERATE_KERNEL);


template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const GpuExecutor> exec, size_type num_blocks,
           uint32 max_block_size, size_type stride,
           const Array<precision<ValueType, IndexType>> &block_precisions,
           const Array<IndexType> &block_pointers,
           const Array<ValueType> &blocks,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta,
           matrix::Dense<ValueType> *x) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ADAPTIVE_BLOCK_JACOBI_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void simple_apply(
    std::shared_ptr<const GpuExecutor> exec, size_type num_blocks,
    uint32 max_block_size, size_type stride,
    const Array<precision<ValueType, IndexType>> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    const matrix::Dense<ValueType> *b,
    matrix::Dense<ValueType> *x) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ADAPTIVE_BLOCK_JACOBI_SIMPLE_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(
    std::shared_ptr<const GpuExecutor> exec, size_type num_blocks,
    const Array<precision<ValueType, IndexType>> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    size_type block_stride, ValueType *result_values,
    size_type result_stride) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ADAPTIVE_BLOCK_JACOBI_CONVERT_TO_DENSE_KERNEL);


}  // namespace adaptive_block_jacobi
}  // namespace gpu
}  // namespace kernels
}  // namespace gko
