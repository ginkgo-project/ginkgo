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
#include "gpu/components/uninitialized_array.hpp"
#include "gpu/components/warp_blas.cuh"


namespace gko {
namespace kernels {
namespace gpu {
namespace kernel {
namespace {


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void __launch_bounds__(warps_per_block *warp_size)
    generate(size_type num_rows, const IndexType *__restrict__ row_ptrs,
             const IndexType *__restrict__ col_idxs,
             const ValueType *__restrict__ values,
             ValueType *__restrict__ block_data, size_type padding,
             const IndexType *__restrict__ block_ptrs, size_type num_blocks)
{
    constexpr int blocks_per_warp = warp_size / subwarp_size;
    const auto bid =
        static_cast<size_type>(blockIdx.x) * warps_per_block * blocks_per_warp +
        threadIdx.z * blocks_per_warp + threadIdx.y;
    ValueType row[max_block_size];
    __shared__ UninitializedArray<ValueType, max_block_size * warps_per_block>
        workspace;

    device::csr::extract_transposed_diag_blocks<max_block_size, subwarp_size,
                                                warps_per_block>(
        row_ptrs, col_idxs, values, block_ptrs, num_blocks, row, 1,
        workspace + threadIdx.z * max_block_size);
    if (bid < num_blocks) {
        const auto block_size = block_ptrs[bid + 1] - block_ptrs[bid];
        auto perm = threadIdx.x;
        auto iperm = threadIdx.x;
        warp::invert_block<max_block_size, subwarp_size>(block_size, row, perm,
                                                         iperm);
        warp::copy_matrix<max_block_size, subwarp_size>(
            block_size, row, 1, perm, iperm,
            block_data + (block_ptrs[bid] * padding), padding);
    }
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void __launch_bounds__(warps_per_block *warp_size)
    apply(const ValueType *__restrict__ blocks, int32 padding,
          const IndexType *__restrict__ block_ptrs, size_type num_blocks,
          const ValueType *__restrict__ b, int32 b_padding,
          ValueType *__restrict__ x, int32 x_padding)
{
    constexpr int blocks_per_warp = warp_size / subwarp_size;
    const auto bid =
        static_cast<size_type>(blockIdx.x) * warps_per_block * blocks_per_warp +
        threadIdx.z * blocks_per_warp + threadIdx.y;
    if (bid >= num_blocks) {
        return;
    }
    const auto block_size = block_ptrs[bid + 1] - block_ptrs[bid];
    ValueType v = zero<ValueType>();
    if (threadIdx.x < block_size) {
        v = b[(block_ptrs[bid] + threadIdx.x) * b_padding];
    }
    warp::multiply_transposed_vec<max_block_size, subwarp_size>(
        block_size, v, blocks + block_ptrs[bid] * padding + threadIdx.x,
        padding, x + block_ptrs[bid] * x_padding, x_padding);
}


}  // namespace
}  // namespace kernel


namespace {


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

    kernel::generate<max_block_size, subwarp_size, warps_per_block>
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

    kernel::apply<max_block_size, subwarp_size, warps_per_block>
        <<<grid_size, block_size, 0, 0>>>(
            as_cuda_type(blocks), block_padding, block_pointers, num_blocks,
            as_cuda_type(b), b_padding, as_cuda_type(x), x_padding);
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_apply, apply);


using compiled_kernels = syn::compile_int_list<1, 13, 16, 32>;


}  // namespace


namespace block_jacobi {


template <typename ValueType, typename IndexType>
void find_blocks(std::shared_ptr<const GpuExecutor> exec,
                 const matrix::Csr<ValueType, IndexType> *system_matrix,
                 uint32 max_block_size, size_type &num_blocks,
                 Array<IndexType> &block_pointers) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_FIND_BLOCKS_KERNEL);


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
