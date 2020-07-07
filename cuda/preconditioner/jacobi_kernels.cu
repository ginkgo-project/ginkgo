/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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
#include "core/preconditioner/jacobi_utils.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/preconditioner/jacobi_common.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The Jacobi preconditioner namespace.
 * @ref Jacobi
 * @ingroup jacobi
 */
namespace jacobi {
namespace {


// a total of 32 warps (1024 threads)
constexpr int default_num_warps = 32;
// with current architectures, at most 32 warps can be scheduled per SM (and
// current GPUs have at most 84 SMs)
constexpr int default_grid_size = 32 * 32 * 128;


#include "common/preconditioner/jacobi_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
size_type find_natural_blocks(std::shared_ptr<const CudaExecutor> exec,
                              const matrix::Csr<ValueType, IndexType> *mtx,
                              int32 max_block_size,
                              IndexType *__restrict__ block_ptrs)
{
    Array<size_type> nums(exec, 1);

    Array<bool> matching_next_row(exec, mtx->get_size()[0] - 1);

    const dim3 block_size(config::warp_size, 1, 1);
    const dim3 grid_size(
        ceildiv(mtx->get_size()[0] * config::warp_size, block_size.x), 1, 1);
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


void initialize_precisions(std::shared_ptr<const CudaExecutor> exec,
                           const Array<precision_reduction> &source,
                           Array<precision_reduction> &precisions)
{
    const auto block_size = default_num_warps * config::warp_size;
    const auto grid_size = min(
        default_grid_size,
        static_cast<int32>(ceildiv(precisions.get_num_elems(), block_size)));
    duplicate_array<default_num_warps><<<grid_size, block_size>>>(
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


namespace {


template <bool conjugate, int warps_per_block, int max_block_size,
          typename ValueType, typename IndexType>
void transpose_jacobi(
    syn::value_list<int, max_block_size>, size_type num_blocks,
    const precision_reduction *block_precisions,
    const IndexType *block_pointers, const ValueType *blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>
        &storage_scheme,
    ValueType *out_blocks)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    constexpr int blocks_per_warp = config::warp_size / subwarp_size;
    const dim3 grid_size(ceildiv(num_blocks, warps_per_block * blocks_per_warp),
                         1, 1);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    if (block_precisions) {
        adaptive_transpose_jacobi<conjugate, max_block_size, subwarp_size,
                                  warps_per_block>
            <<<grid_size, block_size, 0, 0>>>(
                as_cuda_type(blocks), storage_scheme, block_precisions,
                block_pointers, num_blocks, as_cuda_type(out_blocks));
    } else {
        transpose_jacobi<conjugate, max_block_size, subwarp_size,
                         warps_per_block><<<grid_size, block_size, 0, 0>>>(
            as_cuda_type(blocks), storage_scheme, block_pointers, num_blocks,
            as_cuda_type(out_blocks));
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_transpose_jacobi, transpose_jacobi);


}  // namespace


template <typename ValueType, typename IndexType>
void transpose_jacobi(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
    uint32 max_block_size, const Array<precision_reduction> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>
        &storage_scheme,
    Array<ValueType> &out_blocks)
{
    select_transpose_jacobi(
        compiled_kernels(),
        [&](int compiled_block_size) {
            return max_block_size <= compiled_block_size;
        },
        syn::value_list<int, false, config::min_warps_per_block>(),
        syn::type_list<>(), num_blocks, block_precisions.get_const_data(),
        block_pointers.get_const_data(), blocks.get_const_data(),
        storage_scheme, out_blocks.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose_jacobi(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
    uint32 max_block_size, const Array<precision_reduction> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>
        &storage_scheme,
    Array<ValueType> &out_blocks)
{
    select_transpose_jacobi(
        compiled_kernels(),
        [&](int compiled_block_size) {
            return max_block_size <= compiled_block_size;
        },
        syn::value_list<int, true, config::min_warps_per_block>(),
        syn::type_list<>(), num_blocks, block_precisions.get_const_data(),
        block_pointers.get_const_data(), blocks.get_const_data(),
        storage_scheme, out_blocks.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_CONJ_TRANSPOSE_KERNEL);


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
