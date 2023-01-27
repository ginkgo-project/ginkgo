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

#include "core/preconditioner/batch_jacobi_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>


#include "core/matrix/batch_struct.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/exception.cuh"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/intrinsics.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/matrix/batch_struct.hpp"
#include "cuda/preconditioner/jacobi_common.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_jacobi {

namespace {

constexpr int default_block_size = 128;

#include "common/cuda_hip/components/uninitialized_array.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_block_jacobi.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_scalar_jacobi.hpp.inc"
// Note: Do not change the ordering
#include "common/cuda_hip/preconditioner/batch_jacobi_kernels.hpp.inc"
}  // namespace


template <typename ValueType, typename IndexType>
void batch_jacobi_apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
    const size_type num_blocks, const uint32 max_block_size,
    const gko::preconditioner::batched_blocks_storage_scheme& storage_scheme,
    const ValueType* const blocks_array, const IndexType* const block_ptrs,
    const IndexType* const row_part_of_which_block_info,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z)
{
    const auto a_ub = get_batch_struct(sys_mat);
    batch_jacobi_apply_helper(a_ub, num_blocks, max_block_size, storage_scheme,
                              blocks_array, block_ptrs,
                              row_part_of_which_block_info, r, z);

    GKO_CUDA_LAST_IF_ERROR_THROW;
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_JACOBI_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void batch_jacobi_apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* const sys_mat,
    const size_type num_blocks, const uint32 max_block_size,
    const gko::preconditioner::batched_blocks_storage_scheme& storage_scheme,
    const ValueType* const blocks_array, const IndexType* const block_ptrs,
    const IndexType* const row_part_of_which_block_info,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z)
{
    const auto a_ub = get_batch_struct(sys_mat);
    batch_jacobi_apply_helper(a_ub, num_blocks, max_block_size, storage_scheme,
                              blocks_array, block_ptrs,
                              row_part_of_which_block_info, r, z);
    GKO_CUDA_LAST_IF_ERROR_THROW;
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_JACOBI_ELL_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void extract_common_blocks_pattern(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const first_sys_csr,
    const size_type num_blocks,
    const preconditioner::batched_blocks_storage_scheme& storage_scheme,
    const IndexType* const block_pointers,
    const IndexType* const row_part_of_which_block_info,
    IndexType* const blocks_pattern)
{
    const auto nrows = first_sys_csr->get_size()[0];
    dim3 block(default_block_size);
    dim3 grid(ceildiv(nrows * config::warp_size, default_block_size));

    extract_common_block_pattern_kernel<<<grid, block>>>(
        static_cast<int>(nrows), first_sys_csr->get_const_row_ptrs(),
        first_sys_csr->get_const_col_idxs(), num_blocks, storage_scheme,
        block_pointers, row_part_of_which_block_info, blocks_pattern);

    GKO_CUDA_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_EXTRACT_PATTERN_KERNEL);


namespace {

template <int compiled_max_block_size, typename ValueType, typename IndexType>
void compute_block_jacobi_helper(
    syn::value_list<int, compiled_max_block_size>,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    const size_type num_blocks,
    const preconditioner::batched_blocks_storage_scheme& storage_scheme,
    const IndexType* const block_pointers,
    const IndexType* const blocks_pattern, ValueType* const blocks)
{
    constexpr int subwarp_size =
        gko::kernels::cuda::jacobi::get_larger_power(compiled_max_block_size);
    // TODO: Move get_larger_power to some math namespace (since hip -> code
    // duplication)
    const auto nbatch = sys_csr->get_num_batch_entries();
    const auto nrows = sys_csr->get_size().at(0)[0];
    const auto nnz = sys_csr->get_num_stored_elements() / nbatch;

    dim3 block(default_block_size);
    dim3 grid(ceildiv(num_blocks * nbatch * subwarp_size, default_block_size));

    compute_block_jacobi_kernel<subwarp_size><<<grid, block>>>(
        nbatch, static_cast<int>(nnz),
        as_cuda_type(sys_csr->get_const_values()), num_blocks, storage_scheme,
        block_pointers, blocks_pattern, as_cuda_type(blocks));

    GKO_CUDA_LAST_IF_ERROR_THROW;
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_compute_block_jacobi_helper,
                                    compute_block_jacobi_helper);

}  // anonymous namespace


template <typename ValueType, typename IndexType>
void compute_block_jacobi(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    const uint32 user_given_max_block_size, const size_type num_blocks,
    const preconditioner::batched_blocks_storage_scheme& storage_scheme,
    const IndexType* const block_pointers,
    const IndexType* const blocks_pattern, ValueType* const blocks)
{
    using batch_jacobi_compiled_max_block_sizes =
        gko::kernels::cuda::jacobi::compiled_kernels;

    select_compute_block_jacobi_helper(
        batch_jacobi_compiled_max_block_sizes(),
        [&](int compiled_block_size) {
            return user_given_max_block_size <= compiled_block_size;
        },
        syn::value_list<int>(), syn::type_list<>(), sys_csr, num_blocks,
        storage_scheme, block_pointers, blocks_pattern, blocks);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_KERNEL);

}  // namespace batch_jacobi
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
