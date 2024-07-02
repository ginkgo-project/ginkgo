// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/batch_jacobi_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>

#include "common/cuda_hip/components/intrinsics.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "core/base/batch_struct.hpp"
#include "core/base/utils.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/preconditioner/batch_jacobi_helpers.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/batch_struct.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/matrix/batch_struct.hpp"
#include "preconditioner/jacobi_common.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_jacobi {


namespace {


constexpr int default_block_size = 128;

using batch_jacobi_cuda_compiled_max_block_sizes =
    gko::kernels::cuda::jacobi::compiled_kernels;

#include "common/cuda_hip/preconditioner/batch_jacobi_kernels.hpp.inc"


}  // namespace


template <typename IndexType>
void compute_cumulative_block_storage(
    std::shared_ptr<const DefaultExecutor> exec, const size_type num_blocks,
    const IndexType* const block_pointers,
    IndexType* const blocks_cumulative_offsets)
{
    dim3 block(default_block_size);
    dim3 grid(ceildiv(num_blocks, default_block_size));

    compute_block_storage_kernel<<<grid, block, 0, exec->get_stream()>>>(
        num_blocks, block_pointers, blocks_cumulative_offsets);

    components::prefix_sum_nonnegative(exec, blocks_cumulative_offsets,
                                       num_blocks + 1);
}

GKO_INSTANTIATE_FOR_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_CUMULATIVE_BLOCK_STORAGE);


template <typename IndexType>
void find_row_block_map(std::shared_ptr<const DefaultExecutor> exec,
                        const size_type num_blocks,
                        const IndexType* const block_pointers,
                        IndexType* const map_block_to_row)
{
    dim3 block(default_block_size);
    dim3 grid(ceildiv(num_blocks, default_block_size));
    find_row_block_map_kernel<<<grid, block, 0, exec->get_stream()>>>(
        num_blocks, block_pointers, map_block_to_row);
}

GKO_INSTANTIATE_FOR_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_FIND_ROW_BLOCK_MAP);


template <typename ValueType, typename IndexType>
void extract_common_blocks_pattern(
    std::shared_ptr<const DefaultExecutor> exec,
    const gko::matrix::Csr<ValueType, IndexType>* const first_sys_csr,
    const size_type num_blocks, const IndexType* const cumulative_block_storage,
    const IndexType* const block_pointers,
    const IndexType* const map_block_to_row, IndexType* const blocks_pattern)
{
    const auto nrows = first_sys_csr->get_size()[0];
    dim3 block(default_block_size);
    dim3 grid(ceildiv(nrows * config::warp_size, default_block_size));

    extract_common_block_pattern_kernel<<<grid, block, 0, exec->get_stream()>>>(
        static_cast<int>(nrows), first_sys_csr->get_const_row_ptrs(),
        first_sys_csr->get_const_col_idxs(), num_blocks,
        cumulative_block_storage, block_pointers, map_block_to_row,
        blocks_pattern);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_EXTRACT_PATTERN_KERNEL);


namespace {

template <int compiled_max_block_size, typename ValueType, typename IndexType>
void compute_block_jacobi_helper(
    syn::value_list<int, compiled_max_block_size>,
    std::shared_ptr<const DefaultExecutor> exec,
    const batch::matrix::Csr<ValueType, IndexType>* const sys_csr,
    const size_type num_blocks, const IndexType* const cumulative_block_storage,
    const IndexType* const block_pointers,
    const IndexType* const blocks_pattern, ValueType* const blocks)
{
    constexpr int subwarp_size =
        gko::kernels::cuda::jacobi::get_larger_power(compiled_max_block_size);

    const auto nbatch = sys_csr->get_num_batch_items();
    const auto nrows = sys_csr->get_common_size()[0];
    const auto nnz = sys_csr->get_num_stored_elements() / nbatch;

    dim3 block(default_block_size);
    dim3 grid(ceildiv(num_blocks * nbatch * subwarp_size, default_block_size));

    compute_block_jacobi_kernel<subwarp_size>
        <<<grid, block, 0, exec->get_stream()>>>(
            nbatch, static_cast<int>(nnz),
            as_cuda_type(sys_csr->get_const_values()), num_blocks,
            cumulative_block_storage, block_pointers, blocks_pattern,
            as_cuda_type(blocks));
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_compute_block_jacobi_helper,
                                    compute_block_jacobi_helper);

}  // anonymous namespace


template <typename ValueType, typename IndexType>
void compute_block_jacobi(
    std::shared_ptr<const DefaultExecutor> exec,
    const batch::matrix::Csr<ValueType, IndexType>* const sys_csr,
    const uint32 max_block_size, const size_type num_blocks,
    const IndexType* const cumulative_block_storage,
    const IndexType* const block_pointers,
    const IndexType* const blocks_pattern, ValueType* const blocks)
{
    select_compute_block_jacobi_helper(
        batch_jacobi_cuda_compiled_max_block_sizes(),
        [&](int compiled_block_size) {
            return max_block_size <= compiled_block_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, sys_csr, num_blocks,
        cumulative_block_storage, block_pointers, blocks_pattern, blocks);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_KERNEL);


}  // namespace batch_jacobi
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
