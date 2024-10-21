// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/batch_jacobi_kernels.hpp"

#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "reference/base/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"
#include "reference/preconditioner/batch_block_jacobi.hpp"
#include "reference/preconditioner/batch_jacobi_kernels.hpp"
#include "reference/preconditioner/batch_scalar_jacobi.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace batch_jacobi {


template <typename IndexType>
void compute_cumulative_block_storage(
    std::shared_ptr<const DefaultExecutor> exec, const size_type num_blocks,
    const IndexType* block_pointers, IndexType* blocks_cumulative_offsets)
{
    blocks_cumulative_offsets[0] = 0;
    for (int i = 0; i < num_blocks; i++) {
        const auto bsize = block_pointers[i + 1] - block_pointers[i];
        blocks_cumulative_offsets[i + 1] =
            blocks_cumulative_offsets[i] + bsize * bsize;
    }
}

GKO_INSTANTIATE_FOR_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_CUMULATIVE_BLOCK_STORAGE);


template <typename IndexType>
void find_row_block_map(std::shared_ptr<const DefaultExecutor> exec,
                        const size_type num_blocks,
                        const IndexType* block_pointers,
                        IndexType* map_block_to_row)
{
    for (size_type block_idx = 0; block_idx < num_blocks; block_idx++) {
        for (IndexType i = block_pointers[block_idx];
             i < block_pointers[block_idx + 1]; i++) {
            map_block_to_row[i] = block_idx;
        }
    }
}

GKO_INSTANTIATE_FOR_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_FIND_ROW_BLOCK_MAP);


template <typename ValueType, typename IndexType>
void extract_common_blocks_pattern(
    std::shared_ptr<const DefaultExecutor> exec,
    const gko::matrix::Csr<ValueType, IndexType>* first_sys_csr,
    const size_type num_blocks, const IndexType* cumulative_block_storage,
    const IndexType* block_pointers, const IndexType*,
    IndexType* blocks_pattern)
{
    for (size_type k = 0; k < num_blocks; k++) {
        batch_single_kernels::extract_block_pattern_impl(
            k, first_sys_csr, cumulative_block_storage, block_pointers,
            blocks_pattern);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_EXTRACT_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void compute_block_jacobi(
    std::shared_ptr<const DefaultExecutor> exec,
    const batch::matrix::Csr<ValueType, IndexType>* sys_csr, const uint32,
    const size_type num_blocks, const IndexType* cumulative_block_storage,
    const IndexType* block_pointers, const IndexType* blocks_pattern,
    ValueType* blocks)
{
    const auto nbatch = sys_csr->get_num_batch_items();
    const auto A_batch = host::get_batch_struct(sys_csr);

    for (size_type batch_idx = 0; batch_idx < nbatch; batch_idx++) {
        for (size_type k = 0; k < num_blocks; k++) {
            const auto A_entry =
                gko::batch::extract_batch_item(A_batch, batch_idx);
            batch_single_kernels::compute_block_jacobi_impl(
                batch_idx, k, A_entry, num_blocks, cumulative_block_storage,
                block_pointers, blocks_pattern, blocks);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_KERNEL);


}  // namespace batch_jacobi
}  // namespace reference
}  // namespace kernels
}  // namespace gko
