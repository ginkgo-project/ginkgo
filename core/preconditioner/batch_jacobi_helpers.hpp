// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_PRECONDITIONER_BATCH_JACOBI_HELPERS_HPP_
#define GKO_CORE_PRECONDITIONER_BATCH_JACOBI_HELPERS_HPP_


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>


namespace gko {
namespace detail {
namespace batch_jacobi {

/**
 * Returns the offset of the batch with id "batch_id"
 *
 * @param batch_id  the index of the batch entry in the batch
 * @param num_blocks  number of blocks in an individual matrix item
 * @param block_storage_cumulative  the cumulative block storage array
 *
 * @return the offset of the group belonging to block with ID `block_id`
 */
GKO_ATTRIBUTES static size_type get_batch_offset(
    const size_type batch_id, const size_type num_blocks,
    const int32* const block_storage_cumulative) noexcept
{
    return batch_id * block_storage_cumulative[num_blocks];
}

/**
 * Returns the (local) offset of the block with id: "block_id" within its
 * batch entry
 *
 * @param block_id  the id of the block from the perspective of individual
 *                  batch item
 * @param blocks_storage_cumulative  the cumulative block storage array
 *
 * @return the offset of the block with id: `block_id` within its batch
 * entry
 */
GKO_ATTRIBUTES static size_type get_block_offset(
    const size_type block_id,
    const int32* const block_storage_cumulative) noexcept
{
    return block_storage_cumulative[block_id];
}

/**
 * Returns the global offset of the block which belongs to the batch entry
 * with index = batch_id and has local id = "block_id" within its batch
 * entry
 *
 * @param batch_id  the index of the batch entry in the batch
 * @param num_blocks  number of blocks in an individual matrix entry
 * @param block_id  the id of the block from the perspective of individual
 *                  batch entry
 * @param block_storage_cumulative  the cumulative block storage array
 *
 * @return the global offset of the block which belongs to the batch entry
 * with index = "batch_id" and has local id = "block_id" within its batch
 * entry
 */
GKO_ATTRIBUTES static size_type get_global_block_offset(
    const size_type batch_id, const size_type num_blocks,
    const size_type block_id,
    const int32* const block_storage_cumulative) noexcept
{
    return get_batch_offset(batch_id, num_blocks, block_storage_cumulative) +
           get_block_offset(block_id, block_storage_cumulative);
}

/**
 * Returns the stride between the rows of the block.
 *
 * @param block_idx  the id of the block from the perspective of individual
 *                   batch entry
 * @param block_ptrs  the block pointers array
 *
 * @return stride between rows of the block
 */
GKO_ATTRIBUTES static size_type get_stride(
    const int block_idx, const int32* const block_ptrs) noexcept
{
    return block_ptrs[block_idx + 1] - block_ptrs[block_idx];
}


}  // namespace batch_jacobi
}  // namespace detail
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_BATCH_JACOBI_HELPERS_HPP_
