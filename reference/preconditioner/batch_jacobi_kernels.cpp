// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/batch_jacobi_kernels.hpp"


#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "reference/base/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"
#include "reference/preconditioner/batch_block_jacobi.hpp"
#include "reference/preconditioner/batch_scalar_jacobi.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace batch_jacobi {


namespace {


template <typename BatchMatrixType, typename PrecType, typename ValueType>
void apply_jacobi(
    const BatchMatrixType& sys_mat_batch, PrecType& prec,
    const gko::batch::multi_vector::uniform_batch<const ValueType>& rub,
    const gko::batch::multi_vector::uniform_batch<ValueType>& zub)
{
    for (size_type batch_id = 0; batch_id < sys_mat_batch.num_batch_items;
         batch_id++) {
        const auto sys_mat_entry =
            gko::batch::matrix::extract_batch_item(sys_mat_batch, batch_id);
        const auto r_b = gko::batch::extract_batch_item(rub, batch_id);
        const auto z_b = gko::batch::extract_batch_item(zub, batch_id);

        const auto work_arr_size = PrecType::dynamic_work_size(
            sys_mat_batch.num_rows, sys_mat_batch.get_single_item_num_nnz());
        std::vector<ValueType> work(work_arr_size);

        prec.generate(batch_id, sys_mat_entry, work.data());
        prec.apply(r_b, z_b);
    }
}


// Note: Do not change the ordering
#include "reference/preconditioner/batch_jacobi_kernels.hpp.inc"


}  // unnamed namespace


template <typename ValueType, typename IndexType>
void batch_jacobi_apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const batch::matrix::Csr<ValueType, IndexType>* const sys_mat,
    const size_type num_blocks, const uint32 max_block_size,
    const IndexType* const cumulative_block_storage,
    const ValueType* const blocks_array, const IndexType* const block_ptrs,
    const IndexType* const row_block_map_info,
    const batch::MultiVector<ValueType>* const r,
    batch::MultiVector<ValueType>* const z)
{
    const auto sys_mat_batch = host::get_batch_struct(sys_mat);
    batch_jacobi_apply_helper(sys_mat_batch, num_blocks, max_block_size,
                              cumulative_block_storage, blocks_array,
                              block_ptrs, row_block_map_info, r, z);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(
    GKO_DECLARE_BATCH_JACOBI_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void batch_jacobi_apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const batch::matrix::Ell<ValueType, IndexType>* const sys_mat,
    const size_type num_blocks, const uint32 max_block_size,
    const IndexType* const cumulative_block_storage,
    const ValueType* const blocks_array, const IndexType* const block_ptrs,
    const IndexType* const row_block_map_info,
    const batch::MultiVector<ValueType>* const r,
    batch::MultiVector<ValueType>* const z)
{
    const auto sys_mat_batch = host::get_batch_struct(sys_mat);
    batch_jacobi_apply_helper(sys_mat_batch, num_blocks, max_block_size,
                              cumulative_block_storage, blocks_array,
                              block_ptrs, row_block_map_info, r, z);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(
    GKO_DECLARE_BATCH_JACOBI_ELL_APPLY_KERNEL);


template <typename IndexType>
void compute_cumulative_block_storage(
    std::shared_ptr<const DefaultExecutor> exec, const size_type num_blocks,
    const IndexType* const block_pointers,
    IndexType* const blocks_cumulative_storage)
{
    blocks_cumulative_storage[0] = 0;
    for (int i = 0; i < num_blocks; i++) {
        const auto bsize = block_pointers[i + 1] - block_pointers[i];
        blocks_cumulative_storage[i + 1] =
            blocks_cumulative_storage[i] + bsize * bsize;
    }
}

GKO_INSTANTIATE_FOR_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_CUMULATIVE_BLOCK_STORAGE);


template <typename IndexType>
void find_row_block_map(std::shared_ptr<const DefaultExecutor> exec,
                        const size_type num_blocks,
                        const IndexType* const block_pointers,
                        IndexType* const row_block_map_info)
{
    for (size_type block_idx = 0; block_idx < num_blocks; block_idx++) {
        for (IndexType i = block_pointers[block_idx];
             i < block_pointers[block_idx + 1]; i++) {
            row_block_map_info[i] = block_idx;
        }
    }
}

GKO_INSTANTIATE_FOR_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_FIND_ROW_BLOCK_MAP);


template <typename ValueType, typename IndexType>
void extract_common_blocks_pattern(
    std::shared_ptr<const DefaultExecutor> exec,
    const gko::matrix::Csr<ValueType, IndexType>* const first_sys_csr,
    const size_type num_blocks, const IndexType* const cumulative_block_storage,
    const IndexType* const block_pointers, const IndexType* const,
    IndexType* const blocks_pattern)
{
    for (size_type k = 0; k < num_blocks; k++) {
        extract_block_pattern_impl(k, first_sys_csr, cumulative_block_storage,
                                   block_pointers, blocks_pattern);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_EXTRACT_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void compute_block_jacobi(
    std::shared_ptr<const DefaultExecutor> exec,
    const batch::matrix::Csr<ValueType, IndexType>* const sys_csr, const uint32,
    const size_type num_blocks, const IndexType* const cumulative_block_storage,
    const IndexType* const block_pointers,
    const IndexType* const blocks_pattern, ValueType* const blocks)
{
    const auto nbatch = sys_csr->get_num_batch_items();
    const auto A_batch = host::get_batch_struct(sys_csr);

    for (size_type batch_idx = 0; batch_idx < nbatch; batch_idx++) {
        for (size_type k = 0; k < num_blocks; k++) {
            const auto A_entry =
                gko::batch::matrix::extract_batch_item(A_batch, batch_idx);
            compute_block_jacobi_impl(batch_idx, k, A_entry, num_blocks,
                                      cumulative_block_storage, block_pointers,
                                      blocks_pattern, blocks);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_KERNEL);


}  // namespace batch_jacobi
}  // namespace reference
}  // namespace kernels
}  // namespace gko
