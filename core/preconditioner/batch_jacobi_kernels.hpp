// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_PRECONDITIONER_BATCH_JACOBI_KERNELS_HPP_
#define GKO_CORE_PRECONDITIONER_BATCH_JACOBI_KERNELS_HPP_


#include <ginkgo/core/preconditioner/batch_jacobi.hpp>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_BATCH_BLOCK_JACOBI_FIND_ROW_IS_PART_OF_WHICH_BLOCK( \
    IndexType)                                                          \
    void find_row_is_part_of_which_block(                               \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const size_type num_blocks, const IndexType* block_pointers,    \
        IndexType* row_part_of_which_block_info)

#define GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_CUMULATIVE_BLOCK_STORAGE( \
    IndexType)                                                           \
    void compute_cumulative_block_storage(                               \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const size_type num_blocks, const IndexType* block_pointers,     \
        IndexType* blocks_cumulative_storage)

#define GKO_DECLARE_BATCH_BLOCK_JACOBI_EXTRACT_PATTERN_KERNEL(ValueType,   \
                                                              IndexType)   \
    void extract_common_blocks_pattern(                                    \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const matrix::Csr<ValueType, IndexType>* first_sys_csr,            \
        const size_type num_blocks,                                        \
        const batch::preconditioner::batched_jacobi_blocks_storage_scheme< \
            IndexType>& storage_scheme,                                    \
        const IndexType* cumulative_block_storage,                         \
        const IndexType* block_pointers,                                   \
        const IndexType* row_part_of_which_block_info,                     \
        IndexType* blocks_pattern)


#define GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_KERNEL(ValueType, IndexType) \
    void compute_block_jacobi(                                              \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const batch::matrix::Csr<ValueType, IndexType>* sys_csr,            \
        const uint32 max_block_size, const size_type num_blocks,            \
        const batch::preconditioner::batched_jacobi_blocks_storage_scheme<  \
            IndexType>& storage_scheme,                                     \
        const IndexType* cumulative_block_storage,                          \
        const IndexType* block_pointers, const IndexType* blocks_pattern,   \
        ValueType* blocks)

/**
 * @fn batch_jacobi_apply
 *
 * This kernel builds a Jacobi preconditioner for each matrix in
 * the input batch of matrices and applies them to the corresponding vectors
 * in the input vector batches.
 *
 * These functions are mostly meant only for experimentation and testing.
 *
 */
#define GKO_DECLARE_BATCH_JACOBI_APPLY_KERNEL(ValueType, IndexType)          \
    void batch_jacobi_apply(                                                 \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        const batch::matrix::Csr<ValueType, IndexType>* sys_mat,             \
        const size_type num_blocks, const uint32 max_block_size,             \
        const gko::batch::preconditioner::                                   \
            batched_jacobi_blocks_storage_scheme<IndexType>& storage_scheme, \
        const IndexType* cumulative_block_storage,                           \
        const ValueType* blocks_array, const IndexType* block_ptrs,          \
        const IndexType* row_part_of_which_block_info,                       \
        const batch::MultiVector<ValueType>* r,                              \
        batch::MultiVector<ValueType>* z)

#define GKO_DECLARE_BATCH_JACOBI_ELL_APPLY_KERNEL(ValueType, IndexType)      \
    void batch_jacobi_apply(                                                 \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        const batch::matrix::Ell<ValueType, IndexType>* sys_mat,             \
        const size_type num_blocks, const uint32 max_block_size,             \
        const gko::batch::preconditioner::                                   \
            batched_jacobi_blocks_storage_scheme<IndexType>& storage_scheme, \
        const IndexType* cumulative_block_storage,                           \
        const ValueType* blocks_array, const IndexType* block_ptrs,          \
        const IndexType* row_part_of_which_block_info,                       \
        const batch::MultiVector<ValueType>* r,                              \
        batch::MultiVector<ValueType>* z)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                           \
    template <typename IndexType>                                              \
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_CUMULATIVE_BLOCK_STORAGE(           \
        IndexType);                                                            \
    template <typename IndexType>                                              \
    GKO_DECLARE_BATCH_BLOCK_JACOBI_FIND_ROW_IS_PART_OF_WHICH_BLOCK(IndexType); \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_BATCH_BLOCK_JACOBI_EXTRACT_PATTERN_KERNEL(ValueType,           \
                                                          IndexType);          \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_BATCH_JACOBI_ELL_APPLY_KERNEL(ValueType, IndexType);           \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_BATCH_JACOBI_APPLY_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(batch_jacobi,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_BATCH_JACOBI_KERNELS_HPP_