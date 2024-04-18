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
#include "core/preconditioner/batch_jacobi_helpers.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_BATCH_BLOCK_JACOBI_FIND_ROW_BLOCK_MAP(IndexType)     \
    void find_row_block_map(std::shared_ptr<const DefaultExecutor> exec, \
                            const size_type num_blocks,                  \
                            const IndexType* block_pointers,             \
                            IndexType* row_block_map_info)

#define GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_CUMULATIVE_BLOCK_STORAGE( \
    IndexType)                                                           \
    void compute_cumulative_block_storage(                               \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const size_type num_blocks, const IndexType* block_pointers,     \
        IndexType* blocks_cumulative_offsets)

#define GKO_DECLARE_BATCH_BLOCK_JACOBI_EXTRACT_PATTERN_KERNEL(ValueType,       \
                                                              IndexType)       \
    void extract_common_blocks_pattern(                                        \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const matrix::Csr<ValueType, IndexType>* first_sys_csr,                \
        const size_type num_blocks, const IndexType* cumulative_block_storage, \
        const IndexType* block_pointers, const IndexType* row_block_map_info,  \
        IndexType* blocks_pattern)


#define GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_KERNEL(ValueType, IndexType) \
    void compute_block_jacobi(                                              \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const batch::matrix::Csr<ValueType, IndexType>* sys_csr,            \
        const uint32 max_block_size, const size_type num_blocks,            \
        const IndexType* cumulative_block_storage,                          \
        const IndexType* block_pointers, const IndexType* blocks_pattern,   \
        ValueType* blocks)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                  \
    template <typename IndexType>                                     \
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_CUMULATIVE_BLOCK_STORAGE(  \
        IndexType);                                                   \
    template <typename IndexType>                                     \
    GKO_DECLARE_BATCH_BLOCK_JACOBI_FIND_ROW_BLOCK_MAP(IndexType);     \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_BATCH_BLOCK_JACOBI_EXTRACT_PATTERN_KERNEL(ValueType,  \
                                                          IndexType); \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(batch_jacobi,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_BATCH_JACOBI_KERNELS_HPP_
