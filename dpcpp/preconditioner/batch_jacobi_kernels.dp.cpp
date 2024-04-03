// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/batch_jacobi_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/base/batch_struct.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/batch_struct.hpp"
#include "dpcpp/base/batch_struct.hpp"
#include "dpcpp/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_jacobi {


template <typename IndexType>
void compute_cumulative_block_storage(
    std::shared_ptr<const DefaultExecutor> exec, const size_type num_blocks,
    const IndexType* const block_pointers,
    IndexType* const blocks_cumulative_storage) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_CUMULATIVE_BLOCK_STORAGE);


template <typename IndexType>
void find_row_block_map(
    std::shared_ptr<const DefaultExecutor> exec, const size_type num_blocks,
    const IndexType* const block_pointers,
    IndexType* const row_block_map_info) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_FIND_ROW_BLOCK_MAP);


template <typename ValueType, typename IndexType>
void extract_common_blocks_pattern(
    std::shared_ptr<const DefaultExecutor> exec,
    const gko::matrix::Csr<ValueType, IndexType>* const first_sys_csr,
    const size_type num_blocks, const IndexType* const cumulative_block_storage,
    const IndexType* const block_pointers, const IndexType* const,
    IndexType* const blocks_pattern) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_EXTRACT_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void compute_block_jacobi(
    std::shared_ptr<const DefaultExecutor> exec,
    const batch::matrix::Csr<ValueType, IndexType>* const sys_csr, const uint32,
    const size_type num_blocks, const IndexType* const cumulative_block_storage,
    const IndexType* const block_pointers,
    const IndexType* const blocks_pattern,
    ValueType* const blocks) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_KERNEL);


}  // namespace batch_jacobi
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
