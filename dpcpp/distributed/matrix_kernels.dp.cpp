// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/matrix_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {
namespace distributed_matrix {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void separate_diag_off_diag(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<LocalIndexType>& diag_row_idxs,
    array<LocalIndexType>& diag_col_idxs, array<ValueType>& diag_values,
    array<LocalIndexType>& off_diag_row_idxs,
    array<GlobalIndexType>& off_diag_col_idxs,
    array<ValueType>& off_diag_values) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_SEPARATE_DIAG_OFF_DIAG);


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void separate_diag_off_diag_local_rows(
    std::shared_ptr<const DefaultExecutor> exec,
    const array<LocalIndexType>& row_idxs,
    const array<LocalIndexType>& col_idxs,
    const array<GlobalIndexType>& col_map, const array<ValueType>& values,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<LocalIndexType>& diag_row_idxs,
    array<LocalIndexType>& diag_col_idxs, array<ValueType>& diag_values,
    array<LocalIndexType>& off_diag_row_idxs,
    array<GlobalIndexType>& off_diag_col_idxs,
    array<ValueType>& off_diag_values) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_SEPARATE_DIAG_OFF_DIAG_LOCAL_ROWS);


template <typename LocalIndexType, typename GlobalIndexType>
void compress_columns(std::shared_ptr<const DefaultExecutor> exec,
                      const array<GlobalIndexType>& global_cols,
                      array<LocalIndexType>& compact_cols,
                      array<GlobalIndexType>& distinct_cols)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_COMPRESS_COLUMNS);


}  // namespace distributed_matrix
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
