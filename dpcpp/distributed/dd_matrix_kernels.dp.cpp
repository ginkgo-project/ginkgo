// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/dd_matrix_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {
namespace distributed_dd_matrix {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void filter_non_owning_idxs(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<GlobalIndexType>& non_local_row_idxs,
    array<GlobalIndexType>& non_local_col_idxs) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_FILTER_NON_OWNING_IDXS);


}  // namespace distributed_dd_matrix
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
