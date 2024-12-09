// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/assembly_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {
namespace assembly {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void count_non_owning_entries(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    comm_index_type local_part, array<comm_index_type>& send_count,
    array<GlobalIndexType>& send_positions,
    array<GlobalIndexType>& original_positions) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE_BASE(
    GKO_DECLARE_COUNT_NON_OWNING_ENTRIES);


}  // namespace assembly
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
