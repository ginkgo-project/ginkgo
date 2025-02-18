// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/vector_kernels.hpp"

#include "core/components/prefix_sum_kernels.hpp"
#include "reference/distributed/partition_helpers.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace distributed_vector {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void build_local(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        partition,
    comm_index_type local_part, matrix::Dense<ValueType>* local_mtx)
{
    auto row_idxs = input.get_const_row_idxs();
    auto col_idxs = input.get_const_col_idxs();
    auto values = input.get_const_values();
    auto range_parts = partition->get_part_ids();

    size_type range_id_hint = 0;
    for (size_type i = 0; i < input.get_num_stored_elements(); ++i) {
        auto range_id = find_range(row_idxs[i], partition, range_id_hint);
        range_id_hint = range_id;
        auto part_id = range_parts[range_id];
        // skip non-local rows
        if (part_id == local_part) {
            local_mtx->at(map_to_local(row_idxs[i], partition, range_id),
                          static_cast<LocalIndexType>(col_idxs[i])) = values[i];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE_BASE(
    GKO_DECLARE_DISTRIBUTED_VECTOR_BUILD_LOCAL);


}  // namespace distributed_vector
}  // namespace reference
}  // namespace kernels
}  // namespace gko
