// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/dd_matrix_kernels.hpp"

#include "core/base/allocator.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/base/iterator_factory.hpp"
#include "reference/distributed/partition_helpers.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace distributed_dd_matrix {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void filter_non_owning_idxs(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<GlobalIndexType>& non_owning_row_idxs,
    array<GlobalIndexType>& non_owning_col_idxs)
{
    auto input_col_idxs = input.get_const_col_idxs();
    auto input_row_idxs = input.get_const_row_idxs();
    auto col_part_ids = col_partition->get_part_ids();
    auto row_part_ids = row_partition->get_part_ids();

    vector<GlobalIndexType> non_local_col_idxs(exec);
    vector<GlobalIndexType> non_local_row_idxs(exec);
    size_type col_range_id = 0;
    size_type row_range_id = 0;
    for (size_type i = 0; i < input.get_num_stored_elements(); ++i) {
        auto global_col = input_col_idxs[i];
        auto global_row = input_row_idxs[i];
        col_range_id = find_range(global_col, col_partition, col_range_id);
        row_range_id = find_range(global_row, row_partition, row_range_id);
        if (col_part_ids[col_range_id] != local_part) {
            non_local_col_idxs.push_back(global_col);
        }
        if (row_part_ids[row_range_id] != local_part) {
            non_local_row_idxs.push_back(global_row);
        }
    }

    non_owning_col_idxs.resize_and_reset(non_local_col_idxs.size());
    for (size_type i = 0; i < non_local_col_idxs.size(); i++) {
        non_owning_col_idxs.get_data()[i] = non_local_col_idxs[i];
    }
    non_owning_row_idxs.resize_and_reset(non_local_row_idxs.size());
    for (size_type i = 0; i < non_local_row_idxs.size(); i++) {
        non_owning_row_idxs.get_data()[i] = non_local_row_idxs[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_FILTER_NON_OWNING_IDXS);


}  // namespace distributed_dd_matrix
}  // namespace reference
}  // namespace kernels
}  // namespace gko
