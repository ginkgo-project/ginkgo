// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/matrix_kernels.hpp"


#include "core/base/allocator.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace distributed_matrix {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void build_local_nonlocal(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<LocalIndexType>& local_row_idxs,
    array<LocalIndexType>& local_col_idxs, array<ValueType>& local_values,
    array<LocalIndexType>& non_local_row_idxs,
    array<LocalIndexType>& non_local_col_idxs,
    array<ValueType>& non_local_values,
    array<LocalIndexType>& local_gather_idxs,
    array<comm_index_type>& recv_sizes,
    array<GlobalIndexType>& non_local_to_global)
{
    using partition_type =
        experimental::distributed::Partition<LocalIndexType, GlobalIndexType>;
    using global_nonzero = matrix_data_entry<ValueType, GlobalIndexType>;
    auto input_row_idxs = input.get_const_row_idxs();
    auto input_col_idxs = input.get_const_col_idxs();
    auto input_vals = input.get_const_values();
    auto row_part_ids = row_partition->get_part_ids();
    auto col_part_ids = col_partition->get_part_ids();
    auto num_parts = row_partition->get_num_parts();

    auto find_range = [](GlobalIndexType idx, const partition_type* partition,
                         size_type hint) {
        auto range_bounds = partition->get_range_bounds();
        auto num_ranges = partition->get_num_ranges();
        if (range_bounds[hint] <= idx && idx < range_bounds[hint + 1]) {
            return hint;
        } else {
            auto it = std::upper_bound(range_bounds + 1,
                                       range_bounds + num_ranges + 1, idx);
            return static_cast<size_type>(std::distance(range_bounds + 1, it));
        }
    };
    auto map_to_local = [](GlobalIndexType idx, const partition_type* partition,
                           size_type range_id) {
        auto range_bounds = partition->get_range_bounds();
        auto range_starting_indices = partition->get_range_starting_indices();
        return static_cast<LocalIndexType>(idx - range_bounds[range_id]) +
               range_starting_indices[range_id];
    };

    vector<global_nonzero> local_entries(exec);
    vector<global_nonzero> non_local_entries(exec);
    size_type row_range_id = 0;
    size_type col_range_id = 0;
    for (size_type i = 0; i < input.get_num_stored_elements(); ++i) {
        auto global_row = input_row_idxs[i];
        row_range_id = find_range(global_row, row_partition, row_range_id);
        if (row_part_ids[row_range_id] == local_part) {
            auto global_col = input_col_idxs[i];
            col_range_id = find_range(global_col, col_partition, col_range_id);
            if (col_part_ids[col_range_id] == local_part) {
                local_entries.push_back(
                    {map_to_local(global_row, row_partition, row_range_id),
                     map_to_local(global_col, col_partition, col_range_id),
                     input_vals[i]});
            } else {
                non_local_entries.push_back(
                    {map_to_local(global_row, row_partition, row_range_id),
                     global_col, input_vals[i]});
            }
        }
    }

    // create local matrix
    local_row_idxs.resize_and_reset(local_entries.size());
    local_col_idxs.resize_and_reset(local_entries.size());
    local_values.resize_and_reset(local_entries.size());
    for (size_type i = 0; i < local_entries.size(); ++i) {
        const auto& entry = local_entries[i];
        local_row_idxs.get_data()[i] = entry.row;
        local_col_idxs.get_data()[i] = entry.column;
        local_values.get_data()[i] = entry.value;
    }

    // create non-local matrix
    // 1. stable sort global columns according to their part-id and global
    // columns
    auto find_col_part = [&](GlobalIndexType idx) {
        auto range_id = find_range(idx, col_partition, 0);
        return col_part_ids[range_id];
    };
    vector<GlobalIndexType> unique_columns(exec);
    std::transform(non_local_entries.begin(), non_local_entries.end(),
                   std::back_inserter(unique_columns),
                   [](const auto& entry) { return entry.column; });
    std::sort(unique_columns.begin(), unique_columns.end(),
              [&](const auto& a, const auto& b) {
                  auto part_a = find_col_part(a);
                  auto part_b = find_col_part(b);
                  return std::tie(part_a, a) < std::tie(part_b, b);
              });

    // 2. remove duplicate columns, now the new column i has global index
    // unique_columns[i]
    unique_columns.erase(
        std::unique(unique_columns.begin(), unique_columns.end()),
        unique_columns.end());

    // 3. create mapping from unique_columns
    unordered_map<GlobalIndexType, LocalIndexType> non_local_column_map(exec);
    for (size_type i = 0; i < unique_columns.size(); ++i) {
        non_local_column_map[unique_columns[i]] =
            static_cast<LocalIndexType>(i);
    }

    // 3.5 copy unique_columns to array
    non_local_to_global = array<GlobalIndexType>{exec, unique_columns.begin(),
                                                 unique_columns.end()};

    // 4. fill non_local_data
    non_local_row_idxs.resize_and_reset(non_local_entries.size());
    non_local_col_idxs.resize_and_reset(non_local_entries.size());
    non_local_values.resize_and_reset(non_local_entries.size());
    for (size_type i = 0; i < non_local_entries.size(); ++i) {
        const auto& entry = non_local_entries[i];
        non_local_row_idxs.get_data()[i] = entry.row;
        non_local_col_idxs.get_data()[i] = non_local_column_map[entry.column];
        non_local_values.get_data()[i] = entry.value;
    }

    // compute gather idxs and recv_sizes
    local_gather_idxs.resize_and_reset(unique_columns.size());
    std::fill_n(recv_sizes.get_data(), num_parts, 0);
    for (size_type i = 0; i < unique_columns.size(); ++i) {
        col_range_id =
            find_range(unique_columns[i], col_partition, col_range_id);
        local_gather_idxs.get_data()[i] =
            map_to_local(unique_columns[i], col_partition, col_range_id);
        recv_sizes.get_data()[find_col_part(unique_columns[i])]++;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_LOCAL_NONLOCAL);


}  // namespace distributed_matrix
}  // namespace reference
}  // namespace kernels
}  // namespace gko
