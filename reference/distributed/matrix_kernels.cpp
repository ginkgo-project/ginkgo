/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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
    const device_matrix_data<ValueType, GlobalIndexType>& non_local_input,
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
    local_row_idxs.resize_and_reset(input.get_num_elems());
    local_col_idxs.resize_and_reset(input.get_num_elems());
    local_values.resize_and_reset(input.get_num_elems());
    for (size_type i = 0; i < input.get_num_elems(); ++i) {
        local_row_idxs.get_data()[i] = input.get_const_row_idxs()[i];
        local_col_idxs.get_data()[i] = input.get_const_col_idxs()[i];
        local_values.get_data()[i] = input.get_const_values()[i];
    }

    // 3. create mapping from unique_columns
    unordered_map<GlobalIndexType, LocalIndexType> non_local_column_map(exec);
    for (size_type i = 0; i < non_local_input.get_num_elems(); ++i) {
        non_local_column_map[non_local_input.get_const_col_idxs()[i]] =
            static_cast<LocalIndexType>(i);
    }

    // 3.5 copy unique_columns to array
    non_local_to_global.resize_and_reset(non_local_input.get_num_elems());
    for (size_type i = 0; i < non_local_input.get_num_elems(); ++i) {
        non_local_to_global.get_data()[i] =
            non_local_input.get_const_col_idxs()[i];
    }

    // 4. fill non_local_data
    non_local_row_idxs.resize_and_reset(non_local_input.get_num_elems());
    non_local_col_idxs.resize_and_reset(non_local_input.get_num_elems());
    non_local_values.resize_and_reset(non_local_input.get_num_elems());
    for (size_type i = 0; i < non_local_input.get_num_elems(); ++i) {
        non_local_row_idxs.get_data()[i] =
            non_local_input.get_const_row_idxs()[i];
        non_local_col_idxs.get_data()[i] =
            non_local_column_map[non_local_input.get_const_col_idxs()[i]];
        non_local_values.get_data()[i] = non_local_input.get_const_values()[i];
    }

    // compute gather idxs and recv_sizes
    local_gather_idxs.resize_and_reset(non_local_input.get_num_elems());
    std::fill_n(recv_sizes.get_data(), num_parts, 0);

    auto map_to_local = [](GlobalIndexType idx, const partition_type* partition,
                           size_type range_id) {
        auto range_bounds = partition->get_range_bounds();
        auto range_starting_indices = partition->get_range_starting_indices();
        return static_cast<LocalIndexType>(idx - range_bounds[range_id]) +
               range_starting_indices[range_id];
    };

    auto find_col_part = [&](GlobalIndexType idx) {
        auto range_id = find_range(idx, col_partition, 0);
        return col_part_ids[range_id];
    };
    size_type col_range_id = 0;
    for (size_type i = 0; i < non_local_input.get_num_elems(); ++i) {
        col_range_id = find_range(non_local_input.get_const_col_idxs()[i],
                                  col_partition, col_range_id);
        local_gather_idxs.get_data()[i] =
            map_to_local(non_local_input.get_const_col_idxs()[i], col_partition,
                         col_range_id);
        recv_sizes.get_data()[find_col_part(
            non_local_input.get_const_col_idxs()[i])]++;
    }
}

template <typename LocalIndexType, typename GlobalIndexType>
void build_local_nonlocal_scatter_pattern(
    std::shared_ptr<const DefaultExecutor> exec,
    const array<GlobalIndexType>& row_indices,
    const array<GlobalIndexType>& col_indices,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<LocalIndexType>& local_scatter,
    array<LocalIndexType>& non_local_scatter)
{
    using partition_type =
        experimental::distributed::Partition<LocalIndexType, GlobalIndexType>;
    auto input_row_idxs = row_indices.get_const_data();
    auto input_col_idxs = col_indices.get_const_data();
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

    vector<LocalIndexType> local_entries(exec);
    vector<LocalIndexType> non_local_entries(exec);
    size_type row_range_id = 0;
    size_type col_range_id = 0;
    for (size_type i = 0; i < row_indices.get_num_elems(); ++i) {
        auto global_row = input_row_idxs[i];
        row_range_id = find_range(global_row, row_partition, row_range_id);
        if (row_part_ids[row_range_id] == local_part) {
            auto global_col = input_col_idxs[i];
            col_range_id = find_range(global_col, col_partition, col_range_id);
            if (col_part_ids[col_range_id] == local_part) {
                local_entries.push_back(i);
            } else {
                non_local_entries.push_back(i);
            }
        }
    }

    // create local matrix
    local_scatter.resize_and_reset(local_entries.size());
    for (size_type i = 0; i < local_entries.size(); ++i) {
        local_scatter.get_data()[i] = local_entries[i];
    }

    // 4. fill non_local_data
    non_local_scatter.resize_and_reset(non_local_entries.size());
    for (size_type i = 0; i < non_local_entries.size(); ++i) {
        non_local_scatter.get_data()[i] = non_local_entries[i];
    }
}


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
    for (size_type i = 0; i < input.get_num_elems(); ++i) {
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

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_LOCAL_NONLOCAL2);

GKO_INSTANTIATE_FOR_EACH_LOCAL_AND_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_LOCAL_NON_LOCAL_SCATTER_PATTERN);

}  // namespace distributed_matrix
}  // namespace reference
}  // namespace kernels
}  // namespace gko
