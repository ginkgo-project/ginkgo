/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
    const distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<LocalIndexType>& local_row_idxs,
    array<LocalIndexType>& local_col_idxs, array<ValueType>& local_values,
    array<LocalIndexType>& non_local_row_idxs,
    array<LocalIndexType>& non_local_col_idxs,
    array<ValueType>& non_local_values,
    array<LocalIndexType>& local_gather_idxs,
    array<comm_index_type>& recv_offsets,
    array<GlobalIndexType>& non_local_to_global)
{
    using partition_type =
        distributed::Partition<LocalIndexType, GlobalIndexType>;
    using range_index_type = GlobalIndexType;
    using global_nonzero = matrix_data_entry<ValueType, GlobalIndexType>;
    using local_nonzero = matrix_data_entry<ValueType, LocalIndexType>;
    auto input_row_idxs = input.get_const_row_idxs();
    auto input_col_idxs = input.get_const_col_idxs();
    auto input_vals = input.get_const_values();
    auto row_part_ids = row_partition->get_part_ids();
    auto col_part_ids = col_partition->get_part_ids();
    auto num_parts = row_partition->get_num_parts();
    auto recv_offsets_ptr = recv_offsets.get_data();
    size_type row_range_id_hint = 0;
    size_type col_range_id_hint = 0;
    // zero recv_offsets values
    recv_offsets.resize_and_reset(num_parts + 1);
    std::fill_n(recv_offsets_ptr, num_parts + 1, comm_index_type{});

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

    // store non-local columns and their range indices
    map<GlobalIndexType, range_index_type> non_local_cols(exec);
    // store non-local entries with global column idxs
    vector<global_nonzero> global_non_local_entries(exec);
    vector<local_nonzero> local_entries(exec);
    for (size_type i = 0; i < input.get_num_elems(); ++i) {
        const auto global_row = input_row_idxs[i];
        const auto global_col = input_col_idxs[i];
        const auto value = input_vals[i];
        auto row_range_id =
            find_range(global_row, row_partition, row_range_id_hint);
        row_range_id_hint = row_range_id;
        // skip non-local rows
        if (row_part_ids[row_range_id] == local_part) {
            // map to part-local indices
            auto local_row =
                map_to_local(global_row, row_partition, row_range_id);

            auto col_range_id =
                find_range(global_col, col_partition, col_range_id_hint);
            col_range_id_hint = col_range_id;
            if (col_part_ids[col_range_id] == local_part) {
                // store local entry
                auto local_col =
                    map_to_local(global_col, col_partition, col_range_id);
                local_entries.emplace_back(local_row, local_col, value);
            } else {
                // store non-local entry
                auto is_new_col =
                    non_local_cols.emplace(global_col, col_range_id).second;
                // count the number of non-local entries in each part
                if (is_new_col) {
                    recv_offsets_ptr[col_part_ids[col_range_id]]++;
                }
                global_non_local_entries.emplace_back(local_row, global_col,
                                                      value);
            }
        }
    }

    // store local data to output
    local_row_idxs.resize_and_reset(local_entries.size());
    local_col_idxs.resize_and_reset(local_entries.size());
    local_values.resize_and_reset(local_entries.size());
    std::transform(local_entries.begin(), local_entries.end(),
                   detail::make_zip_iterator(local_row_idxs.get_data(),
                                             local_col_idxs.get_data(),
                                             local_values.get_data()),
                   [](const auto& entry) {
                       return std::make_tuple(entry.row, entry.column,
                                              entry.value);
                   });

    // build recv_offsets
    components::prefix_sum(exec, recv_offsets_ptr, num_parts + 1);

    // collect and renumber non-local columns
    const auto num_non_local_cols =
        static_cast<size_type>(recv_offsets_ptr[num_parts]);
    local_gather_idxs.resize_and_reset(num_non_local_cols);
    std::unordered_map<GlobalIndexType, LocalIndexType> global_to_non_local;
    for (const auto& entry : non_local_cols) {
        auto range = entry.second;
        auto part = col_part_ids[range];
        auto idx = recv_offsets_ptr[part];
        local_gather_idxs.get_data()[idx] =
            map_to_local(entry.first, col_partition, range);
        global_to_non_local[entry.first] = idx;
        ++recv_offsets_ptr[part];
    }

    // build local-to-global map for non-local columns
    non_local_to_global.resize_and_reset(num_non_local_cols);
    std::fill_n(non_local_to_global.get_data(),
                non_local_to_global.get_num_elems(),
                invalid_index<GlobalIndexType>());
    for (const auto& key_value : global_to_non_local) {
        const auto global_idx = key_value.first;
        const auto local_idx = key_value.second;
        non_local_to_global.get_data()[local_idx] = global_idx;
    }

    // shift recv_offsets to the back, insert 0 in front again
    LocalIndexType local_prev{};
    for (size_type i = 0; i <= num_parts; i++) {
        recv_offsets_ptr[i] = std::exchange(local_prev, recv_offsets_ptr[i]);
    }

    // map non-local values to local column indices
    non_local_row_idxs.resize_and_reset(global_non_local_entries.size());
    non_local_col_idxs.resize_and_reset(global_non_local_entries.size());
    non_local_values.resize_and_reset(global_non_local_entries.size());
    std::transform(global_non_local_entries.begin(),
                   global_non_local_entries.end(),
                   detail::make_zip_iterator(non_local_row_idxs.get_data(),
                                             non_local_col_idxs.get_data(),
                                             non_local_values.get_data()),
                   [&](const auto& entry) {
                       return std::make_tuple(
                           static_cast<LocalIndexType>(entry.row),
                           global_to_non_local[entry.column], entry.value);
                   });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_LOCAL_NONLOCAL);


}  // namespace distributed_matrix
}  // namespace reference
}  // namespace kernels
}  // namespace gko
