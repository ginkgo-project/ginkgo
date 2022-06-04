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


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/base/allocator.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace distributed_matrix {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void build_diag_offdiag(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<LocalIndexType>& diag_row_idxs,
    array<LocalIndexType>& diag_col_idxs, array<ValueType>& diag_values,
    array<LocalIndexType>& offdiag_row_idxs,
    array<LocalIndexType>& offdiag_col_idxs, array<ValueType>& offdiag_values,
    array<LocalIndexType>& local_gather_idxs, comm_index_type* recv_sizes,
    array<GlobalIndexType>& local_to_global_ghost)
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
    size_type row_range_id_hint = 0;
    size_type col_range_id_hint = 0;
    // zero recv_sizes values
    std::fill_n(recv_sizes, num_parts, comm_index_type{});

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

    // store offdiagonal columns and their range indices
    map<GlobalIndexType, range_index_type> offdiag_cols(exec);
    // store offdiagonal entries with global column idxs
    vector<global_nonzero> global_offdiag_entries(exec);
    vector<local_nonzero> diag_entries(exec);

    auto num_threads = static_cast<size_type>(omp_get_max_threads());
    auto num_input = input.get_num_elems();
    auto size_per_thread = (num_input + num_threads - 1) / num_threads;
    std::vector<size_type> diag_entry_offsets(num_threads, 0);
    std::vector<size_type> offdiag_entry_offsets(num_threads, 0);

#pragma omp parallel
    {
        std::unordered_map<GlobalIndexType, range_index_type>
            thread_offdiag_cols;
        std::vector<global_nonzero> thread_offdiag_entries;
        std::vector<local_nonzero> thread_diag_entries;
        std::vector<comm_index_type> thread_recv_sizes;
        auto thread_id = omp_get_thread_num();
        auto thread_begin = thread_id * size_per_thread;
        auto thread_end = std::min(thread_begin + size_per_thread, num_input);
        // separate diagonal and off-diagonal entries for our input chunk
        for (auto i = thread_begin; i < thread_end; ++i) {
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
                    // store diagonal entry
                    auto local_col =
                        map_to_local(global_col, col_partition, col_range_id);
                    thread_diag_entries.emplace_back(local_row, local_col,
                                                     value);
                } else {
                    thread_offdiag_cols.emplace(global_col, col_range_id);
                    thread_offdiag_entries.emplace_back(local_row, global_col,
                                                        value);
                }
            }
        }
        diag_entry_offsets[thread_id] = thread_diag_entries.size();
        offdiag_entry_offsets[thread_id] = thread_offdiag_entries.size();

#pragma omp critical
        {
            // collect global off-diagonal columns
            offdiag_cols.insert(thread_offdiag_cols.begin(),
                                thread_offdiag_cols.end());
        }
#pragma omp barrier
#pragma omp single
        {
            // assign output ranges to the individual threads
            size_type diag{};
            size_type offdiag{};
            for (size_type thread = 0; thread < num_threads; ++thread) {
                auto size_diag = diag_entry_offsets[thread];
                auto size_offdiag = offdiag_entry_offsets[thread];
                diag_entry_offsets[thread] = diag;
                offdiag_entry_offsets[thread] = offdiag;
                diag += size_diag;
                offdiag += size_offdiag;
            }
            diag_entries.resize(diag);
            global_offdiag_entries.resize(offdiag);
        }
        // write back the local data to the output ranges
        auto diag = diag_entry_offsets[thread_id];
        auto offdiag = offdiag_entry_offsets[thread_id];
        for (const auto& entry : thread_diag_entries) {
            diag_entries[diag] = entry;
            diag++;
        }
        for (const auto& entry : thread_offdiag_entries) {
            global_offdiag_entries[offdiag] = entry;
            offdiag++;
        }
    }
    // store diagonal data to output
    diag_row_idxs.resize_and_reset(diag_entries.size());
    diag_col_idxs.resize_and_reset(diag_entries.size());
    diag_values.resize_and_reset(diag_entries.size());
#pragma omp parallel for
    for (size_type i = 0; i < diag_entries.size(); ++i) {
        const auto& entry = diag_entries[i];
        diag_row_idxs.get_data()[i] = entry.row;
        diag_col_idxs.get_data()[i] = entry.column;
        diag_values.get_data()[i] = entry.value;
    }

    // count off-diagonal columns per part
    for (const auto& entry : offdiag_cols) {
        auto col_range_id = entry.second;
        recv_sizes[col_part_ids[col_range_id]]++;
    }
    const auto num_ghost_elems =
        std::accumulate(recv_sizes, recv_sizes + num_parts, size_type{});
    components::prefix_sum(exec, recv_sizes, num_parts);

    // collect and renumber offdiagonal columns
    local_gather_idxs.resize_and_reset(num_ghost_elems);
    std::unordered_map<GlobalIndexType, LocalIndexType> offdiag_global_to_local;
    for (const auto& entry : offdiag_cols) {
        auto range = entry.second;
        auto part = col_part_ids[range];
        auto idx = recv_sizes[part];
        local_gather_idxs.get_data()[idx] =
            map_to_local(entry.first, col_partition, entry.second);
        offdiag_global_to_local[entry.first] = idx;
        ++recv_sizes[part];
    }

    // build local-to-global map for offdiag columns
    local_to_global_ghost.resize_and_reset(num_ghost_elems);
    std::fill_n(local_to_global_ghost.get_data(),
                local_to_global_ghost.get_num_elems(),
                invalid_index<GlobalIndexType>());
    for (const auto& key_value : offdiag_global_to_local) {
        const auto global_idx = key_value.first;
        const auto local_idx = key_value.second;
        local_to_global_ghost.get_data()[local_idx] = global_idx;
    }

    // shift recv_sizes to the back, insert 0 in front again
    LocalIndexType local_prev = num_parts ? recv_sizes[0] : 0;
    for (size_type i = num_parts - 1; i > 0; --i) {
        recv_sizes[i] -= recv_sizes[i - 1];
    }

    // map off-diag values to local column indices
    offdiag_row_idxs.resize_and_reset(global_offdiag_entries.size());
    offdiag_col_idxs.resize_and_reset(global_offdiag_entries.size());
    offdiag_values.resize_and_reset(global_offdiag_entries.size());
#pragma omp parallel for
    for (size_type i = 0; i < global_offdiag_entries.size(); i++) {
        auto global = global_offdiag_entries[i];
        offdiag_row_idxs.get_data()[i] =
            static_cast<LocalIndexType>(global.row);
        offdiag_col_idxs.get_data()[i] = offdiag_global_to_local[global.column];
        offdiag_values.get_data()[i] = global.value;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_DIAG_OFFDIAG);


}  // namespace distributed_matrix
}  // namespace omp
}  // namespace kernels
}  // namespace gko
