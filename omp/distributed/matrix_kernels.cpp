/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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
#include "core/components/prefix_sum.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace distributed_matrix {


template <typename ValueType, typename LocalIndexType>
void build_diag_offdiag(
    std::shared_ptr<const DefaultExecutor> exec,
    const Array<matrix_data_entry<ValueType, global_index_type>> &input,
    const distributed::Partition<LocalIndexType> *partition,
    comm_index_type local_part,
    Array<matrix_data_entry<ValueType, LocalIndexType>> &diag_data,
    Array<matrix_data_entry<ValueType, LocalIndexType>> &offdiag_data,
    Array<LocalIndexType> &local_gather_idxs, comm_index_type *recv_offsets,
    ValueType deduction_help)
{
    using range_index_type = global_index_type;
    using part_index_type = comm_index_type;
    using global_nonzero = matrix_data_entry<ValueType, global_index_type>;
    using local_nonzero = matrix_data_entry<ValueType, LocalIndexType>;
    using local_index_type = LocalIndexType;
    auto input_data = input.get_const_data();
    auto range_bounds = partition->get_const_range_bounds();
    auto range_parts = partition->get_const_part_ids();
    auto range_ranks = partition->get_range_ranks();
    auto num_parts = partition->get_num_parts();
    auto num_ranges = partition->get_num_ranges();
    // zero recv_offsets values
    std::fill_n(recv_offsets, num_parts + 1, comm_index_type{});

    // helpers for retrieving range info
    struct range_info {
        range_index_type index{};
        global_index_type begin{};
        global_index_type end{};
        local_index_type base_rank{};
        part_index_type part{};
    };
    auto find_range = [&](global_index_type idx) {
        auto it = std::upper_bound(range_bounds + 1,
                                   range_bounds + num_ranges + 1, idx);
        return std::distance(range_bounds + 1, it);
    };
    auto update_range = [&](global_index_type idx, range_info &info) {
        if (idx < info.begin || idx >= info.end) {
            info.index = find_range(idx);
            info.begin = range_bounds[info.index];
            info.end = range_bounds[info.index + 1];
            info.base_rank = range_ranks[info.index];
            info.part = range_parts[info.index];
            // assert(info.index < num_ranges);
        }
        // assert(idx >= info.begin && idx < info.end);
    };
    auto map_to_local = [&](global_index_type idx,
                            range_info info) -> local_index_type {
        return static_cast<local_index_type>(idx - info.begin) + info.base_rank;
    };

    // store offdiagonal columns and their range indices
    unordered_map<global_index_type, range_index_type> offdiag_cols{{exec}};
    // store offdiagonal entries with global column idxs
    vector<global_nonzero> global_offdiag_entries{{exec}};
    vector<local_nonzero> diag_entries{{exec}};

    auto num_threads = static_cast<size_type>(omp_get_max_threads());
    auto num_input = input.get_num_elems();
    auto size_per_thread = (num_input + num_threads - 1) / num_threads;
    vector<size_type> diag_entry_offsets{num_threads, 0, {exec}};
    vector<size_type> offdiag_entry_offsets{num_threads, 0, {exec}};

#pragma omp parallel
    {
        range_info row_range{};
        range_info col_range{};
        unordered_map<global_index_type, range_index_type> thread_offdiag_cols{
            {exec}};
        vector<global_nonzero> thread_offdiag_entries{{exec}};
        vector<local_nonzero> thread_diag_entries{{exec}};
        auto thread_id = omp_get_thread_num();
        auto thread_begin = thread_id * size_per_thread;
        auto thread_end = std::min(thread_begin + size_per_thread, num_input);
        // separate diagonal and off-diagonal entries for our input chunk
        for (auto i = thread_begin; i < thread_end; ++i) {
            auto entry = input_data[i];
            update_range(entry.row, row_range);
            // skip non-local rows
            if (row_range.part != local_part) {
                continue;
            }
            // map to part-local indices
            auto local_row = map_to_local(entry.row, row_range);
            update_range(entry.column, col_range);
            if (col_range.part == local_part) {
                // store diagonal entry
                auto local_col = map_to_local(entry.column, col_range);
                thread_diag_entries.emplace_back(local_row, local_col,
                                                 entry.value);
            } else {
                // store off-diagonal entry
                thread_offdiag_cols.emplace(entry.column, col_range.index);
                thread_offdiag_entries.emplace_back(local_row, entry.column,
                                                    entry.value);
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
        for (auto &entry : thread_diag_entries) {
            diag_entries[diag] = entry;
            diag++;
        }
        for (auto &entry : thread_offdiag_entries) {
            global_offdiag_entries[offdiag] = entry;
            offdiag++;
        }
    }
    // store diagonal data to output
    {
        auto diag_view = Array<local_nonzero>::view(exec, diag_entries.size(),
                                                    diag_entries.data());
        diag_data = diag_view;
    }
    // count off-diagonal columns per part
    // TODO
    // build recv_offsets
    components::prefix_sum(exec, recv_offsets, num_parts + 1);
    local_gather_idxs.resize_and_reset(recv_offsets[num_parts]);
    unordered_map<global_index_type, LocalIndexType> offdiag_global_to_local{
        {exec}};
    // collect and renumber offdiagonal columns
    for (auto entry : offdiag_cols) {
        auto range = entry.second;
        auto range_begin = range_bounds[range];
        auto range_rank = range_ranks[range];
        auto part = range_parts[range];
        auto idx = recv_offsets[part];
        local_gather_idxs.get_data()[idx] = static_cast<comm_index_type>(
            entry.first - range_begin + range_rank);
        offdiag_global_to_local[entry.first] = idx;
        ++recv_offsets[part];
    }
    // shift recv_offsets to the back, insert 0 in front again
    LocalIndexType local_prev{};
    for (size_type i = 0; i <= num_parts; i++) {
        recv_offsets[i] = std::exchange(local_prev, recv_offsets[i]);
    }
    // map off-diag values to local column indices
    offdiag_data.resize_and_reset(global_offdiag_entries.size());
#pragma omp for
    for (size_type i = 0; i < global_offdiag_entries.size(); i++) {
        auto global = global_offdiag_entries[i];
        offdiag_data.get_data()[i] = {static_cast<LocalIndexType>(global.row),
                                      offdiag_global_to_local[global.column],
                                      global.value};
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BUILD_DIAG_OFFDIAG);


}  // namespace distributed_matrix
}  // namespace omp
}  // namespace kernels
}  // namespace gko