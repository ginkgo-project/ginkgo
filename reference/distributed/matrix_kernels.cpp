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


#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/csr_kernels.hpp"

namespace gko {
namespace kernels {
namespace reference {
namespace distributed_matrix {


template <typename ValueType, typename LocalIndexType>
void build_diag_offdiag(
    std::shared_ptr<const DefaultExecutor> exec,
    const Array<matrix_data_entry<ValueType, global_index_type>>& input,
    const distributed::Partition<LocalIndexType>* partition,
    comm_index_type local_part,
    Array<matrix_data_entry<ValueType, LocalIndexType>>& diag_data,
    Array<matrix_data_entry<ValueType, LocalIndexType>>& offdiag_data,
    Array<LocalIndexType>& local_gather_idxs,
    comm_index_type* recv_offsets,  // why not pass as array
    Array<global_index_type>& local_row_to_global,
    Array<global_index_type>& local_offdiag_col_to_global,
    ValueType deduction_help)
{
    using range_index_type = global_index_type;
    using part_index_type = comm_index_type;
    using global_nonzero = matrix_data_entry<ValueType, global_index_type>;
    using local_nonzero = matrix_data_entry<ValueType, LocalIndexType>;
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
        LocalIndexType base_rank{};
        part_index_type part{};
    };
    auto find_range = [&](global_index_type idx) {
        auto it = std::upper_bound(range_bounds + 1,
                                   range_bounds + num_ranges + 1, idx);
        return std::distance(range_bounds + 1, it);
    };
    auto update_range = [&](global_index_type idx, range_info& info) {
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
                            range_info info) -> LocalIndexType {
        return static_cast<LocalIndexType>(idx - info.begin) + info.base_rank;
    };

    local_row_to_global.resize_and_reset(partition->get_part_size(local_part));
    std::fill_n(local_row_to_global.get_data(),
                local_row_to_global.get_num_elems(), -1);

    range_info row_range{};
    range_info col_range{};
    // store offdiagonal columns and their range indices
    map<global_index_type, range_index_type> offdiag_cols{{exec}};
    // store offdiagonal entries with global column idxs
    vector<global_nonzero> global_offdiag_entries{{exec}};
    vector<local_nonzero> diag_entries{{exec}};
    for (size_type i = 0; i < input.get_num_elems(); ++i) {
        auto entry = input_data[i];
        update_range(entry.row, row_range);
        // skip non-local rows
        if (row_range.part != local_part) {
            continue;
        }
        // map to part-local indices
        update_range(entry.column, col_range);
        auto local_row = map_to_local(entry.row, row_range);
        local_row_to_global.get_data()[local_row] = entry.row;
        if (col_range.part == local_part) {
            // store diagonal entry
            auto local_col = map_to_local(entry.column, col_range);
            diag_entries.emplace_back(local_row, local_col, entry.value);
        } else {
            // store off-diagonal entry
            auto new_col =
                offdiag_cols.emplace(entry.column, col_range.index).second;
            // count the number of off-diagonal entries in each part
            if (new_col) {
                recv_offsets[col_range.part]++;
            }
            global_offdiag_entries.emplace_back(local_row, entry.column,
                                                entry.value);
        }
    }
    // store diagonal data to output
    {
        auto diag_view = Array<local_nonzero>::view(exec, diag_entries.size(),
                                                    diag_entries.data());
        diag_data = diag_view;
    }
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
    // build local-to-global map for offdiag columns
    local_offdiag_col_to_global.resize_and_reset(
        local_gather_idxs.get_num_elems());
    std::fill_n(local_offdiag_col_to_global.get_data(),
                local_offdiag_col_to_global.get_num_elems(), -1);
    for (const auto& key_value : offdiag_global_to_local) {
        const auto global_idx = key_value.first;
        const auto local_idx = key_value.second;
        local_offdiag_col_to_global.get_data()[local_idx] = global_idx;
    }
    // shift recv_offsets to the back, insert 0 in front again
    LocalIndexType local_prev{};
    for (size_type i = 0; i <= num_parts; i++) {
        recv_offsets[i] = std::exchange(local_prev, recv_offsets[i]);
    }
    // map off-diag values to local column indices
    offdiag_data.resize_and_reset(global_offdiag_entries.size());
    for (size_type i = 0; i < global_offdiag_entries.size(); i++) {
        auto global = global_offdiag_entries[i];
        offdiag_data.get_data()[i] = {static_cast<LocalIndexType>(global.row),
                                      offdiag_global_to_local[global.column],
                                      global.value};
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BUILD_DIAG_OFFDIAG);


template <typename SourceType, typename TargetType>
void map_to_global_idxs(std::shared_ptr<const DefaultExecutor> exec,
                        const SourceType* input, size_t n, TargetType* output,
                        const TargetType* map)
{
    std::transform(input, input + n, output,
                   [&](const auto& idx) { return map[idx]; });
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_MAP_TO_GLOBAL_IDXS);


template <typename FirstIt, typename SecondIt, typename OutputIt>
void merge_sorted(FirstIt begin_first, FirstIt end_first, SecondIt begin_second,
                  SecondIt end_second, OutputIt out)
{
    FirstIt it_first = begin_first;
    SecondIt it_second = begin_second;

    while (it_first != end_first || it_second != end_second) {
        if (it_first == end_first)
            *out++ = *it_second++;
        else if (it_second == end_second)
            *out++ = *it_first++;
        else
            *out++ = *it_first < *it_second ? *it_first++ : *it_second++;
    }
}


template <typename ValueType, typename LocalIndexType>
void merge_diag_offdiag(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, LocalIndexType>* diag,
                        const matrix::Csr<ValueType, LocalIndexType>* offdiag,
                        matrix::Csr<ValueType, LocalIndexType>* result)
{
    auto num_rows = result->get_size()[0];

    auto* diag_row_ptrs = diag->get_const_row_ptrs();
    auto* diag_col_idxs = diag->get_const_col_idxs();
    auto* diag_values = diag->get_const_values();

    auto* offdiag_row_ptrs = offdiag->get_const_row_ptrs();
    auto* offdiag_col_idxs = offdiag->get_const_col_idxs();
    auto* offdiag_values = offdiag->get_const_values();

    auto* local_row_ptrs = result->get_row_ptrs();
    auto* local_col_idxs = result->get_col_idxs();
    auto* local_values = result->get_values();

    local_row_ptrs[0] = 0;
    for (global_index_type i = 0; i < num_rows; ++i) {
        auto diag_offset = diag_row_ptrs[i];
        auto offdiag_offset = offdiag_row_ptrs[i];
        auto local_offset = local_row_ptrs[i];

        auto diag_nnz_in_row = diag_row_ptrs[i + 1] - diag_offset;
        auto offdiag_nnz_in_row = offdiag_row_ptrs[i + 1] - offdiag_offset;

        detail::IteratorFactory<LocalIndexType, ValueType> diag_factory{
            const_cast<LocalIndexType*>(diag_col_idxs) + diag_offset,
            const_cast<ValueType*>(diag_values) + diag_offset,
            static_cast<size_type>(diag_nnz_in_row)};
        detail::IteratorFactory<LocalIndexType, ValueType> offdiag_factory{
            const_cast<LocalIndexType*>(offdiag_col_idxs) + offdiag_offset,
            const_cast<ValueType*>(offdiag_values) + offdiag_offset,
            static_cast<size_type>(offdiag_nnz_in_row)};

        local_row_ptrs[i + 1] =
            local_row_ptrs[i] + diag_nnz_in_row + offdiag_nnz_in_row;
        detail::IteratorFactory<LocalIndexType, ValueType> local_factory{
            local_col_idxs + local_offset, local_values + local_offset,
            static_cast<size_type>(diag_nnz_in_row + offdiag_nnz_in_row)};

        merge_sorted(diag_factory.begin(), diag_factory.end(),
                     offdiag_factory.begin(), offdiag_factory.end(),
                     local_factory.begin());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_MERGE_DIAG_OFFDIAG);

template <typename ValueType, typename LocalIndexType>
void combine_local_mtxs(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, LocalIndexType>* local,
                        matrix::Csr<ValueType, LocalIndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COMBINE_LOCAL_MTXS);


}  // namespace distributed_matrix
}  // namespace reference
}  // namespace kernels
}  // namespace gko
