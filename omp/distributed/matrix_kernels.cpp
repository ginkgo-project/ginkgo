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


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/base/allocator.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace distributed_matrix {


/**
 * Maps indices into the compact range [0, N), where N is the number of unique
 * indices. Also reorders the input keys and indices,
 *
 * The comp and unq parameters can be used to group the indices. The comp gives
 * the ordering between to indices, and unq checks if two indices are equal.
 *
 * Consider the following example using default comparisons:
 * ```
 * comp = std::less<>;
 * unq = std::equal_to<>;
 * I = [3, 2, 7, 7]
 * ```
 * then the output iterator will hold:
 * ```
 * O = [1, 0, 2, 2]
 * ```
 */
template <typename IndexIt, typename OutputIt, typename Compare,
          typename Unique>
std::tuple<IndexIt, OutputIt> compress_indices(IndexIt indices_first,
                                               IndexIt indices_last,
                                               OutputIt out, Compare&& comp,
                                               Unique&& unq)
{
    using index_type = typename std::iterator_traits<IndexIt>::value_type;
    using out_index_type = typename std::iterator_traits<OutputIt>::value_type;

    auto size = std::distance(indices_first, indices_last);

    std::vector<index_type> original_indices(size);
#pragma omp parallel for
    for (size_type i = 0; i < size; ++i) {
        original_indices[i] = *(indices_first + i);
    }

    std::sort(indices_first, indices_last, comp);
    auto unique_indices_end = std::unique(indices_first, indices_last, unq);

#pragma omp parallel for
    for (size_type i = 0; i < size; ++i) {
        auto iit = original_indices.begin() + i;
        auto oit = out + i;
        auto segment_begin =
            std::lower_bound(indices_first, unique_indices_end, *iit, comp);
        auto segment_end =
            std::upper_bound(indices_first, unique_indices_end, *iit, comp);

        *oit = static_cast<out_index_type>(std::distance(
            indices_first, std::lower_bound(segment_begin, segment_end, *iit)));
    }

    return std::make_tuple(unique_indices_end, out + size);
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
    using range_index_type = GlobalIndexType;
    using global_nonzero = matrix_data_entry<ValueType, GlobalIndexType>;
    using local_nonzero = matrix_data_entry<ValueType, LocalIndexType>;
    auto input_row_idxs = input.get_const_row_idxs();
    auto input_col_idxs = input.get_const_col_idxs();
    auto input_vals = input.get_const_values();
    auto row_part_ids = row_partition->get_part_ids();
    auto col_part_ids = col_partition->get_part_ids();
    auto num_parts = row_partition->get_num_parts();
    auto recv_sizes_ptr = recv_sizes.get_data();
    size_type row_range_id_hint = 0;
    size_type col_range_id_hint = 0;
    // zero recv_sizes values
    std::fill_n(recv_sizes_ptr, num_parts, comm_index_type{});

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

    // store non-local entries with global column idxs
    vector<global_nonzero> non_local_entries(exec);
    vector<local_nonzero> local_entries(exec);

    auto num_threads = static_cast<size_type>(omp_get_max_threads());
    auto num_input = input.get_num_elems();
    auto size_per_thread = (num_input + num_threads - 1) / num_threads;
    std::vector<size_type> local_entry_offsets(num_threads, 0);
    std::vector<size_type> non_local_entry_offsets(num_threads, 0);

#pragma omp parallel firstprivate(col_range_id_hint, row_range_id_hint)
    {
        std::vector<global_nonzero> thread_non_local_entries;
        std::vector<local_nonzero> thread_local_entries;
        std::vector<comm_index_type> thread_recv_sizes;
        auto thread_id = omp_get_thread_num();
        auto thread_begin = thread_id * size_per_thread;
        auto thread_end = std::min(thread_begin + size_per_thread, num_input);
        // separate local and non-local entries for our input chunk
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
                    // store local entry
                    auto local_col =
                        map_to_local(global_col, col_partition, col_range_id);
                    thread_local_entries.emplace_back(local_row, local_col,
                                                      value);
                } else {
                    thread_non_local_entries.emplace_back(local_row, global_col,
                                                          value);
                }
            }
        }
        local_entry_offsets[thread_id] = thread_local_entries.size();
        non_local_entry_offsets[thread_id] = thread_non_local_entries.size();

#pragma omp barrier
#pragma omp single
        {
            // assign output ranges to the individual threads
            size_type local{};
            size_type non_local{};
            for (size_type thread = 0; thread < num_threads; ++thread) {
                auto size_local = local_entry_offsets[thread];
                auto size_non_local = non_local_entry_offsets[thread];
                local_entry_offsets[thread] = local;
                non_local_entry_offsets[thread] = non_local;
                local += size_local;
                non_local += size_non_local;
            }
            local_entries.resize(local);
            non_local_entries.resize(non_local);
        }
        // write back the local data to the output ranges
        auto local = local_entry_offsets[thread_id];
        auto non_local = non_local_entry_offsets[thread_id];
        for (const auto& entry : thread_local_entries) {
            local_entries[local] = entry;
            local++;
        }
        for (const auto& entry : thread_non_local_entries) {
            non_local_entries[non_local] = entry;
            non_local++;
        }
    }
    // store local data to output
    local_row_idxs.resize_and_reset(local_entries.size());
    local_col_idxs.resize_and_reset(local_entries.size());
    local_values.resize_and_reset(local_entries.size());
#pragma omp parallel for
    for (size_type i = 0; i < local_entries.size(); ++i) {
        const auto& entry = local_entries[i];
        local_row_idxs.get_data()[i] = entry.row;
        local_col_idxs.get_data()[i] = entry.column;
        local_values.get_data()[i] = entry.value;
    }

    // store non-local row, values to output
    non_local_row_idxs.resize_and_reset(non_local_entries.size());
    non_local_col_idxs.resize_and_reset(non_local_entries.size());
    non_local_values.resize_and_reset(non_local_entries.size());
    std::vector<GlobalIndexType> unique_columns(non_local_entries.size());
#pragma omp parallel for
    for (size_type i = 0; i < non_local_entries.size(); i++) {
        auto global = non_local_entries[i];
        non_local_row_idxs.get_data()[i] =
            static_cast<LocalIndexType>(global.row);
        non_local_values.get_data()[i] = global.value;
        unique_columns[i] = global.column;
    }

    // map non-local global column indices into compresses column index space
    auto find_col_part = [&](GlobalIndexType idx) {
        auto range_id = find_range(idx, col_partition, 0);
        return col_part_ids[range_id];
    };
    auto compress_result = compress_indices(
        unique_columns.begin(), unique_columns.end(),
        non_local_col_idxs.get_data(),
        [&](const auto& a, const auto& b) {
            auto part_a = find_col_part(a);
            auto part_b = find_col_part(b);
            return std::tie(part_a, a) < std::tie(part_b, b);
        },
        [&](const auto& a, const auto& b) {
            auto part_a = find_col_part(a);
            auto part_b = find_col_part(b);
            return std::tie(part_a, a) == std::tie(part_b, b);
        });
    auto unique_columns_end = std::get<0>(compress_result);
    unique_columns.erase(unique_columns_end, unique_columns.end());

    // copy unique_columns to array
    non_local_to_global = array<GlobalIndexType>{exec, unique_columns.begin(),
                                                 unique_columns.end()};

    // compute gather idxs and recv_sizes
    local_gather_idxs.resize_and_reset(unique_columns.size());
    std::fill_n(recv_sizes.get_data(), num_parts, 0);
    size_type col_range_id = 0;
#pragma omp parallel for firstprivate(col_range_id)
    for (size_type i = 0; i < unique_columns.size(); ++i) {
        col_range_id =
            find_range(unique_columns[i], col_partition, col_range_id);
        local_gather_idxs.get_data()[i] =
            map_to_local(unique_columns[i], col_partition, col_range_id);
        auto col_part = find_col_part(unique_columns[i]);
#pragma omp atomic
        recv_sizes.get_data()[col_part]++;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_LOCAL_NONLOCAL);


}  // namespace distributed_matrix
}  // namespace omp
}  // namespace kernels
}  // namespace gko
